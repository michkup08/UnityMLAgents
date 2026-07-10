from typing import Dict, cast
import attr
import copy

from mlagents.torch_utils import torch, default_device

from mlagents.trainers.buffer import AgentBuffer, BufferKey, RewardSignalUtil

from mlagents_envs.timers import timed
from mlagents.trainers.policy.torch_policy import TorchPolicy
from mlagents.trainers.optimizer.torch_optimizer import TorchOptimizer
from mlagents.trainers.settings import (
    TrainerSettings,
    OnPolicyHyperparamSettings,
    ScheduleType,
)
from mlagents.trainers.torch_entities.networks import ValueNetwork
from mlagents.trainers.torch_entities.agent_action import AgentAction
from mlagents.trainers.torch_entities.action_log_probs import ActionLogProbs
from mlagents.trainers.torch_entities.utils import ModelUtils
from mlagents.trainers.trajectory import ObsUtil


@attr.s(auto_attribs=True)
class PPGSettings(OnPolicyHyperparamSettings):
    beta: float = 5.0e-3
    epsilon: float = 0.2
    lambd: float = 0.95
    num_epoch: int = 3
    learning_rate_schedule: ScheduleType = ScheduleType.LINEAR
    beta_schedule: ScheduleType = ScheduleType.LINEAR
    epsilon_schedule: ScheduleType = ScheduleType.LINEAR

    num_policy_updates_per_aux: int = 16  # co ile aktualizacji polityki odpalamy fazę Aux
    aux_epochs: int = 6 # ile epok w fazie aux
    kl_penalty_coef: float = 1.0 # kara za zmianę polityki w fazie aux (beta_clone)
    shared_critic: bool = False


class TorchPPGOptimizer(TorchOptimizer):
    def __init__(self, policy: TorchPolicy, trainer_settings: TrainerSettings):

        super().__init__(policy, trainer_settings)
        reward_signal_configs = trainer_settings.reward_signals
        reward_signal_names = [key.value for key, _ in reward_signal_configs.items()]

        self.hyperparameters: PPGSettings = cast(
            PPGSettings, trainer_settings.hyperparameters
        )

        if self.hyperparameters.shared_critic:
            self._critic = policy.actor
        else:
            self._critic = ValueNetwork(
                reward_signal_names,
                policy.behavior_spec.observation_specs,
                network_settings=trainer_settings.network_settings,
            )
            self._critic.to(default_device())

        self.decay_learning_rate = ModelUtils.DecayedValue(
            self.hyperparameters.learning_rate_schedule,
            self.hyperparameters.learning_rate,
            1e-10,
            self.trainer_settings.max_steps,
        )
        self.decay_epsilon = ModelUtils.DecayedValue(
            self.hyperparameters.epsilon_schedule,
            self.hyperparameters.epsilon,
            0.1,
            self.trainer_settings.max_steps,
        )
        self.decay_beta = ModelUtils.DecayedValue(
            self.hyperparameters.beta_schedule,
            self.hyperparameters.beta,
            1e-5,
            self.trainer_settings.max_steps,
        )
        
        # parametry dla fazy aux (zawsze Aktor + opcjonalnie Krytyk, jeśli nie jest współdzielony)
        aux_params = list(self.policy.actor.parameters())
        if not self.hyperparameters.shared_critic:
            aux_params += list(self._critic.parameters())

        self.aux_optimizer = torch.optim.Adam(
            aux_params, lr=self.trainer_settings.hyperparameters.learning_rate
        )
        self.stats_name_to_update_name = {
            "Losses/Value Loss": "value_loss",
            "Losses/Policy Loss": "policy_loss",
        }

        self.stream_names = list(self.reward_signals.keys())

        phase1_params = list(self.policy.actor.parameters())
        if not self.hyperparameters.shared_critic:
            # parametry niezależnego krytyka dodane do optymalizatora fazy 1
            phase1_params += list(self._critic.parameters())

        self.actor_optimizer = torch.optim.Adam(
            phase1_params, lr=self.trainer_settings.hyperparameters.learning_rate
        )

        # inicjalizacja atrybutu dla zamrożonego aktora
        self.frozen_actor = None

    @property
    def critic(self):
        return self._critic

    @timed
    def update_policy(self, batch: AgentBuffer, num_sequences: int) -> Dict[str, float]:
        """
        FAZA 1: Aktualizacja samej polityki z bonusem za entropię.
        Brak wpływu na krytyka/funkcję wartości.
        """
        decay_lr = self.decay_learning_rate.get_value(self.policy.get_current_step())
        decay_eps = self.decay_epsilon.get_value(self.policy.get_current_step())
        decay_bet = self.decay_beta.get_value(self.policy.get_current_step())

        returns = {}
        old_values = {}
        for name in self.reward_signals:
            old_values[name] = ModelUtils.list_to_tensor(
                batch[RewardSignalUtil.value_estimates_key(name)]
            )
            returns[name] = ModelUtils.list_to_tensor(
                batch[RewardSignalUtil.returns_key(name)]
            )

        n_obs = len(self.policy.behavior_spec.observation_specs)
        current_obs = ObsUtil.from_buffer(batch, n_obs)
        current_obs = [ModelUtils.list_to_tensor(obs) for obs in current_obs]

        act_masks = ModelUtils.list_to_tensor(batch[BufferKey.ACTION_MASK])
        actions = AgentAction.from_buffer(batch)

        memories = [
            ModelUtils.list_to_tensor(batch[BufferKey.MEMORY][i])
            for i in range(0, len(batch[BufferKey.MEMORY]), self.policy.sequence_length)
        ]
        if len(memories) > 0:
            memories = torch.stack(memories).unsqueeze(0)

        value_memories = [
            ModelUtils.list_to_tensor(batch[BufferKey.CRITIC_MEMORY][i])
            for i in range(0, len(batch[BufferKey.CRITIC_MEMORY]), self.policy.sequence_length)
        ]
        if len(value_memories) > 0:
            value_memories = torch.stack(value_memories).unsqueeze(0)

        # puszczenie danych przez Aktora
        run_out = self.policy.actor.get_stats(
            current_obs,
            actions,
            masks=act_masks,
            memories=memories,
            sequence_length=self.policy.sequence_length,
        )

        log_probs = run_out["log_probs"].flatten()
        entropy = run_out["entropy"]
        old_log_probs = ActionLogProbs.from_buffer(batch).flatten()
        loss_masks = ModelUtils.list_to_tensor(batch[BufferKey.MASKS], dtype=torch.bool)

        # standardowy loss z PPO
        policy_loss = ModelUtils.trust_region_policy_loss(
            ModelUtils.list_to_tensor(batch[BufferKey.ADVANTAGES]),
            log_probs,
            old_log_probs,
            loss_masks,
            decay_eps,
        )
        
        values, _ = self.critic.critic_pass(
            current_obs,
            memories=value_memories,
            sequence_length=self.policy.sequence_length,
        )
        
        value_loss = ModelUtils.trust_region_value_loss(
            values, old_values, returns, decay_eps, loss_masks
        )

        # łączony błąd do optymalizacji
        loss = (
            policy_loss 
            + 0.5 * value_loss 
            - decay_bet * ModelUtils.masked_mean(entropy, loss_masks)
        )

        ModelUtils.update_learning_rate(self.actor_optimizer, decay_lr)
        self.actor_optimizer.zero_grad()
        loss.backward()
        self.actor_optimizer.step()

        return {
            "Losses/Policy Loss": torch.abs(policy_loss).item(),
            "Losses/Value Loss": value_loss.item(),
            "Policy/Learning Rate": decay_lr,
            "Policy/Epsilon": decay_eps,
            "Policy/Beta": decay_bet,
        }

    def prepare_aux_phase(self) -> None:
        """
        Tworzy głęboką kopię (zamraża) aktora z końca Fazy 1.
        Posłuży on jako kotwica (Target Network) dla Behavioral Cloning w Fazie 2,
        całkowicie omijając potrzebę manipulowania gigantycznym buforem.
        """
        self.frozen_actor = copy.deepcopy(self.policy.actor)
        self.frozen_actor.eval()

    @timed
    def update_auxiliary(self, batch: AgentBuffer, num_sequences: int) -> Dict[str, float]:
        """
        FAZA 2: Faza pomocnicza (Auxiliary).
        Uczenie krytyka z równoczesną karą za zmianę oryginalnej polityki (KL Divergence).
        """
        decay_lr = self.decay_learning_rate.get_value(self.policy.get_current_step())
        decay_eps = self.decay_epsilon.get_value(self.policy.get_current_step())

        n_obs = len(self.policy.behavior_spec.observation_specs)
        current_obs = ObsUtil.from_buffer(batch, n_obs)
        current_obs = [ModelUtils.list_to_tensor(obs) for obs in current_obs]

        # pamięć dla aktora
        memories = [
            ModelUtils.list_to_tensor(batch[BufferKey.MEMORY][i])
            for i in range(0, len(batch[BufferKey.MEMORY]), self.policy.sequence_length)
        ]
        if len(memories) > 0:
            memories = torch.stack(memories).unsqueeze(0)

        # pamięć dla krytyka
        value_memories = [
            ModelUtils.list_to_tensor(batch[BufferKey.CRITIC_MEMORY][i])
            for i in range(0, len(batch[BufferKey.CRITIC_MEMORY]), self.policy.sequence_length)
        ]
        if len(value_memories) > 0:
            value_memories = torch.stack(value_memories).unsqueeze(0)

        with torch.no_grad():
            fresh_values, _ = self.critic.critic_pass(
                current_obs,
                memories=value_memories,
                sequence_length=self.policy.sequence_length,
            )

        returns = {}
        old_values = {}
        for name in self.reward_signals:
            old_values[name] = ModelUtils.list_to_tensor(
                batch[RewardSignalUtil.value_estimates_key(name)]
            )
            advantages = ModelUtils.list_to_tensor(
                batch[RewardSignalUtil.advantage_key(name)]
            )
            # return = stare GAE + nowa, lepsza funkcja wartości
            returns[name] = advantages + fresh_values[name]

        act_masks = ModelUtils.list_to_tensor(batch[BufferKey.ACTION_MASK])
        actions = AgentAction.from_buffer(batch)

        # bieżące prawdopodobieństwa z uczącego się aktora
        run_out = self.policy.actor.get_stats(
            current_obs,
            actions,
            masks=act_masks,
            memories=memories,
            sequence_length=self.policy.sequence_length,
        )
        log_probs = run_out["log_probs"].flatten()
        
        # 2. "stare" prawdopodobieństwa z zamrożonego aktora
        with torch.no_grad():
            frozen_run_out = self.frozen_actor.get_stats(
                current_obs,
                actions,
                masks=act_masks,
                memories=memories,
                sequence_length=self.policy.sequence_length,
            )
            old_log_probs = frozen_run_out["log_probs"].flatten()
        
        # wartości krytyka
        values, _ = self.critic.critic_pass(
            current_obs,
            memories=value_memories,
            sequence_length=self.policy.sequence_length,
        )
        loss_masks = ModelUtils.list_to_tensor(batch[BufferKey.MASKS], dtype=torch.bool)

        # obliczanie value loss
        value_loss = ModelUtils.trust_region_value_loss(
            values, old_values, returns, decay_eps, loss_masks
        )

        # aproksymacja dywergencji KL dla bezpieczeństwa polityki
        ratio = torch.exp(log_probs - old_log_probs)
        kl_div = torch.mean(ratio - 1.0 - torch.log(ratio + 1e-8))

        # łączony błąd fazy aux
        aux_loss = value_loss + (self.hyperparameters.kl_penalty_coef * kl_div)

        ModelUtils.update_learning_rate(self.aux_optimizer, decay_lr)
        self.aux_optimizer.zero_grad()
        aux_loss.backward()
        self.aux_optimizer.step()

        return {
            "Losses/Value Loss": value_loss.item(),
            "Losses/KL Penalty": kl_div.item(),
            "Losses/Aux Total Loss": aux_loss.item(),
            "Policy/Learning Rate": decay_lr,
        }

    def get_modules(self):
        modules = {
            "Optimizer:actor_optimizer": self.actor_optimizer,
            "Optimizer:aux_optimizer": self.aux_optimizer,
            "Optimizer:critic": self._critic,
        }
        for reward_provider in self.reward_signals.values():
            modules.update(reward_provider.get_modules())
        return modules