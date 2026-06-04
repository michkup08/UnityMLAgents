from typing import cast, Type, Union, Dict, Any

import numpy as np

from mlagents_envs.base_env import BehaviorSpec
from mlagents_envs.logging_util import get_logger
from mlagents.trainers.buffer import BufferKey, RewardSignalUtil, AgentBuffer
from mlagents.trainers.trainer.on_policy_trainer import OnPolicyTrainer
from mlagents.trainers.policy.policy import Policy
from mlagents.trainers.trainer.trainer_utils import get_gae
from mlagents.trainers.optimizer.torch_optimizer import TorchOptimizer
from mlagents.trainers.policy.torch_policy import TorchPolicy
from mlagents.trainers.trajectory import Trajectory
from mlagents.trainers.behavior_id_utils import BehaviorIdentifiers
from mlagents.trainers.settings import TrainerSettings

from mlagents.trainers.torch_entities.networks import SimpleActor, SharedActorCritic

# Zaimportowanie naszego optymalizatora PPG i ustawień
from mlagents.trainers.ppg.optimizer_torch_ppg import TorchPPGOptimizer, PPGSettings

logger = get_logger(__name__)

TRAINER_NAME = "ppg" 


class PPGTrainer(OnPolicyTrainer):
    """The PPGTrainer is an implementation of the Phasic Policy Gradient algorithm."""

    def __init__(
        self,
        behavior_name: str,
        reward_buff_cap: int,
        trainer_settings: TrainerSettings,
        training: bool,
        load: bool,
        seed: int,
        artifact_path: str,
    ):
        super().__init__(
            behavior_name,
            reward_buff_cap,
            trainer_settings,
            training,
            load,
            seed,
            artifact_path,
        )
        self.hyperparameters: PPGSettings = cast(
            PPGSettings, self.trainer_settings.hyperparameters
        )
        self.seed = seed
        self.shared_critic = self.hyperparameters.shared_critic
        self.policy: TorchPolicy = None  # type: ignore

        # Bufor pomocniczy (Auxiliary Buffer), który zbiera dane z kilku faz polityki
        self.aux_buffer = AgentBuffer()
        # Licznik aktualizacji polityki, potrzebny do odpalenia fazy pomocniczej
        self.policy_update_count = 0

    # ==============================================================================
    # --- PRZYWRÓCONA FUNKCJA (Wymagana przez architekturę ML-Agents) ---
    # ==============================================================================
    def _process_trajectory(self, trajectory: Trajectory) -> None:
        """
        Takes a trajectory and processes it, putting it into the update buffer.
        Processing involves calculating value and advantage targets for model updating step.
        """
        super()._process_trajectory(trajectory)
        agent_id = trajectory.agent_id

        agent_buffer_trajectory = trajectory.to_agentbuffer()
        self._warn_if_group_reward(agent_buffer_trajectory)

        if self.is_training:
            self.policy.actor.update_normalization(agent_buffer_trajectory)
            self.optimizer.critic.update_normalization(agent_buffer_trajectory)

        (
            value_estimates,
            value_next,
            value_memories,
        ) = self.optimizer.get_trajectory_value_estimates(
            agent_buffer_trajectory,
            trajectory.next_obs,
            trajectory.done_reached and not trajectory.interrupted,
        )
        if value_memories is not None:
            agent_buffer_trajectory[BufferKey.CRITIC_MEMORY].set(value_memories)

        for name, v in value_estimates.items():
            agent_buffer_trajectory[RewardSignalUtil.value_estimates_key(name)].extend(v)
            self._stats_reporter.add_stat(
                f"Policy/{self.optimizer.reward_signals[name].name.capitalize()} Value Estimate",
                np.mean(v),
            )

        self.collected_rewards["environment"][agent_id] += np.sum(
            agent_buffer_trajectory[BufferKey.ENVIRONMENT_REWARDS]
        )
        for name, reward_signal in self.optimizer.reward_signals.items():
            evaluate_result = (
                reward_signal.evaluate(agent_buffer_trajectory) * reward_signal.strength
            )
            agent_buffer_trajectory[RewardSignalUtil.rewards_key(name)].extend(
                evaluate_result
            )
            self.collected_rewards[name][agent_id] += np.sum(evaluate_result)

        tmp_advantages = []
        tmp_returns = []
        for name in self.optimizer.reward_signals:
            bootstrap_value = value_next[name]

            local_rewards = agent_buffer_trajectory[
                RewardSignalUtil.rewards_key(name)
            ].get_batch()
            local_value_estimates = agent_buffer_trajectory[
                RewardSignalUtil.value_estimates_key(name)
            ].get_batch()

            local_advantage = get_gae(
                rewards=local_rewards,
                value_estimates=local_value_estimates,
                value_next=bootstrap_value,
                gamma=self.optimizer.reward_signals[name].gamma,
                lambd=self.hyperparameters.lambd,
            )
            local_return = local_advantage + local_value_estimates
            
            agent_buffer_trajectory[RewardSignalUtil.returns_key(name)].set(local_return)
            agent_buffer_trajectory[RewardSignalUtil.advantage_key(name)].set(local_advantage)
            tmp_advantages.append(local_advantage)
            tmp_returns.append(local_return)

        global_advantages = list(
            np.mean(np.array(tmp_advantages, dtype=np.float32), axis=0)
        )
        global_returns = list(np.mean(np.array(tmp_returns, dtype=np.float32), axis=0))
        agent_buffer_trajectory[BufferKey.ADVANTAGES].set(global_advantages)
        agent_buffer_trajectory[BufferKey.DISCOUNTED_RETURNS].set(global_returns)

        self._append_to_update_buffer(agent_buffer_trajectory)

        if trajectory.done_reached:
            self._update_end_episode_stats(agent_id, self.optimizer)

    def _is_ready_update(self) -> bool:
        """
        Nadpisujemy tę funkcję z OnPolicyTrainer i zawsze zwracamy False.
        Zabraniamy tym samym klasie nadrzędnej wywoływania domyślnego
        `self.optimizer.update()`, ponieważ PPG ma swoją własną logikę faz.
        """
        return False

    # ==============================================================================
    # --- LOGIKA PPG: NADPISANA METODA ADVANCE() ---
    # ==============================================================================
    def advance(self) -> None:
        """
        Nadpisuje domyślne zachowanie OnPolicyTrainer.
        Steruje dwiema fazami:
        1. Faza Polityki (gromadzi też dane do aux_buffer).
        2. Faza Pomocnicza (odpalana co N aktualizacji polityki).
        """
        # Odpytanie bazowego Trainera (uruchamia _process_trajectory i odbiera dane z Unity)
        super().advance()

        if not self.is_training:
            return

        batch_size = self.trainer_settings.hyperparameters.batch_size
        seq_len = self.policy.sequence_length

        # Sprawdź, czy update_buffer ma wystarczająco danych na Fazę 1
        if self.update_buffer.num_experiences >= self.trainer_settings.hyperparameters.buffer_size:
            
            buffer_len = self.update_buffer.num_experiences

            # FAZA 1: Trenowanie Polityki (Aktora)
            for _ in range(self.hyperparameters.num_epoch):
                # Tasowanie danych (ważne dla stabilności)
                self.update_buffer.shuffle(sequence_length=seq_len)
                
                # Ręczne cięcie na mini-batche (zgodnie z API ML-Agents)
                for l in range(buffer_len // batch_size):
                    start = l * batch_size
                    end = (l + 1) * batch_size
                    minibatch = self.update_buffer.make_mini_batch(start, end)
                    
                    update_stats = self.optimizer.update_policy(minibatch, seq_len)
                    
                    for stat_name, value in update_stats.items():
                        self._stats_reporter.add_stat(stat_name, value)
            
            # Kopiowanie danych z Fazy 1 do Bufora Pomocniczego
            for key, field in self.update_buffer.items():
                self.aux_buffer[key].extend(field)
            self.update_buffer = AgentBuffer() # Czyścimy standardowy bufor
            self.policy_update_count += 1

            # FAZA 2: Sprawdź, czy nadszedł czas na Fazę Pomocniczą
            if self.policy_update_count >= self.hyperparameters.num_policy_updates_per_aux:
                logger.info(f"Rozpoczynam Fazę Aux na {self.aux_buffer.num_experiences} próbkach...")
                
                aux_buffer_len = self.aux_buffer.num_experiences
                
                for _ in range(self.hyperparameters.aux_epochs):
                    self.aux_buffer.shuffle(sequence_length=seq_len)
                    
                    for l in range(aux_buffer_len // batch_size):
                        start = l * batch_size
                        end = (l + 1) * batch_size
                        aux_minibatch = self.aux_buffer.make_mini_batch(start, end)
                        
                        aux_stats = self.optimizer.update_auxiliary(aux_minibatch, seq_len)
                        
                        for stat_name, value in aux_stats.items():
                            self._stats_reporter.add_stat(stat_name, value)
                
                # Zakończenie Fazy 2: Wyczyszczenie bufora i licznika
                self.aux_buffer = AgentBuffer()
                self.policy_update_count = 0

    def create_optimizer(self) -> TorchOptimizer:
        return TorchPPGOptimizer(  
            cast(TorchPolicy, self.policy), self.trainer_settings  
        )

    def create_policy(
        self, parsed_behavior_id: BehaviorIdentifiers, behavior_spec: BehaviorSpec
    ) -> TorchPolicy:
        actor_cls: Union[Type[SimpleActor], Type[SharedActorCritic]] = SimpleActor
        actor_kwargs: Dict[str, Any] = {
            "conditional_sigma": False,
            "tanh_squash": False,
        }
        if self.shared_critic:
            reward_signal_configs = self.trainer_settings.reward_signals
            reward_signal_names = [
                key.value for key, _ in reward_signal_configs.items()
            ]
            actor_cls = SharedActorCritic
            actor_kwargs.update({"stream_names": reward_signal_names})

        policy = TorchPolicy(
            self.seed,
            behavior_spec,
            self.trainer_settings.network_settings,
            actor_cls,
            actor_kwargs,
        )
        return policy

    def get_policy(self, name_behavior_id: str) -> Policy:
        return self.policy

    @staticmethod
    def get_trainer_name() -> str:
        return TRAINER_NAME