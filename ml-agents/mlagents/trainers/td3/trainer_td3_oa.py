from typing import cast, Union, Type, Dict, Any
from mlagents.trainers.buffer import BufferKey, RewardSignalUtil
from mlagents.trainers.trainer.off_policy_trainer import OffPolicyTrainer, logger
from mlagents.trainers.settings import TrainerSettings
from mlagents.trainers.policy.torch_policy import TorchPolicy
from mlagents.trainers.trajectory import Trajectory, ObsUtil
from mlagents_envs.base_env import BehaviorSpec, ActionTuple
from mlagents.trainers.behavior_id_utils import BehaviorIdentifiers
from mlagents_envs.logging_util import get_logger

# Importujemy naszego Optymalizatora TD3
from mlagents.trainers.td3.optimizer_torch_td3_oa import TorchTD3Optimizer
# Importujemy klasę bazową dla struktury sieci neuronowej, tak samo jak PPO i SAC
from mlagents.trainers.torch_entities.networks import SimpleActor

import numpy as np
from mlagents.trainers.policy.torch_policy import TorchPolicy
from mlagents.trainers.action_info import ActionInfo

logger = get_logger(__name__)

class OUNoise:
    """Proces Ornsteina-Uhlenbecka generujący płynny, skorelowany szum dla robotyki."""
    def __init__(self, action_dimension, mu=0.0, theta=0.15, sigma=0.2):
        self.action_dimension = action_dimension
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = None
        # self.state = np.ones(self.action_dimension) * self.mu
        # self.reset()

    # def reset(self):
    #     self.state = np.ones(self.action_dimension) * self.mu

    def sample(self, batch_size):
        if self.state is None or self.state.shape[0] != batch_size:
            self.state = np.ones((batch_size, self.action_dimension)) * self.mu

        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(batch_size, self.action_dimension)
        self.state = x + dx
        return self.state

    
# probnie podejscie z punktami kluczowymi co x klatek
class EpisodicNoiseTD3Policy(TorchPolicy):
    """Polityka TD3 z szumem, który wygasa w trakcie pojedynczego epizodu i resetuje się po upadku."""
    def __init__(self, *args, start_noise: float = 0.5, end_noise: float = 0.05, decay_steps: int = 1500, **kwargs):
        super().__init__(*args, **kwargs)
        self.start_noise = start_noise
        self.end_noise = end_noise
        self.decay_steps = decay_steps  # Jak szybko szum zanika w trakcie życia agenta
        
        # Słownik zapamiętujący, ile kroków przeżył dany agent
        self.agent_steps: Dict[str, int] = {}

    def reset_noise_for_agent(self, agent_id: str):
        """Metoda wywoływana z zewnątrz po śmierci/resecie agenta."""
        self.agent_steps[agent_id] = 0

    def evaluate(self, decision_requests, global_agent_ids: list) -> Dict[str, Any]:
        run_out = super().evaluate(decision_requests, global_agent_ids)
        action_tuple = run_out.get("action")
        
        if action_tuple is not None and action_tuple.continuous is not None:
            raw_actions = action_tuple.continuous 
            deterministic_actions = np.tanh(raw_actions)
            noisy_actions = np.zeros_like(deterministic_actions)
            
            # Szum wyliczamy indywidualnie dla każdego agenta (bo mogą resetować się w różnym czasie)
            for i, agent_id in enumerate(global_agent_ids):
                if agent_id not in self.agent_steps:
                    self.agent_steps[agent_id] = 0
                    
                current_step = self.agent_steps[agent_id]
                
                # Obliczanie ułamka zaniku szumu
                fraction = min(1.0, current_step / self.decay_steps)
                current_noise_scale = self.start_noise - fraction * (self.start_noise - self.end_noise)
                
                # Dodajemy szum Gaussa do deterministycznej akcji
                noise = np.random.normal(loc=0.0, scale=current_noise_scale, size=deterministic_actions[i].shape)
                noisy_actions[i] = deterministic_actions[i] + noise
                
                # Inkrementujemy wiek agenta
                self.agent_steps[agent_id] += 1
            
            final_actions = np.clip(noisy_actions, -1.0, 1.0)
            
            run_out["action"] = ActionTuple(
                continuous=np.float32(final_actions), 
                discrete=action_tuple.discrete
            )
            
        return run_out

TRAINER_NAME = "td3"

class TD3Trainer(OffPolicyTrainer):
    """Trener dla algorytmu TD3 (Twin Delayed DDPG)."""

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
        self.seed = seed
        self.step = 0
        self.policy: TorchPolicy = None  # type: ignore
        self.checkpoint_replay_buffer = trainer_settings.hyperparameters.save_replay_buffer

    def _process_trajectory(self, trajectory: Trajectory) -> None:
        """
        Nadpisana z klasy abstrakcyjnej. 
        Gdy używamy algorytmów Off-Policy z buforem Replay (jak TD3/SAC), 
        główna obróbka trajektorii dzieje się w logice nadrzędnej.
        """
        super()._process_trajectory(trajectory)

        last_step = trajectory.steps[-1]
        agent_id = trajectory.agent_id
        agent_buffer_trajectory = trajectory.to_agentbuffer()

        # --- Zabezpieczenie przed błędną konfiguracją Unity
        self._warn_if_group_reward(agent_buffer_trajectory)

        # --- Dynamiczna normalizacja zmysłów (Running Statistics)
        if self.is_training:
            # Aktor automatycznie aktualizuje średnią/wariancję tego, co widzi
            self.policy.actor.update_normalization(agent_buffer_trajectory)
            # Jeśli podepniemy normalizator do Optymalizatora (Krytyków), tu go wołamy
            if hasattr(self.optimizer, "update_normalization"):
                self.optimizer.update_normalization(agent_buffer_trajectory)

        # 1. Zliczamy bazowe nagrody ze środowiska (aby Logger wiedział, ile zdobyliśmy punktów)
        self.collected_rewards["environment"][agent_id] += sum(
            agent_buffer_trajectory[BufferKey.ENVIRONMENT_REWARDS]
        )

        # 2. Ewaluujemy i zliczamy ewentualne nagrody dodatkowe (np. z plików YAML, jak nagroda 'extrinsic')
        for name, reward_signal in self.optimizer.reward_signals.items():
            evaluate_result = (
                reward_signal.evaluate(agent_buffer_trajectory) * reward_signal.strength
            )
            # Dodajemy zewaluowane nagrody z powrotem do bufora
            agent_buffer_trajectory[RewardSignalUtil.rewards_key(name)].extend(
                evaluate_result
            )
            # Dodajemy je do puli do raportu statystyk
            self.collected_rewards[name][agent_id] += sum(evaluate_result)

        # --- TensorBoard Analytics (Z raportu na żywo z trajektorii)
        # Pobieramy, co Krytyk sądził o tej ścieżce i wrzucamy na wykres
        value_estimates, _, _ = self.optimizer.get_trajectory_value_estimates(
            agent_buffer_trajectory, trajectory.next_obs, trajectory.done_reached
        )
        
        for name, v in value_estimates.items():
            # Zobaczymy to w TensorBoardzie w sekcji Policy!
            if name in self.optimizer.reward_signals:
                stat_name = self.optimizer.reward_signals[name].name.capitalize()
            else:
                stat_name = name.capitalize()
            self._stats_reporter.add_stat(
                f"Policy/{stat_name} Value",
                np.mean(v),
            )

        # 3. KLUCZOWY MOMENT: Wysyłanie raportu
        # Sprawdzamy, czy Unity nadało flagę końca epizodu (np. ze względu na uderzenie w limit Max Step)
        # Bootstrap using the last step rather than the bootstrap step if max step is reached.
        # Set last element to duplicate obs and remove dones.
        if last_step.interrupted:
            last_step_obs = last_step.obs
            for i, obs in enumerate(last_step_obs):
                agent_buffer_trajectory[ObsUtil.get_name_at_next(i)][-1] = obs
            agent_buffer_trajectory[BufferKey.DONE][-1] = False

        self._append_to_update_buffer(agent_buffer_trajectory)
        if trajectory.done_reached:
            # Poniższa funkcja zbierze sumę z self.collected_rewards,
            # wyśle statystyki na ekran (Mean Reward), zapisze do TensorBoarda
            # i zresetuje licznik self.collected_rewards dla tego agenta.
            self._update_end_episode_stats(agent_id, self.optimizer)
            if hasattr(self.policy, "reset_noise_for_agent"):
                self.policy.reset_noise_for_agent(trajectory.agent_id)

    def create_policy(
        self, parsed_behavior_id: BehaviorIdentifiers, behavior_spec: BehaviorSpec
    ) -> TorchPolicy:
        """
        Tworzy politykę (mózg) opartą na PyTorchu z hiperparametrami dla TD3.
        """
        # TD3 zazwyczaj korzysta z prostej, rozłącznej architektury dla Aktora (brak SharedCritic)
        actor_cls: Type[SimpleActor] = SimpleActor
        
        # W TD3 nie potrzebujemy kompresji ani zaleceń odchyleń, jakie są w SAC/PPO
        actor_kwargs: Dict[str, Any] = {
            "conditional_sigma": False,
            "tanh_squash": False,  # TD3 dodaje szum PRZED Tanh, więc nie chcemy, by sieć sama się ściśnięła
        }

        # Tworzymy instancję mózgu
        policy = EpisodicNoiseTD3Policy(
            self.seed,
            behavior_spec,
            self.trainer_settings.network_settings,
            SimpleActor,
            actor_kwargs,
            start_noise=0.6,   # Zaczynamy agresywnie
            end_noise=0.05,    # Kończymy na bardzo małym szumie dla precyzji
            decay_steps=1500   # Szum zanika przez 1500 klatek danego epizodu
        )
        self.maybe_load_replay_buffer()
        return policy

    def create_optimizer(self) -> TorchTD3Optimizer:
        """Tworzy matematyczny silnik (Optymalizator) dla TD3."""
        return TorchTD3Optimizer(cast(TorchPolicy, self.policy), self.trainer_settings)

    def get_policy(self, name_behavior_id: str) -> TorchPolicy:
        """Pobiera aktualną politykę z trainera."""
        return self.policy

    @staticmethod
    def get_trainer_name() -> str:
        return TRAINER_NAME