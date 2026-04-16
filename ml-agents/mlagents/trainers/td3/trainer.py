from typing import cast, Type, Dict, Any
import numpy as np

from mlagents.trainers.buffer import BufferKey, RewardSignalUtil
from mlagents.trainers.trainer.off_policy_trainer import OffPolicyTrainer
from mlagents.trainers.settings import TrainerSettings
from mlagents.trainers.policy.torch_policy import TorchPolicy
from mlagents.trainers.trajectory import Trajectory, ObsUtil
from mlagents_envs.base_env import BehaviorSpec, ActionTuple
from mlagents.trainers.behavior_id_utils import BehaviorIdentifiers
from mlagents_envs.logging_util import get_logger

# Importujemy naszego Optymalizatora TD3 oraz bazowego Aktora
from mlagents.trainers.td3.optimizer_torch import TorchTD3Optimizer
from mlagents.trainers.torch_entities.networks import SimpleActor

logger = get_logger(__name__)
TRAINER_NAME = "td3"

class OUNoise:
    """Wektoryzowany proces Ornsteina-Uhlenbecka (obsługuje wielu agentów naraz)."""
    def __init__(self, action_dimension, mu=0.0, theta=0.15, sigma=0.2):
        self.action_dimension = action_dimension
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = None 

    def sample(self, current_batch_size):
        # Inicjalizujemy stan jako macierz [Liczba agentów x Liczba silników]
        if self.state is None or self.state.shape[0] != current_batch_size:
            self.state = np.ones((current_batch_size, self.action_dimension)) * self.mu
        
        x = self.state
        # Generujemy losowość dla każdego agenta i silnika niezależnie
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(current_batch_size, self.action_dimension)
        self.state = x + dx
        return self.state


class TD3NoisePolicy(TorchPolicy):
    def __init__(self, *args, 
                 start_noise: float = 0.5,   
                 end_noise: float = 0.05,    
                 decay_steps: int = 500000,  
                 warmup_steps: int = 50000,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.start_noise = start_noise
        self.end_noise = end_noise
        self.decay_steps = decay_steps
        self.warmup_steps = warmup_steps
        self.ou_noise = None
        
        # Zliczamy łączną liczbę wykonanych kroków (wszystkich agentów na scenie)
        self.total_agent_steps = 0 

    def evaluate(self, decision_requests, global_agent_ids: list) -> Dict[str, Any]:
        run_out = super().evaluate(decision_requests, global_agent_ids)
        action_tuple = run_out.get("action")
        
        if action_tuple is not None and action_tuple.continuous is not None:
            # Wyciągamy na bieżąco ile agentów prosi o decyzję i ile mają stawów
            current_batch_size = action_tuple.continuous.shape[0]  
            action_size = action_tuple.continuous.shape[1] 
            
            self.total_agent_steps += current_batch_size
            raw_actions = action_tuple.continuous 

            if self.ou_noise is None:
                self.ou_noise = OUNoise(action_size, sigma=1.0)

            # --------------------------------------------------
            # 1. FAZA ROZGRZEWKI (Prawdziwa padaczka z białego szumu)
            # --------------------------------------------------
            if self.total_agent_steps < self.warmup_steps:
                # Agresywne, rwące akcje od -1 do 1. Omijamy Tanh.
                random_actions = np.random.uniform(-1.0, 1.0, size=(current_batch_size, action_size))
                
                run_out["action"] = ActionTuple(
                    continuous=np.float32(random_actions), 
                    discrete=action_tuple.discrete
                )
                return run_out 

            # --------------------------------------------------
            # 2. FAZA NAUKI (Sieć neuronowa + wektoryzowany szum OU)
            # --------------------------------------------------
            fraction = min(1.0, max(0.0, (self.total_agent_steps - self.warmup_steps) / self.decay_steps))
            current_noise_scale = self.start_noise - fraction * (self.start_noise - self.end_noise)

            # Pobieramy płynny szum dla całej paczki agentów
            noise = self.ou_noise.sample(current_batch_size) * current_noise_scale
            
            # Dodajemy szum PRZED funkcją Tanh (ochrona przed znikającym gradientem)
            noisy_raw = raw_actions + noise
            final_actions = np.tanh(noisy_raw)
            
            run_out["action"] = ActionTuple(
                continuous=np.float32(final_actions), 
                discrete=action_tuple.discrete
            )
            
        return run_out


class TD3Trainer(OffPolicyTrainer):
    """Trener dla algorytmu TD3 (Twin Delayed DDPG)."""

    def __init__(self, behavior_name: str, reward_buff_cap: int, trainer_settings: TrainerSettings, training: bool, load: bool, seed: int, artifact_path: str):
        super().__init__(behavior_name, reward_buff_cap, trainer_settings, training, load, seed, artifact_path)
        self.seed = seed
        self.step = 0
        self.policy: TorchPolicy = None  # type: ignore

    def _process_trajectory(self, trajectory: Trajectory) -> None:
        super()._process_trajectory(trajectory)

        last_step = trajectory.steps[-1]
        agent_id = trajectory.agent_id
        agent_buffer_trajectory = trajectory.to_agentbuffer()

        self.collected_rewards["environment"][agent_id] += sum(agent_buffer_trajectory[BufferKey.ENVIRONMENT_REWARDS])

        for name, reward_signal in self.optimizer.reward_signals.items():
            evaluate_result = reward_signal.evaluate(agent_buffer_trajectory) * reward_signal.strength
            agent_buffer_trajectory[RewardSignalUtil.rewards_key(name)].extend(evaluate_result)
            self.collected_rewards[name][agent_id] += sum(evaluate_result)

        if last_step.interrupted:
            last_step_obs = last_step.obs
            for i, obs in enumerate(last_step_obs):
                agent_buffer_trajectory[ObsUtil.get_name_at_next(i)][-1] = obs
            agent_buffer_trajectory[BufferKey.DONE][-1] = False

        self._append_to_update_buffer(agent_buffer_trajectory)
        if trajectory.done_reached:
            self._update_end_episode_stats(agent_id, self.optimizer)

    def create_policy(self, parsed_behavior_id: BehaviorIdentifiers, behavior_spec: BehaviorSpec) -> TorchPolicy:
        actor_cls: Type[SimpleActor] = SimpleActor
        
        actor_kwargs: Dict[str, Any] = {
            "conditional_sigma": False,
            "tanh_squash": False,  # Wyłączone, bo uciskamy sami po dodaniu szumu OU
        }

        # Wstrzykujemy naszą nakładkę z szumem
        policy = TD3NoisePolicy(
            self.seed,
            behavior_spec,
            self.trainer_settings.network_settings,
            actor_cls,
            actor_kwargs,
            start_noise=0.5,
            end_noise=0.05,
            decay_steps=1000000,
            warmup_steps=50000  # <--- Ważne żeby zgadzało się z YAML (buffer_init_steps)
        )
        return policy

    def create_optimizer(self) -> TorchTD3Optimizer:
        return TorchTD3Optimizer(cast(TorchPolicy, self.policy), self.trainer_settings)

    def get_policy(self, name_behavior_id: str) -> TorchPolicy:
        return self.policy

    @staticmethod
    def get_trainer_name() -> str:
        return TRAINER_NAME