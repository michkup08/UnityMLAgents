# ## ML-Agent Learning (DroQ)
# Zmodyfikowana implementacja bazująca na SAC z uwzględnieniem Dropout Q-Functions.

from typing import cast
import numpy as np

from mlagents_envs.logging_util import get_logger
from mlagents_envs.base_env import BehaviorSpec
from mlagents.trainers.buffer import BufferKey
from mlagents.trainers.optimizer.torch_optimizer import TorchOptimizer
from mlagents.trainers.trainer.off_policy_trainer import OffPolicyTrainer
from mlagents.trainers.policy.torch_policy import TorchPolicy
from mlagents.trainers.policy.policy import Policy
from mlagents.trainers.droQ.optimizer_torch import TorchDroQOptimizer, DroQSettings
from mlagents.trainers.trajectory import Trajectory, ObsUtil
from mlagents.trainers.behavior_id_utils import BehaviorIdentifiers
from mlagents.trainers.settings import TrainerSettings

from mlagents.trainers.torch_entities.networks import SimpleActor

logger = get_logger(__name__)

BUFFER_TRUNCATE_PERCENT = 0.8

TRAINER_NAME = "droq"


class DroQTrainer(OffPolicyTrainer):
    """
    DroQTrainer to implementacja algorytmu DroQ (Dropout Q-Functions for Doubly Q-Learning),
    stanowiąca ulepszenie SAC, wspierająca dyskretne akcje i sieci rekurencyjne.
    """

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
        self.policy: TorchPolicy = None 
        self.optimizer: TorchDroQOptimizer = None
        self.hyperparameters: DroQSettings = cast(
            DroQSettings, trainer_settings.hyperparameters
        )
        self._step = 0

        self.update_steps = 1
        self.reward_signal_update_steps = 1

        self.steps_per_update = self.hyperparameters.steps_per_update
        self.reward_signal_steps_per_update = (
            self.hyperparameters.reward_signal_steps_per_update
        )

        self.checkpoint_replay_buffer = self.hyperparameters.save_replay_buffer

    def _process_trajectory(self, trajectory: Trajectory) -> None:
        super()._process_trajectory(trajectory)
        last_step = trajectory.steps[-1]
        agent_id = trajectory.agent_id  

        agent_buffer_trajectory = trajectory.to_agentbuffer()
        self._warn_if_group_reward(agent_buffer_trajectory)

        if self.is_training:
            self.policy.actor.update_normalization(agent_buffer_trajectory)
            self.optimizer.critic.update_normalization(agent_buffer_trajectory)

        self.collected_rewards["environment"][agent_id] += np.sum(
            agent_buffer_trajectory[BufferKey.ENVIRONMENT_REWARDS]
        )
        for name, reward_signal in self.optimizer.reward_signals.items():
            evaluate_result = (
                reward_signal.evaluate(agent_buffer_trajectory) * reward_signal.strength
            )
            self.collected_rewards[name][agent_id] += np.sum(evaluate_result)

        (
            value_estimates,
            _,
            value_memories,
        ) = self.optimizer.get_trajectory_value_estimates(
            agent_buffer_trajectory, trajectory.next_obs, trajectory.done_reached
        )
        if value_memories is not None:
            agent_buffer_trajectory[BufferKey.CRITIC_MEMORY].set(value_memories)

        for name, v in value_estimates.items():
            self._stats_reporter.add_stat(
                f"Policy/{self.optimizer.reward_signals[name].name.capitalize()} Value",
                np.mean(v),
            )

        if last_step.interrupted:
            last_step_obs = last_step.obs
            for i, obs in enumerate(last_step_obs):
                agent_buffer_trajectory[ObsUtil.get_name_at_next(i)][-1] = obs
            agent_buffer_trajectory[BufferKey.DONE][-1] = False

        self._append_to_update_buffer(agent_buffer_trajectory)

        if trajectory.done_reached:
            self._update_end_episode_stats(agent_id, self.optimizer)

    def create_optimizer(self) -> TorchOptimizer:
        return TorchDroQOptimizer( 
            cast(TorchPolicy, self.policy), self.trainer_settings
        ) 

    def create_policy(
        self, parsed_behavior_id: BehaviorIdentifiers, behavior_spec: BehaviorSpec
    ) -> TorchPolicy:
        actor_cls = SimpleActor
        actor_kwargs = {"conditional_sigma": True, "tanh_squash": True}

        policy = TorchPolicy(
            self.seed,
            behavior_spec,
            self.trainer_settings.network_settings,
            actor_cls,
            actor_kwargs,
        )
        self.maybe_load_replay_buffer()
        return policy

    def get_policy(self, name_behavior_id: str) -> Policy:
        return self.policy

    @staticmethod
    def get_trainer_name() -> str:
        return TRAINER_NAME