"""Callback for rendering gifs during evaluation."""
import logging
import os
from typing import TYPE_CHECKING, Dict, Optional, Union, Tuple

from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.env.base_env import BaseEnv
from ray.rllib.evaluation import RolloutWorker
from ray.rllib.evaluation.episode import Episode
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.rllib.policy import Policy
from ray.rllib.utils.typing import PolicyID, AgentID
from ray.rllib.policy.sample_batch import SampleBatch
from simfire.enums import BurnStatus

from torch import nn
import numpy as np

if TYPE_CHECKING:
    from ray.rllib.algorithms.algorithm import Algorithm

    from simharness2.environments.reactive import ReactiveHarness

logger = logging.getLogger("ray.rllib")


class RenderSaliencyEnv(DefaultCallbacks):
    """To use this callback, set {"callbacks": RenderEnv} in the algo config."""

    def on_algorithm_init(
        self,
        *,
        algorithm: "Algorithm",
        **kwargs,
    ) -> None:
        """Callback run when a new algorithm instance has finished setup.

        This method gets called at the end of Algorithm.setup() after all
        the initialization is done, and before actually training starts.

        Args:
            algorithm: Reference to the Algorithm instance.
            kwargs: Forward compatibility placeholder.
        """
        logdir = algorithm.logdir
        # TODO: Handle edge case where num_evaluation_workers == 0.
        algorithm.workers.foreach_worker(
            lambda w: w.foreach_env(lambda env: env.set_trial_results_path(logdir)),
            local_worker=False,
        )
        # Make the trial result path accessible to each env (for gif saving).
        algorithm.evaluation_workers.foreach_worker(
            lambda w: w.foreach_env(lambda env: env.set_trial_results_path(logdir)),
            local_worker=False,
        )

    def on_episode_created(
        self,
        *,
        worker: "RolloutWorker",
        base_env: BaseEnv,
        # policies: Dict[PolicyID, Policy],
        # episode: Union[Episode, EpisodeV2],
        env_index: Optional[int] = None,
        **kwargs,
    ) -> None:
        """Callback run right after an Episode has started.

        This method gets called after the Episode(V2)'s respective sub-environment's
        (usually a gym.Env) `reset()` is called by RLlib.

        1) Episode(V2) created: Triggers callback `on_episode_created`.
        2) Respective sub-environment (gym.Env) is `reset()`.
        3) Episode(V2) starts: This callback fires.
        4) Stepping through sub-environment/episode commences.

        Args:
            worker: Reference to the current rollout worker.
            base_env: BaseEnv running the episode. The underlying
                sub environment objects can be retrieved by calling
                `base_env.get_sub_environments()`.
            policies: Mapping of policy id to policy objects. In single
                agent mode there will only be a single "default" policy.
            episode: Episode object which contains the episode's
                state. You can use the `episode.user_data` dict to store
                temporary data, and `episode.custom_metrics` to store custom
                metrics for the episode.
            env_index: The index of the sub-environment that started the episode
                (within the vector of sub-environments of the BaseEnv).
            kwargs: Forward compatibility placeholder.
        """
        env: ReactiveHarness = base_env.vector_env.envs[env_index]

        if worker.config.in_evaluation:
            logger.info("Creating evaluation episode...")
            # Ensure the evaluation env is rendering mode, if it should be.
            if env._should_render and not env.sim.rendering:
                logger.info("Enabling rendering for evaluation env.")
                # TODO: Refactor below 3 lines into `env.render()` method?
                os.environ["SDL_VIDEODRIVER"] = "dummy"
                base_env.vector_env.envs[env_index].sim.reset()
                base_env.vector_env.envs[env_index].sim.rendering = True
            elif not env._should_render and env.sim.rendering:
                logger.error(
                    "Simulation is in rendering mode, but `env._should_render` is False."
                )

    def on_episode_step(
        self,
        *,
        worker: "RolloutWorker",
        base_env: BaseEnv,
        policies: Optional[Dict[PolicyID, Policy]] = None,
        episode: Union[Episode, EpisodeV2],
        env_index: Optional[int] = None,
        **kwargs,
    ) -> None:
        """Runs on each episode step.

        Args:
            worker: Reference to the current rollout worker.
            base_env: BaseEnv running the episode. The underlying
                sub environment objects can be retrieved by calling
                `base_env.get_sub_environments()`.
            policies: Mapping of policy id to policy objects.
                In single agent mode there will only be a single
                "default_policy".
            episode: Episode object which contains episode
                state. You can use the `episode.user_data` dict to store
                temporary data, and `episode.custom_metrics` to store custom
                metrics for the episode.
            env_index: The index of the sub-environment that stepped the episode
                (within the vector of sub-environments of the BaseEnv).
            kwargs: Forward compatibility placeholder.
        """
        env: ReactiveHarness = base_env.vector_env.envs[env_index]

        # agent location, agent's selected action from respective location.
        default_policy: Policy = policies["default_policy"]
        prev_agent_pos = env.agent_pos
        movement_map = np.zeros(env.state.shape[:2])
        interaction_map = np.zeros(env.state.shape[:2])
        for y in range(env.state.shape[0]):
            for x in range(env.state.shape[1]):
                prev_y, prev_x = prev_agent_pos
                env.state[prev_y, prev_x, 0] = env.sim.fire_map[prev_y, prev_x]
                new_agent_pos = (y, x)
                prev_agent_pos = new_agent_pos
                # Skip if BURNING or BURNED
                if env.state[y, x, 0] in [BurnStatus.BURNING, BurnStatus.BURNED]:
                    continue
                env.state[y, x, 0] = env.sim_agent_id

                action = default_policy.compute_single_action(env.state, explore=False)[0]
                movement, interaction = env._parse_action(action)
                movement_str = env.movements[movement]
                interaction_str = env.interactions[interaction]
                movement_map[y, x] = movement
                interaction_map[y, x] = interaction

        import matplotlib.pyplot as plt
        plt.imshow(movement_map, cmap="hot", interpolation="nearest")
        plt.savefig("movement_map.png")

        plt.figure()
        plt.imshow(interaction_map, cmap="hot", interpolation="nearest")
        plt.savefig("interaction_map.png")
        print("done")

        # get observation space, place agent at location X,Y, and query model for action
        # TODO: add helper method to harness to build obs space from current timestep
        # TODO: and previous timestep, so we can get action output for diff agent start pos

    def on_episode_end(
        self,
        *,
        worker: "RolloutWorker",
        base_env: BaseEnv,
        policies: Dict[PolicyID, Policy],
        episode: Union[Episode, EpisodeV2, Exception],
        env_index: Optional[int] = None,
        **kwargs,
    ) -> None:
        """Runs when an episode is done.

        Args:
            worker: Reference to the current rollout worker.
            base_env: BaseEnv running the episode. The underlying
                sub environment objects can be retrieved by calling
                `base_env.get_sub_environments()`.
            policies: Mapping of policy id to policy
                objects. In single agent mode there will only be a single
                "default_policy".
            episode: Episode object which contains episode
                state. You can use the `episode.user_data` dict to store
                temporary data, and `episode.custom_metrics` to store custom
                metrics for the episode.
                In case of environment failures, episode may also be an Exception
                that gets thrown from the environment before the episode finishes.
                Users of this callback may then handle these error cases properly
                with their custom logics.
            env_index: The index of the sub-environment that ended the episode
                (within the vector of sub-environments of the BaseEnv).
            kwargs: Forward compatibility placeholder.
        """
        env: ReactiveHarness = base_env.vector_env.envs[env_index]
        # Save a GIF from the last episode
        # TODO: Do we also want to save the fire spread graph?
        if worker.config.in_evaluation:
            logdir = env._trial_results_path
            eval_iters = env._num_eval_iters
            # Check if there is a gif "ready" to be saved
            if env._should_render and env.sim.rendering:
                # FIXME Update logic to handle saving same gif when writing to Aim UI
                gif_save_path = os.path.join(
                    logdir, "gifs", f"eval_iter_{eval_iters}.gif"
                )
                # FIXME: Can we save each gif in a folder that relates it to episode iter?
                logger.info(f"Saving GIF to {gif_save_path}...")
                base_env.vector_env.envs[env_index].sim.save_gif(gif_save_path)
                # Save the gif_path so that we can write image to aim server, if desired
                # NOTE: `save_path` is a list after the above; do element access for now
                logger.debug(f"Type of gif_save_path: {type(gif_save_path)}")
                episode.media.update({"gif": gif_save_path})

                # Try to collect and log episode history, if it was saved.
                if env.harness_analytics.sim_analytics.save_history:
                    env.harness_analytics.save_sim_history(logdir, eval_iters)

            # sim.save_spread_graph(save_dir)

    def on_evaluate_start(
        self,
        *,
        algorithm: "Algorithm",
        **kwargs,
    ) -> None:
        """Callback before evaluation starts.

        This method gets called at the beginning of Algorithm.evaluate().

        Args:
            algorithm: Reference to the algorithm instance.
            kwargs: Forward compatibility placeholder.
        """
        # TODO: Add note in docs that the local worker IS NOT rendered. With this
        # assumption, we should always set `evaluation.evaluation_num_workers >= 1`.
        # TODO: Handle edge case where num_evaluation_workers == 0.
        logger.info("Starting evaluation...")
        # Increment the number of evaluation iterations
        algorithm.evaluation_workers.foreach_worker(
            lambda w: w.foreach_env(lambda env: env._increment_evaluation_iterations()),
            local_worker=False,
        )
        # TODO: Use a function to decide if this round should be rendered (ie log10).
        # TODO: Additionally, log the total number of episodes run so far.
        # Enable the evaluation environment (s) to be rendered.
        algorithm.evaluation_workers.foreach_worker(
            lambda w: w.foreach_env(lambda env: env._configure_env_rendering(True)),
            local_worker=False,
        )

    def on_postprocess_trajectory(
        self,
        *,
        worker: "RolloutWorker",
        episode: Episode,
        agent_id: AgentID,
        policy_id: PolicyID,
        policies: Dict[PolicyID, Policy],
        postprocessed_batch: SampleBatch,
        original_batches: Dict[AgentID, Tuple[Policy, SampleBatch]],
        **kwargs,
    ) -> None:
        """Called immediately after a policy's postprocess_fn is called.

        You can use this callback to do additional postprocessing for a policy,
        including looking at the trajectory data of other agents in multi-agent
        settings.

        Args:
            worker: Reference to the current rollout worker.
            episode: Episode object.
            agent_id: Id of the current agent.
            policy_id: Id of the current policy for the agent.
            policies: Mapping of policy id to policy objects. In single
                agent mode there will only be a single "default_policy".
            postprocessed_batch: The postprocessed sample batch
                for this agent. You can mutate this object to apply your own
                trajectory postprocessing.
            original_batches: Mapping of agents to their unpostprocessed
                trajectory data. You should not mutate this object.
            kwargs: Forward compatibility placeholder.
        """
        breakpoint()

    def on_evaluate_end(
        self,
        *,
        algorithm: "Algorithm",
        evaluation_metrics: dict,
        **kwargs,
    ) -> None:
        """Runs when the evaluation is done.

        Runs at the end of Algorithm.evaluate().

        Args:
            algorithm: Reference to the algorithm instance.
            evaluation_metrics: Results dict to be returned from algorithm.evaluate().
                You can mutate this object to add additional metrics.
            kwargs: Forward compatibility placeholder.
        """
        # TODO: Add note in docs that the local worker IS NOT rendered. With this
        # assumption, we should always set `evaluation.evaluation_num_workers >= 1`.
        # TODO: Handle edge case where num_evaluation_workers == 0.
        breakpoint()

        # TODO: Use a function to decide if this round should be rendered (ie log10).
        # Disable the evaluation environment (s) to be rendered.
        algorithm.evaluation_workers.foreach_worker(
            lambda w: w.foreach_env(lambda env: env._configure_env_rendering(False)),
            local_worker=False,
        )
