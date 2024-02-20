"""Callback for rendering gifs during evaluation."""

import logging
import os
from math import log
from typing import TYPE_CHECKING, Dict, Optional, Union

from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.env.base_env import BaseEnv
from ray.rllib.evaluation import RolloutWorker
from ray.rllib.evaluation.episode import Episode
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.rllib.policy import Policy
from ray.rllib.utils.typing import PolicyID, ResultDict  # AgentID, EnvType,

import simharness2.utils.utils as utils


if TYPE_CHECKING:
    from ray.rllib.algorithms.algorithm import Algorithm
    from simfire.sim.simulation import FireSimulation

    from simharness2.environments.fire_harness import FireHarness

logger = logging.getLogger(__name__)

TRAIN_KEY = "train"
EVAL_KEY = "evaluation"
# TODO: Add a config option to control rendering settings.
# Switch to enable rendering of training environments.
RENDER_TRAIN_ENVS = True
# NOTE: Probably better to use a dictionary so that "eval" and "train" are not forced to
# use the same interval setup, but good enough for the time being. When this update is
# added, the logic in RenderEnv.should_render_env will need to be updated accordingly.
# Options: "log" or "linear"
RENDER_INTERVAL_TYPE = "linear"
# Set the base for the logarithmic interval
LOGARITHMIC_BASE = 10
# Set the step size for the linear interval
LINEAR_INTERVAL_STEP = 1  # 0


class RenderEnv(DefaultCallbacks):
    """To use this callback, set {"callbacks": RenderEnv} in the algo config."""

    def __init__(self, legacy_callbacks_dict: Dict[str, callable] = None):
        super().__init__(legacy_callbacks_dict=legacy_callbacks_dict)
        # Utilities used throughout methods in the callback.
        self.render_current_episode = False
        self.curr_iter = -1

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
        utils.validate_evaluation_config(algorithm.config)
        logdir = algorithm.logdir
        workers = [algorithm.workers, algorithm.evaluation_workers]
        # TODO: Handle edge case where num_evaluation_workers == 0.
        # Make the trial result path accessible to each env (for gif saving).
        for worker in workers:
            worker.foreach_worker(
                lambda w: w.foreach_env(lambda env: setattr(env, "trial_logdir", logdir)),
                local_worker=True,
            )
        # self.render_envs = algorithm.config.env_config.get("render_envs", "all")

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
        env: FireHarness[FireSimulation] = base_env.get_sub_environments()[env_index]
        env_type = EVAL_KEY if worker.config.in_evaluation else TRAIN_KEY
        self.render_current_episode = self.should_render_env(env, env_type)
        if self.render_current_episode:
            env_ctx = worker.env_context
            w_idx, v_idx = env_ctx.worker_index, env_ctx.vector_index
            logger.info(
                f"Preparing to render {env_type} environment (w: {w_idx}, v: {v_idx})..."
            )
            env._configure_env_rendering(True)

    def should_render_env(
        self, env: "FireHarness[FireSimulation]", env_type: str
    ) -> bool:
        """Check if the environment should be rendered."""
        if env_type == TRAIN_KEY:
            self.curr_iter = env.current_result.get("training_iteration", 0)
        else:
            self.curr_iter = env._num_eval_iters

        if env_type == TRAIN_KEY and RENDER_TRAIN_ENVS or env_type == EVAL_KEY:
            # Use specified interval type to determine if the env should be rendered.
            if RENDER_INTERVAL_TYPE == "log":
                value = log(self.curr_iter, LOGARITHMIC_BASE)
                return value.is_integer() and value > 0
            elif RENDER_INTERVAL_TYPE == "linear":
                return self.curr_iter % LINEAR_INTERVAL_STEP == 0

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
        env: FireHarness[FireSimulation] = base_env.get_sub_environments()[env_index]
        env_type = EVAL_KEY if worker.config.in_evaluation else TRAIN_KEY
        # FIXME: Condition is overkill, but ensures callback, env, and env.sim are
        # "on the same page".
        if self.render_current_episode and env._should_render and env.sim.rendering:
            logdir = env.trial_logdir
            env_ctx = worker.env_context
            w_idx, v_idx = env_ctx.worker_index, env_ctx.vector_index

            # Save a GIF from the last episode
            # Check if there is a gif "ready" to be saved
            # FIXME Update logic to handle saving same gif when writing to Aim UI
            context_dict = {}
            # FIXME: Should we round lat, lon to a certain precision??
            lat, lon = env.sim.config.landfire_lat_long_box.points[0]
            op_data_lat_lon = f"operational_lat_{lat}_lon_{lon}"
            fire_init_pos = env.sim.config.fire.fire_initial_position
            context_dict.update({"fire_initial_position": str(fire_init_pos)})
            # FIXME: Finalize path for saving gifs (and add note to docs) - for example,
            # save each gif in a folder that relates it to episode iter?
            env_episode_id = f"iter_{self.curr_iter}_w_{w_idx}_v_{v_idx}"
            gif_save_path = os.path.join(
                logdir,
                env_type,
                "gifs",
                op_data_lat_lon,
                f"fire_init_pos_x_{fire_init_pos[0]}_y_{fire_init_pos[1]}",
                f"{env_episode_id}.gif",
            )
            logger.info(f"Saving GIF to {gif_save_path}...")
            base_env.get_sub_environments()[env_index].sim.save_gif(gif_save_path)
            # Save the gif_path so that we can write image to aim server, if desired
            # NOTE: `save_path` is a list after the above; do element access for now
            logger.debug(f"Type of gif_save_path: {type(gif_save_path)}")
            gif_data = {
                "path": gif_save_path,
                "name": op_data_lat_lon,
                "step": self.curr_iter,
                # "epoch":
                "context": context_dict,
            }
            episode.media.update({"gif_data": gif_data})

            # Try to collect and log episode history, if it was saved.
            if env.harness_analytics.sim_analytics.save_history:
                episode_history_dir = os.path.join(logdir, env_type)
                env.harness_analytics.save_sim_history(
                    episode_history_dir, env_episode_id
                )

            env._configure_env_rendering(False)

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

    def on_train_result(
        self,
        *,
        algorithm: "Algorithm",
        result: ResultDict,
        **kwargs,
    ) -> None:
        """Called at the end of Algorithm.train().

        Args:
            algorithm: Current Algorithm instance.
            result: Dict of results returned from Algorithm.train() call.
                You can mutate this object to add additional metrics.
            kwargs: Forward compatibility placeholder.
        """
        # Update the current result for each environment.
        algorithm.workers.foreach_worker(
            lambda w: w.foreach_env(lambda env: setattr(env, "current_result", result)),
            local_worker=False,
        )
