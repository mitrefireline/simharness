"""Callback for rendering gifs during evaluation."""
import logging
import os
from typing import TYPE_CHECKING, Dict, Optional, Union

from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.env.base_env import BaseEnv
from ray.rllib.env.env_context import EnvContext
from ray.rllib.evaluation import RolloutWorker
from ray.rllib.evaluation.episode import Episode
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.rllib.policy import Policy
from ray.rllib.utils.typing import PolicyID  # AgentID, EnvType,
from simfire.sim.simulation import FireSimulation
from simfire.utils.config import Config

if TYPE_CHECKING:
    from ray.rllib.algorithms.algorithm import Algorithm

    from simharness2.environments.reactive import ReactiveHarness

logger = logging.getLogger(__name__)


class RenderEnv(DefaultCallbacks):
    """To use this callback, set {"callbacks": RenderEnv} in the algo config."""

    def __init__(self, legacy_callbacks_dict: Dict[str, callable] = None):
        """TODO.

        Args:
            legacy_callbacks_dict (Dict[str, callable], optional): _description_.
                Defaults to None.
        """
        super().__init__(legacy_callbacks_dict)

        # Empty path, updated within first `on_episode_start` call (for each rollout).
        self.trial_results_path: str = ""

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
        # Put the trial result path in Ray's object store (accessible by all workers).
        algorithm.evaluation_workers.foreach_worker(
            lambda w: w.foreach_env(lambda env: env.set_trial_results_path(logdir)),
            local_worker=False,
        )

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
        env_ctx = worker.env_context
        vector_idx = env_ctx.vector_index
        in_evaluation = worker.policy_config["in_evaluation"]
        env: ReactiveHarness = base_env.vector_env.envs[vector_idx]
        headless = env.sim.config.simulation.headless
        # Save a GIF and fire spread graph from the last episode
        if not headless and in_evaluation:
            if not os.environ.get("SDL_VIDEODRIVER", None):
                logger.warn("SETTING SDL_VIDEODRIVE... AGAIN.")

            os.environ["SDL_VIDEODRIVER"] = "dummy"
            # FIXME use fire id instead of worker+vector idxs
            id = f"worker_idx_{env_ctx.worker_index}_vector_idx_{vector_idx}"
            # FIXME what happens when cli.mode == tune??
            logdir = env._trial_results_path
            episode_num = env.harness_analytics.episodes_total
            save_path = os.path.join(logdir, "gifs", f"episode_{episode_num}_{id}.gif")
            # FIXME: Can we save each gif in a folder that relates it to episode iter?
            logger.warning(f"ATTEMPTING TO SAVE GIF TO {save_path}")
            if hasattr(base_env.vector_env.envs[vector_idx].sim, "_game"):
                base_env.vector_env.envs[vector_idx].sim.save_gif(save_path)
            else:
                logger.warning("SIM object does NOT have _game attr.")
            # Save the gif_path so that we can write image to aim server, if desired
            # NOTE: `save_path` is a list after the above, so do element access for now
            episode.media.update({id: save_path})
            # sim.save_spread_graph(save_dir)
            sim_data = base_env.vector_env.envs[vector_idx].sim.config.yaml_data
            # attempting to avoid memory leaks, etc.
            del base_env.vector_env.envs[env_index].sim
            # Disable pygame and deactivate rendering mode
            sim_data["simulation"]["headless"] = True
            base_env.vector_env.envs[vector_idx].sim = FireSimulation(
                Config(config_dict=sim_data)
            )
            # base_env.vector_env.envs[vector_idx].sim.reset()
            # log.info(type(base_env.vector_env.envs[env_index].simulation._game))

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
        # if hasattr(self, "trial_results_path"):
        #     logger.warning("self.trial_results_path: %s", self.trial_results_path)
        # NOTE: We can access `algorithm.iteration` to get current episode/iteration
        # - maybe its best to just store this as a variable within the callback class?
        # something like: self.current_iteration = algorithm.iteration
        # algo_iter = algorithm.iteration
        # logger.warning("algorithm.logdir: %s", algorithm.logdir)
        # os.environ["TRIAL_RESULTS_PATH"] = algorithm.logdir
        # FIXME: move below into if statement
        # We want to render the eval environment at eps 10, 100, 1000, 10000, etc.
        # if log10(algo_iter).is_integer() and log10(algo_iter) > 0:
        func = _prepare_env_for_rendering
        # FIXME: `env_ctx` is not used in `func`, so we can use `foreach_env` method?
        #
        algorithm.evaluation_workers.foreach_worker(
            lambda w: w.foreach_env_with_context(func),
            local_worker=False,
        )

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
        # eval_metrics_keys = list(evaluation_metrics.keys())
        # Create environment variable that will be used to save the gif (s)
        # FIXME does logdir capture the correct dir if cli.mode=tune??
        # logdir = algorithm.logdir
        # # should it be episode_{iter + 1}??
        # save_dir = os.path.join(logdir, "gifs", f"episode_{algorithm.iteration}")
        # os.environ["GIF_SAVE_DIR"] = save_dir
        # fn = _prepare_env_after_rendering_and_save_gif
        # algorithm.evaluation_workers.foreach_env_with_context(fn)
        # Add gif paths to


def _prepare_env_for_rendering(base_env: BaseEnv, env_ctx: EnvContext):
    """Prepare the environment for rendering by enabling pygame."""
    # FIXME: Assumption to verify: method is NEVER called by local worker
    sim_data = base_env.sim.config.yaml_data
    sim_data["simulation"]["headless"] = False

    # Instantiate new simulation object with updated parameters
    del base_env.sim
    base_env.sim = FireSimulation(Config(config_dict=sim_data))
    # Set simulation to rendering mode
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    base_env.sim.rendering = True
