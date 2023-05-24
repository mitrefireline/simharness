import os
import logging
from math import log10
from typing import TYPE_CHECKING, Dict, Optional, Union

# from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.env.base_env import BaseEnv
from ray.rllib.env.env_context import EnvContext
from ray.rllib.evaluation.episode import Episode
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.rllib.policy import Policy
from ray.rllib.utils.typing import PolicyID  # AgentID, EnvType,
from simfire.sim.simulation import FireSimulation
from simfire.utils.config import Config
from simfire.utils.log import create_logger

if TYPE_CHECKING:
    from ray.rllib.algorithms.algorithm import Algorithm
    from ray.rllib.evaluation import RolloutWorker

log = create_logger(__name__)


class SetEnvSeedsCallback(DefaultCallbacks):
    """To use this callback, set {"callbacks": CustomCallback} in the algo config."""

    # def __init__(self, legacy_callbacks_dict: Dict[str, callable] = None):
    #     super().__init__(legacy_callbacks_dict)

    #     # Place holder variable that will be used to track episode iterations
    #     self.algorithm_iteration = -1

    # TODO: Can probably use this callback to log run params to aim, might be safer?
    def on_algorithm_init(
        self,
        *,
        algorithm: "Algorithm",
        **kwargs,
    ) -> None:
        """Callback run when a new algorithm instance has finished setup.
        This method gets called at the end of Algorithm.setup() after all
        the initialization is done, and before training actually starts.
        Args:
            algorithm: Reference to the trainer instance.
            kwargs: Forward compatibility placeholder.
        """
        # Initial value is expected to be set to -1, so update it here.
        # if self.algorithm_iteration == -1:
        #     self.algorithm_iteration = algorithm.iteration
        # algo_cfg = algorithm.config
        # if algorithm.config.env_config.get("simulation", None):
        #     render_sim = FireSimulation(Config(_SIMULATION_CONFIG_FILE))
        #     algorithm.config.env_config.update("simulation", render_sim)

        # try:
        #     # Use `FireSimulation` that can render with Pygame.
        #     render_sim = FireSimulation(Config(_SIMULATION_CONFIG_FILE))
        #     # algorithm.config["evaluation_config"]["env_config"].update(
        #     #     {"simulation": }
        #     # )
        # except BaseException as e:
        #     log.error(e)
        # if not algorithm._episode_history:
        #     log.warn("EPISODE_HISTORY IS EMPTY")
        pass

    # TODO: Can maybe use this callback to inspect/debug/verify environment parallelization
    # def on_sub_environment_created()
    def on_episode_created(
        self,
        *,
        worker: "RolloutWorker",
        base_env: BaseEnv,
        policies: Dict[PolicyID, Policy],
        env_index: int,
        episode: Union[Episode, EpisodeV2],
        **kwargs,
    ) -> None:
        """Callback run when a new episode is created (but has not started yet!).
        This method gets called after a new Episode(V2) instance is created to
        start a new episode. This happens before the respective sub-environment's
        (usually a gym.Env) `reset()` is called by RLlib.
        1) Episode(V2) created: This callback fires.
        2) Respective sub-environment (gym.Env) is `reset()`.
        3) Callback `on_episode_start` is fired.
        4) Stepping through sub-environment/episode commences.
        Args:
            worker: Reference to the current rollout worker.
            base_env: BaseEnv running the episode. The underlying
                sub environment objects can be retrieved by calling
                `base_env.get_sub_environments()`.
            policies: Mapping of policy id to policy objects. In single
                agent mode there will only be a single "default" policy.
            env_index: The index of the sub-environment that is about to be reset
                (within the vector of sub-environments of the BaseEnv).
            episode: The newly created episode. This is the one that will be started
                with the upcoming reset. Only after the reset call, the
                `on_episode_start` event will be triggered.
            kwargs: Forward compatibility placeholder.
        """
        log = logging.getLogger(__name__)
        in_evaluation = worker.policy_config["in_evaluation"]
        if in_evaluation:
            env = base_env.vector_env.envs[env_index]
            log.warning("beginning ON_EPISODE_CREATED callback...")
            log.warning(f"Current eval round: {env._current_eval_round}")

        # TODO find out what attribute of `episode` gives the iteration
        # if in_evaluation and log10(episode_iter).is_integer() and log10(episode_iter) > 0:
        # if in_evaluation:
        # base_env.vector_env.envs[env_index].simulation = FireSimulation(
        #     Config(_SIMULATION_VIEW_EVAL_CONFIG_FILE)
        # )
        # log.warn(f"WE ARE IN EVALUATION AT ITERATION {self.algorithm_iteration}")
        # We only want to save a .gif for the evaluation episodes
        # if in_evaluation:
        #     sim_cfg = base_env.vector_env.envs[env_index].simulation.config
        #     # Enable pygame
        #     base_env.vector_env.envs[
        #         env_index
        #     ].simulation.config.simulation.headless = False
        # sim.reset()

    # TODO: This callback can be used to correctly seed each sub-environment before the
    # sub-environment is `reset()` (ie. instead of setting seeds INSIDE of `reset()`).
    # TODO: This callback can be used to check if the current episode is an evaluation
    # episode via: worker.policy_config["in_evaluation"]. If yes, we need to figure out
    # a tidy way to set the env into render mode, and save the env in `on_episode_end()`
    def on_episode_start(
        self,
        *,
        worker: "RolloutWorker",
        base_env: BaseEnv,
        policies: Dict[PolicyID, Policy],
        episode: Union[Episode, EpisodeV2],
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
        # NOTE: To check if the `RolloutWorker`` is in evaluation mode, access
        # `worker.policy_config["in_evaluation"]`
        # NOTE: The underlying sub-environment objects can be retrieved by calling
        # `base_env.get_sub_environments()`.
        # log.warn("beginning ON_EPISODE_START callback...")
        # in_evaluation = worker.policy_config["in_evaluation"]
        # headless = base_env.vector_env.envs[
        #     env_index
        # ].simulation.config.simulation.headless
        # # Try to set the environment to rendering mode
        # if not headless and in_evaluation:
        #     log.warn(f"headless: {headless}; in_evaluation: {in_evaluation}")
        #     os.environ["SDL_VIDEODRIVER"] = "dummy"
        #     # sim == `FireSimulation`
        #     # Activate rendering mode
        #     base_env.vector_env.envs[env_index].simulation.rendering = True
        #     # additional logic to prepare simfire for rendering
        #     # sim.rendering = True
        # elif in_evaluation:
        #     log.debug(f"IN EVAL, BUT HEADLESS IS {headless}")

    # TODO: This callback can be used to call `save_gif()` on each evaluation env
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
        log = logging.getLogger(__name__)
        in_evaluation = worker.policy_config["in_evaluation"]
        if in_evaluation:
            env = base_env.vector_env.envs[env_index]
            log.warning("beginning ON_EPISODE_END callback...")
            log.warning(f"Current eval round: {env._current_eval_round}")
            # Reset the number of eval rounds
            if env._current_eval_round == env._total_eval_rounds:
                base_env.vector_env.envs[env_index]._current_eval_round = 1
            # Increment the number of eval rounds
            else:
                base_env.vector_env.envs[env_index]._current_eval_round += 1

        # env_ctx = worker.env_context
        # # worker_index = env_ctx.worker_index
        # vector_idx = env_ctx.vector_index
        # in_evaluation = worker.policy_config["in_evaluation"]
        # headless = base_env.vector_env.envs[
        #     vector_idx
        # ].simulation.config.simulation.headless
        # # Save a GIF and fire spread graph from the last episode
        # if not headless and in_evaluation and env_ctx.worker_index > 0:
        #     if not os.environ.get("SDL_VIDEODRIVER", None):
        #         log.warn("SETTING SDL_VIDEODRIVE... AGAIN.")
        #         os.environ["SDL_VIDEODRIVER"] = "dummy"
        #     id = f"worker_idx_{env_ctx.worker_index}_vector_idx_{vector_idx}"
        #     # FIXME what happens when cli.mode == tune??
        #     logdir = worker.config.logger_config["logdir"]
        #     # logdir = os.getcwd()
        #     save_path = os.path.join(logdir, "gifs", f"{id}.gif")
        #     # FIXME: Can we save each gif in a folder that relates it to episode iter?
        #     log.warn(f"SAVING GIF TO {save_path}")
        #     if hasattr(base_env.vector_env.envs[vector_idx].simulation, "_game"):
        #         base_env.vector_env.envs[vector_idx].simulation.save_gif(save_path)
        #     # Save the gif_path so that we can write image to aim server, if desired
        #     # NOTE: `save_path` is a list after the above, so do element access for now
        #     episode.media.update({id: save_path})
        #     # sim.save_spread_graph(save_dir)
        #     # attempting to avoid memory leaks, etc.
        #     # del base_env.vector_env.envs[env_index].simulation
        #     # Disable pygame and deactivate rendering mode
        #     base_env.vector_env.envs[vector_idx].simulation = FireSimulation(
        #         Config(_EVAL_SIM_CONFIG_MAP[env_ctx.worker_index])
        #     )
        #     base_env.vector_env.envs[vector_idx].simulation.reset()
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
        # NOTE: We can access `algorithm.iteration` to get current episode/iteration
        # - maybe its best to just store this as a variable within the callback class?
        # something like: self.current_iteration = algorithm.iteration
        # algo_iter = algorithm.iteration
        # FIXME: move below into if statement
        # We want to render the eval environment at eps 10, 100, 1000, 10000, etc.
        # if log10(algo_iter).is_integer() and log10(algo_iter) > 0:
        # fn = _prepare_env_for_rendering
        # algorithm.evaluation_workers.foreach_env_with_context(fn)

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
    # env_index = env_ctx.vector_index
    worker_idx = env_ctx.worker_index
    # if worker_idx > 0:
    #     # TODO: once below works, update simulation based on the worker, vector idxs
    #     # if isinstance(base_env, FireSimulation):
    #     base_env.simulation = FireSimulation(Config(_EVAL_SIM_CONFIG_MAP[worker_idx]))


# def _prepare_env_after_rendering_and_save_gif(base_env: BaseEnv, env_ctx: EnvContext):
#     # env_index = env_ctx.vector_index
#     # id = f"worker_idx_{env_ctx.worker_index}_vector_idx_{env_ctx.vector_index}.gif"
#     # # Save a GIF and fire spread graph from the last episode
#     # # TODO: once below works, update simulation based on the worker, vector idxs
#     # # if isinstance(base_env, FireSimulation):
#     # save_dir = os.environ.get("GIF_SAVE_DIR", None)
#     # if save_dir:
#     #     fname = os.path.join(save_dir, id)
#     #     base_env.simulation.save_gif(fname)
#     # sim.save_spread_graph(save_dir)

#     del base_env.simulation  # attempting to avoid memory leaks, etc.
#     # Disable pygame and deactivate rendering mode
#     base_env.simulation = FireSimulation(Config(_SIMULATION_EVAL_CONFIG_FILE))
