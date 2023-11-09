"""Enables rendering eval environments and initializing the `FireSimulation.fire_map`."""
import logging
import os
import time
from itertools import chain
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple, Union

import numpy as np
import ray
from ray import ObjectRef
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.env.base_env import BaseEnv
from ray.rllib.evaluation.episode import Episode
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.rllib.policy import Policy
from ray.rllib.utils.typing import PolicyID  # AgentID, EnvType,
from pprint import pformat

import simharness2.utils.fire_data as fire_data
import simharness2.utils.utils as utils

if TYPE_CHECKING:
    from ray.rllib.algorithms.algorithm import Algorithm
    from ray.rllib.evaluation import RolloutWorker
    from simfire.sim.simulation import FireSimulation

    from simharness2.environments.reactive import ReactiveHarness


logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
handler.setFormatter(
    logging.Formatter("%(asctime)s\t%(levelname)s %(filename)s:%(lineno)s -- %(message)s")
)
logger.addHandler(handler)
# logger.propagate = False


class DoEverything(DefaultCallbacks):
    """Enables env rendering and robust initialization for a `FireSimulation` object.

    This callback is intended to be used to initialize and reset the `FireSimulation`
    object stored under `ReactiveHarness.sim`. Additionally, this callback will handle
    setting the `ReactiveHarness.sim` into rendering mode when desired for eval envs.
    """

    def __init__(self, legacy_callbacks_dict: Dict[str, callable] = None):
        super().__init__(legacy_callbacks_dict=legacy_callbacks_dict)
        # This will be updated with the user provided value from the config file.
        self.data_object_refs: Dict[str, ObjectRef] = {"train": None, "eval": None}
        self.fire_pos_cfg: Dict[str, Any] = None
        # This will store each sampled fire position - the value will be the number of
        # times it has been sampled (ie. total episodes trained with this position).
        self.fire_pos_counter: Dict[Tuple[int, int], int] = {}

    def on_algorithm_init(
        self,
        *,
        algorithm: "Algorithm",
        **kwargs,
    ) -> None:
        """Sets initial state of each `ReactiveHarness.sim` object across rollouts.

        Approach:
        - Sample from eval dataset and update the underlying `FireSimulation` on each
          evaluation rollout, `algorithm.evaluation_workers`. The scenarios used to
          evaluate agent performance should be fixed across the trial.
        - Sample from train dataset and update the underlying `FireSimulation` on each
          training rollout, `algorithm.workers`. This serves as the "initialization", and
          training fires will be updated after every `resample_interval` episodes.

        NOTE: This method is called at the end of `Algorithm.setup()`, after all the
        initialization is done, and before training actually starts.

        Arguments:
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

        # TODO: Do we want to generate data using a deepcopy of `sim`?
        sim: "FireSimulation" = algorithm.config.env_config.get("sim")
        _check_fire_init_pos_is_static(sim)
        fire_pos_cfg = algorithm.config.env_config.get("fire_initial_position")
        self.fire_pos_cfg = _validate_fire_init_config(fire_pos_cfg, sim.fire_map.size)

        # Retrieve the train/eval data using the provided fire initial position config.
        # TODO: Save the train/eval data to disk, ie. as `.npy` files.
        train_data, eval_data = _prepare_fire_map_data(sim, self.fire_pos_cfg, logdir)

        # Final check to ensure the sample size is valid wrt the number of workers/envs.
        self._check_sample_size_vs_workers(algorithm)

        # Initialize the `FireSimulation` for each training rollout.
        # Generate new indices randomly, w/o replacement, then create the array subset.
        train_indices = np.random.choice(
            len(train_data), size=self.train_sample_size, replace=False
        )
        train_subset = train_data[train_indices]
        self._train_envs_per_worker: int = algorithm.config.num_envs_per_worker
        pos_used = algorithm.workers.foreach_worker(
            lambda w: w.foreach_env(
                lambda env: env._initialize_simfire(
                    train_subset, self._train_envs_per_worker
                )
            ),
            local_worker=True,  # FIXME: Should this be True?
        )

        # TODO: Optimize this to scale when sample size is large.
        for pos in chain(*pos_used):
            if self.fire_pos_counter.get(pos):
                self.fire_pos_counter[pos] += 1
            else:
                self.fire_pos_counter[pos] = 1

        assert len(self.fire_pos_counter) == len(train_subset)
        total_pos = sum(self.fire_pos_counter.values())
        assert total_pos == utils.get_total_training_envs(algorithm)
        logger.debug(f"self.fire_pos_counter: \n{pformat(self.fire_pos_counter)}")

        # Initialize the `FireSimulation` for each evaluation rollout.
        # Generate new indices randomly, w/o replacement, then create the array subset.
        eval_indices = np.random.choice(
            len(eval_data), size=self.eval_sample_size, replace=False
        )
        eval_subset = eval_data[eval_indices]
        self._eval_envs_per_worker = algorithm.config.evaluation_config.get(
            "num_envs_per_worker"
        )
        if self._eval_envs_per_worker is None:
            self._eval_envs_per_worker = algorithm.config.num_envs_per_worker
        algorithm.evaluation_workers.foreach_worker(
            lambda w: w.foreach_env(
                lambda env: env._initialize_simfire(
                    eval_subset, self._eval_envs_per_worker
                )
            ),
            local_worker=False,  # FIXME: Should this be True?
        )

        # Put data into the distributed object store, and store the respective refs.
        self.data_object_refs["train"] = ray.put(train_data)
        self.data_object_refs["eval"] = ray.put(eval_data)

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
        env: ReactiveHarness = base_env.get_sub_environments()[env_index]

        if worker.config.in_evaluation:
            logger.info("Creating evaluation episode...")
            # Ensure the evaluation env is rendering mode, if it should be.
            if env._should_render and not env.sim.rendering:
                logger.info("Enabling rendering for evaluation env.")
                # TODO: Refactor below 3 lines into `env.render()` method?
                os.environ["SDL_VIDEODRIVER"] = "dummy"
                base_env.get_sub_environments()[env_index].sim.reset()
                base_env.get_sub_environments()[env_index].sim.rendering = True
            elif not env._should_render and env.sim.rendering:
                logger.error(
                    "Simulation is in rendering mode, but `env._should_render` is False."
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
        env: ReactiveHarness = base_env.get_sub_environments()[env_index]
        # Save a GIF from the last episode
        # TODO: Do we also want to save the fire spread graph?
        if worker.config.in_evaluation:
            logdir = env._trial_results_path
            eval_iters = env._num_eval_iters
            # Check if there is a gif "ready" to be saved
            if env._should_render and env.sim.rendering:
                # FIXME Update logic to handle saving same gif when writing to Aim UI
                context_dict = {}
                lat, lon = env.sim.config.landfire_lat_long_box.points[0]
                op_data_lat_lon = f"operational_lat_{lat}_lon_{lon}"
                fire_init_pos = env.sim.config.fire.fire_initial_position
                context_dict.update({"fire_initial_position": fire_init_pos})
                # FIXME Use nested structure for dir (gifs/<op_loc>/<fire_init_pos>)?
                gif_save_path = os.path.join(
                    logdir, "gifs", f"eval_iter_{eval_iters}.gif"
                )
                # FIXME: Can we save each gif in a folder that relates it to episode iter?
                logger.info(f"Saving GIF to {gif_save_path}...")
                base_env.get_sub_environments()[env_index].sim.save_gif(gif_save_path)
                # Save the gif_path so that we can write image to aim server, if desired
                # NOTE: `save_path` is a list after the above; do element access for now
                logger.debug(f"Type of gif_save_path: {type(gif_save_path)}")
                gif_data = {
                    "path": gif_save_path,
                    "name": op_data_lat_lon,
                    "step": eval_iters,
                    # "epoch":
                    "context": context_dict,
                }
                episode.media.update({"gif_data": gif_data})

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

        # TODO: Use a function to decide if this round should be rendered (ie log10).
        # Disable the evaluation environment (s) to be rendered.
        algorithm.evaluation_workers.foreach_worker(
            lambda w: w.foreach_env(lambda env: env._configure_env_rendering(False)),
            local_worker=False,
        )

    def on_train_result(
        self,
        *,
        algorithm: "Algorithm",
        result: dict,
        **kwargs,
    ) -> None:
        """Updates initial state of each `ReactiveHarness.sim` object across rollouts.

        Approach:
        - Check whether `resample_interval` episodes have past, and if so, get a (new)
          sample of size `sample_size` from the train dataset. Then, distribute chosen
          scenarios across the training rollouts.

        Arguments:
            algorithm: Current Algorithm instance.
            result: Dict of results returned from Algorithm.train() call.
                You can mutate this object to add additional metrics.
            kwargs: Forward compatibility placeholder.
        """
        curr_iter = algorithm.iteration
        # Only re-initialize the `FireSimulation` when the resample interval is met.
        if curr_iter % self.resample_interval == 0:
            logger.info(
                f"Re-initializing each simulation after training iteration: {curr_iter}"
            )
            train_data = ray.get(self.data_object_refs["train"])
            # Generate new indices randomly, w/o replacement, then create the arr subset.
            # TODO: Would shuffling `train_data` and then sampling be more robust?
            train_indices = np.random.choice(
                len(train_data), size=self.train_sample_size, replace=False
            )
            train_subset = train_data[train_indices]
            self._train_envs_per_worker = algorithm.config.num_envs_per_worker
            pos_used = algorithm.workers.foreach_worker(
                lambda w: w.foreach_env(
                    lambda env: env._initialize_simfire(
                        train_subset, self._train_envs_per_worker
                    )
                ),
                local_worker=True,  # FIXME: Should this be True?
            )

            # TODO: Optimize this to scale when sample size is large.
            for pos in chain(*pos_used):
                if self.fire_pos_counter.get(pos):
                    self.fire_pos_counter[pos] += 1
                else:
                    self.fire_pos_counter[pos] = 1

            logger.debug(f"self.fire_pos_counter: \n{pformat(self.fire_pos_counter)}")

            # Put data back into the distributed object store and store the ref.
            self.data_object_refs["train"] = ray.put(train_data)

    @property
    def resample_interval(self) -> int:
        """The number of training iterations between resampling the train dataset."""
        return self.fire_pos_cfg.get("sampler").get("resample_interval")

    @property
    def train_sample_size(self) -> int:
        """The number of scenarios to sample from the train dataset."""
        return self.fire_pos_cfg.get("sampler").get("sample_size").get("train")

    @property
    def eval_sample_size(self) -> int:
        """The number of scenarios to sample from the eval dataset."""
        return self.fire_pos_cfg.get("sampler").get("sample_size").get("eval")

    def _check_sample_size_vs_workers(self, algorithm: "Algorithm") -> None:
        """Ensure the sample size is valid wrt the number of workers/envs.

        NOTE: Currently, this method only checks the sample size wrt the number of
        expected total workers for training and evaluation. All workers are assumed to
        be healthy. Next iteration should leverage `WorkerSet.num_healthy_workers()`.
        """
        # Get the total number of training and evaluation envs.
        train_envs = utils.get_total_training_envs(algorithm)
        eval_envs = utils.get_total_evaluation_envs(algorithm)

        logger.debug(f"Total number of training envs: {train_envs}")
        logger.debug(f"Total number of evaluation envs: {eval_envs}")

        # Check training sample size.
        if self.train_sample_size > train_envs:
            msg = (
                "Invalid value for `sampler.sample_size.train`: "
                f"{self.train_sample_size}. The value cannot be greater than the "
                f"number of training envs, which is {train_envs}. Either decrease "
                "the value of `sampler.sample_size.train` or increase the number of "
                "training envs with `rollouts.num_rollout_workers` and/or "
                "`rollouts.num_envs_per_worker."
            )
            raise ValueError(msg)
        elif self.train_sample_size < train_envs:
            logger.warning(
                "The number of training envs is greater than the number of scenarios "
                "to sample from the train dataset. This will result in some scenarios "
                "appearing more than once in collected sample batches of experiences."
            )
        # Check evaluation sample size.
        if self.eval_sample_size > eval_envs:
            msg = (
                "Invalid value for `sampler.sample_size.eval`: "
                f"{self.eval_sample_size}. The value cannot be greater than the "
                f"number of evaluation envs, which is {eval_envs}. Either decrease "
                "the value of `sampler.sample_size.eval` or increase the number of "
                "evaluation envs with `evaluation.num_evaluation_workers` and "
                "`evaluation.evaluation_duration`."
            )
            raise ValueError(msg)
        elif self.eval_sample_size < eval_envs:
            logger.warning(
                "The number of evaluation envs is greater than the number of scenarios "
                "to sample from the eval dataset. This will result in some scenarios "
                "appearing more than once in collected sample batches of experiences."
            )


def _check_fire_init_pos_is_static(sim: "FireSimulation") -> None:
    """Ensure the `fire.fire_initial_position.type` is static."""
    fire_init_pos_type = sim.config.yaml_data["fire"]["fire_initial_position"]["type"]
    if fire_init_pos_type != "static":
        msg = (
            "Invalid value for `fire.fire_initial_position.type`: "
            f"{fire_init_pos_type}. The value must be `static`."
        )
        raise ValueError(msg)


def _validate_fire_init_config(
    fire_pos_cfg: Dict[str, Any], fire_map_size: int
) -> Dict[str, Any]:
    """Ensure the required environment configuration information has been provided."""
    if fire_pos_cfg is None:
        # TODO: Add more descriptive message about where to update the config.
        msg = (
            "The `fire_initial_position` key must be provided to use this callback. "
            "This should be specified under the "
            "`environment.env_config.fire_initial_position` key."
        )
        raise ValueError(msg)
    elif fire_pos_cfg.get("generator") is None:
        # TODO: Add more descriptive message about where to update the config.
        msg = (
            "The `generator` key must be provided to use this callback. Enable "
            "`generator` for generating dataset of fire start locations to sample from."
        )
        raise ValueError(msg)
    elif fire_pos_cfg.get("sampler") is None:
        # TODO: Add more descriptive message about where to update the config.
        msg = (
            "The `sampler` key must be provided to use this callback. Enable "
            "`sampler` to control sampling of new fire start locations."
        )
        raise ValueError(msg)
    # Provided configuration is valid, so return it.
    else:
        # Ensure sampling config is valid wrt the expected "dataset" to be generated.
        # TODO: hydra should ENFORCE the existence of the `output_size` key.
        generator_output_size = fire_pos_cfg["generator"].get("output_size")
        if fire_pos_cfg["generator"].get("make_all_positions"):
            generator_output_size = fire_map_size

        sampler_population_size = fire_pos_cfg["sampler"].get("population_size")
        if sampler_population_size is not None:
            # TODO: hydra should ENFORCE the existence of the `train` key.
            train_sample_size = fire_pos_cfg["sampler"].get("sample_size").get("train")
            if generator_output_size < sampler_population_size:
                msg = (
                    "Invalid value for `sampler.population_size`: "
                    f"{sampler_population_size}. The value cannot be greater than the "
                    f"`generator.output_size`, which is {generator_output_size}."
                )
                raise ValueError(msg)
            elif sampler_population_size < train_sample_size:
                msg = (
                    "Invalid value for `sampler.sample_size.train`: "
                    f"{train_sample_size}. The value cannot be greater than the "
                    f"`sampler.population_size`, which is {sampler_population_size}."
                )
                raise ValueError(msg)
        return fire_pos_cfg


def _prepare_fire_map_data(
    sim: "FireSimulation", fire_pos_cfg: Dict[str, Any], logdir: str = None
) -> Tuple[np.recarray, np.recarray]:
    """Prepare the fire map data for the environment."""
    generator_cfg = fire_pos_cfg.get("generator")
    sampler_cfg = fire_pos_cfg.get("sampler")

    # Generate the dataset using the provided configuration for `generator`.
    start_time = time.time()
    fire_df = fire_data.generate_fire_initial_position_data(sim, **generator_cfg)
    end_time = time.time()
    total_runtime = end_time - start_time
    logger.debug(f"Total generator runtime: {total_runtime} seconds.")
    logger.debug(f"Total generator runtime: {total_runtime/60:.2f} minutes")

    # Down sample the dataset using the provided configuration for `sampler`.
    return fire_data.filter_fire_initial_position_data(
        fire_df=fire_df, logdir=logdir, **sampler_cfg
    )


# import simfire.utils.config as simfire_cfg


# def _fire_initial_position_data_is_generated(
#     sim_cfg: simfire_cfg.Config,
#     save_path: str,
#     output_size: int = 1,
#     make_all_positions: bool = False,
# ):
#     """Check if the fire initial position data has been generated."""
