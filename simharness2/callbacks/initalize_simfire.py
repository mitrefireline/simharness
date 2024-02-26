from typing import TYPE_CHECKING, Dict, Any, Tuple
import logging
import time
from itertools import chain
from pprint import pformat

import ray
from ray import ObjectRef
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.env.base_env import BaseEnv
import simharness2.utils.utils as utils
import simharness2.utils.fire_data as fire_data
from simharness2.environments.harness import RLlibEnvContextMetadata

import numpy as np

if TYPE_CHECKING:
    from ray.rllib.algorithms.algorithm import Algorithm
    from ray.rllib.evaluation import RolloutWorker
    from ray.rllib.env.env_context import EnvContext

    from simfire.sim.simulation import FireSimulation
    from simharness2.environments.fire_harness import FireHarness

logger = logging.getLogger(__name__)


class InitializeSimfire(DefaultCallbacks):
    """Enables robust initialization for a `FireSimulation` object.

    This callback is intended to be used to initialize and reset the `FireSimulation`
    object stored under `ReactiveHarness.sim`.
    """

    def __init__(self, legacy_callbacks_dict: Dict[str, callable] = None):
        super().__init__(legacy_callbacks_dict=legacy_callbacks_dict)
        # This will be updated with the user provided value from the config file.
        self.data_object_refs: Dict[str, ObjectRef] = {"train": None, "eval": None}
        self.fire_pos_cfg: Dict[str, Any] = None
        # This will store each sampled fire position - the value will be the number of
        # times it has been sampled (ie. total episodes trained with this position).
        self.fire_pos_counter: Dict[Tuple[int, int], int] = {}
        # self._train_envs_per_worker: int
        # self._eval_envs_per_worker: int

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
        # Set `rllib_env_context` for each env (needed w/in `env._initialize_simfire`).
        all_workers = [algorithm.workers, algorithm.evaluation_workers]
        for worker in all_workers:
            worker.foreach_worker(
                lambda w: w.foreach_env_with_context(_set_harness_env_context),
                local_worker=True,
            )

        # TODO: Do we want to generate data using a deepcopy of `sim`?
        sim: "FireSimulation" = algorithm.config.env_config.get("sim")
        _check_fire_init_pos_is_static(sim)
        fire_pos_cfg = algorithm.config.env_config.get("fire_initial_position")
        self.fire_pos_cfg = _validate_fire_init_config(fire_pos_cfg, sim.fire_map.size)

        # Retrieve the train/eval data using the provided fire initial position config.
        logdir = algorithm.logdir
        train_data, eval_data = _prepare_fire_map_data(sim, self.fire_pos_cfg, logdir)

        # Final check to ensure the sample size is valid wrt the number of workers/envs.
        self._check_sample_size_vs_workers(algorithm)

        # Initialize the `FireSimulation` for each training rollout.
        # Generate new indices randomly, w/o replacement, then create the array subset.
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

    def on_episode_start(
        self,
        *,
        worker: "RolloutWorker",
        base_env: BaseEnv,
        env_index: int,
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
        env = base_env.get_sub_environments()[env_index]
        # breakpoint()

    def on_episode_end(
        self,
        *,
        worker: "RolloutWorker",
        base_env: BaseEnv,
        env_index: int,
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
        env = base_env.get_sub_environments()[env_index]
        # breakpoint()

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


# TODO: Move this method to a more "general" location; it's a utility!
def _set_harness_env_context(harness: "FireHarness", env_context: "EnvContext"):
    """Add the provided env context to the harness."""
    # Extract rllib metadata from the env context.
    w_idx, v_idx = env_context.worker_index, env_context.vector_index
    remote, recreated_worker = env_context.remote, env_context.recreated_worker
    num_workers = env_context.num_workers
    # Create a new `RLlibEnvContextMetadata` object and add it to the harness.
    env_context_data = RLlibEnvContextMetadata(
        worker_index=w_idx,
        vector_index=v_idx,
        remote=remote,
        num_workers=num_workers,
        recreated_worker=recreated_worker,
    )
    harness.rllib_env_context = env_context_data


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
