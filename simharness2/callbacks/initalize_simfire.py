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
        self.op_locs_cfg: Dict[str, Any] = None
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
        # Validate the configuration for the `FireSimulation` object.
        _check_terrain_is_operational(sim)
        _check_fire_init_pos_is_static(sim)

        # NOTE: We are not doing any validation of the provided op_locs config.
        op_locs_cfg = algorithm.config.env_config.get("operational_locations")
        self.op_locs_cfg = op_locs_cfg

        fire_pos_cfg = algorithm.config.env_config.get("fire_initial_position")
        self.fire_pos_cfg = _validate_fire_init_config(fire_pos_cfg, sim.fire_map.size)

        # Ensure number of scenarios to sample is valid wrt number of workers/envs.
        self._check_sample_size_vs_workers(algorithm)

        # Retrieve the train/eval data using the provided fire initial position config.
        logdir = algorithm.logdir
        train_data, eval_data = _prepare_fire_map_data(sim, self.fire_pos_cfg, logdir)

        # Initialize the `FireSimulation` for each training rollout.
        # Generate new indices randomly, w/o replacement, then create the array subset.
        train_indices = np.random.choice(
            len(train_data), size=self.train_scenarios_per_location, replace=False
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
            len(eval_data), size=self.eval_scenarios_per_location, replace=False
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
        # NOTE: Allow user to set value to -1 to disable resampling.
        if self.resample_interval == -1:
            logger.debug(
                "The `resample_interval` is set to -1, so the current train scenarios "
                "will be used for the entire training process."
            )
            return
        # Only re-initialize the `FireSimulation` when the resample interval is met.
        if curr_iter % self.resample_interval == 0:
            logger.info(
                f"Re-initializing each simulation after training iteration: {curr_iter}"
            )
            train_data = ray.get(self.data_object_refs["train"])
            # Generate new indices randomly, w/o replacement, then create the arr subset.
            # TODO: Would shuffling `train_data` and then sampling be more robust?
            train_indices = np.random.choice(
                len(train_data), size=self.train_scenarios_per_location, replace=False
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
            eval_workers = algorithm.evaluation_workers


    @property
    def resample_interval(self) -> int:
        """The number of training iterations between resampling the train dataset."""
        return self.fire_pos_cfg.get("sampler").get("resample_interval")

    @property
    def train_scenarios_per_location(self) -> int:
        """The number of scenarios to sample from train dataset for each location."""
        return self.fire_pos_cfg.get("sampler").get("sample_size").get("train")

    @property
    def eval_scenarios_per_location(self) -> int:
        """The number of scenarios to sample from eval dataset for each location."""
        return self.fire_pos_cfg.get("sampler").get("sample_size").get("eval")

    @property
    def total_train_scenarios(self) -> int:
        """The total number of fire scenarios to use for each training iteration."""
        num_op_locs = self.op_locs_cfg.get("sample_size").get("train")
        return self.train_scenarios_per_location * num_op_locs

    @property
    def total_eval_scenarios(self) -> int:
        """The total number of fire scenarios to use for each evaluation iteration."""
        num_op_locs = self.op_locs_cfg.get("sample_size").get("eval")
        return self.eval_scenarios_per_location * num_op_locs

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
        if self.total_train_scenarios > train_envs:
            msg = (
                "The total number of training scenarios ({}) cannot exceed the number "
                "of training environments ({}). The total number of training scenarios "
                "is calculated as the product of the number of locations "
                "(`simulation.operational_location.sample_size.train`) and the number "
                "of scenarios per location "
                "(`simulation.fire_initial_position.sampler.sample_size.train)."
            ).format(self.total_train_scenarios, train_envs)
            raise ValueError(msg)
        elif self.total_train_scenarios < train_envs:
            msg = (
                "The number of training environments ({}) is greater than the total "
                "number of training scenarios ({}). This will result in some scenarios "
                "appearing more than once in collected sample batches of experiences. "
                "Consider increasing the number of locations "
                "(`simulation.operational_location.sample_size.train`) or the number of "
                "scenarios per location "
                "(`simulation.fire_initial_position.sampler.sample_size.train) to increase "
                "the total number of training scenarios."
            ).format(train_envs, self.total_train_scenarios)
            logger.warning(msg)
        # Check evaluation sample size.
        if self.total_eval_scenarios > eval_envs:
            msg = (
                "The total number of eval scenarios ({}) cannot exceed the number "
                "of eval environments ({}). The total number of evaluation scenarios "
                "is calculated as the product of the number of locations "
                "(`simulation.operational_location.sample_size.eval`) and the number "
                "of scenarios per location "
                "(`simulation.fire_initial_position.sampler.sample_size.eval)."
            )
            raise ValueError(msg)
        elif self.total_eval_scenarios < eval_envs:
            msg = (
                "The number of evaluation environments ({}) is greater than the total "
                "number of evaluation scenarios ({}). This will result in some scenarios "
                "appearing more than once in collected sample batches of experiences. "
                "Consider increasing the number of locations "
                "(`simulation.operational_location.sample_size.eval`) or the number of "
                "scenarios per location "
                "(`simulation.fire_initial_position.sampler.sample_size.eval) to increase "
                "the total number of evaluation scenarios."
            ).format(eval_envs, self.total_eval_scenarios)
            logger.warning(msg)


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


def _check_terrain_is_operational(sim: "FireSimulation") -> None:
    """Ensure `topography.type` and `fuel.type` are operational for `terrain`."""
    topo_type = sim.config.yaml_data["terrain"]["topography"]["type"]
    fuel_type = sim.config.yaml_data["terrain"]["fuel"]["type"]
    if topo_type != "operational" or fuel_type != "operational":
        msg = (
            "Invalid value for `terrain.topography.type` or `terrain.fuel.type`: "
            f"{topo_type} and {fuel_type}, respectively. The values must BOTH be "
            "`operational`."
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
