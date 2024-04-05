from typing import TYPE_CHECKING, Dict, Any, List, Tuple
import logging
from itertools import chain
from pprint import pformat
import json
import random

import numpy as np
import ray
from ray import ObjectRef
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.env.base_env import BaseEnv

import simharness2.utils.utils as utils
from simharness2.environments.fire_harness import BurnMDOperationalLocation
import simharness2.environments.utils as env_utils

if TYPE_CHECKING:
    from ray.rllib.algorithms.algorithm import Algorithm
    from simfire.sim.simulation import FireSimulation


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
                lambda w: w.foreach_env_with_context(env_utils.set_harness_env_context),
                local_worker=True,
            )

        # TODO: Do we want to generate data using a deepcopy of `sim`?
        sim: "FireSimulation" = algorithm.config.env_config.get("sim")
        # Validate the configuration for the `FireSimulation` object.
        env_utils.check_terrain_is_operational(sim)
        env_utils.check_fire_init_pos_is_static(sim)

        # NOTE: We are not doing any validation of the provided op_locs config.
        op_locs_cfg = algorithm.config.env_config.get("operational_locations")
        self.op_locs_cfg = op_locs_cfg
        fire_pos_cfg = algorithm.config.env_config.get("fire_initial_position")
        self.fire_pos_cfg = env_utils.validate_fire_init_config(
            fire_pos_cfg, sim.fire_map.size
        )
        # Ensure number of scenarios to sample is valid wrt number of workers/envs.
        self._check_sample_size_vs_workers(algorithm)
        self._train_envs_per_worker = algorithm.config.num_envs_per_worker
        self._eval_envs_per_worker = algorithm.config.evaluation_config.get(
            "num_envs_per_worker"
        )
        if self._eval_envs_per_worker is None:
            self._eval_envs_per_worker = algorithm.config.num_envs_per_worker

        # TODO: Add check to ensure each location is valid. For more info, see:
        # https://github.com/mitrefireline/simfire/blob/0d46451db183a58d209ef789c509f00eca0daedf/simfire/utils/config.py#L306
        # Seed each respective env with the operational locations.
        train_locs, eval_locs = self.get_operational_locations()

        # TODO: Do we want to 'track' locations used for training and evaluation?
        train_locs_used = algorithm.workers.foreach_worker(
            lambda w: w.foreach_env(
                lambda env: env._set_operational_location(
                    locations=train_locs,
                    num_envs_per_worker=self._train_envs_per_worker,
                )
            ),
            local_worker=True,  # FIXME: Should this be True?
        )
        eval_locs_used = algorithm.evaluation_workers.foreach_worker(
            lambda w: w.foreach_env(
                lambda env: env._set_operational_location(
                    locations=eval_locs,
                    num_envs_per_worker=self._eval_envs_per_worker,
                )
            ),
            local_worker=False,  # FIXME: Should this be True?
        )

        train_locs_set = set()
        for loc in chain(*train_locs_used):
            train_locs_set.add(loc)

        eval_locs_set = set()
        for loc in chain(*eval_locs_used):
            eval_locs_set.add(loc)

        # TODO: Final step - for each location, prepare the fire data and store it!
        # FIXME: Need to refactor login in `prepare_fire_map_data()`; it returns both
        # train and eval data, but we only need to store the data for each set.

        # FIXME: Rather than preparing data based on train/eval, we should prepare data
        # for each location, and then distribute the data to the respective envs.

        # Retrieve the train/eval data using the provided fire initial position config.
        logdir = algorithm.logdir
        train_data, eval_data = env_utils.prepare_fire_map_data(
            sim,
            self.fire_pos_cfg,
            logdir,
            locations=train_locs,
        )

        # Initialize the `FireSimulation` for each training rollout.
        # Generate new indices randomly, w/o replacement, then create the array subset.
        train_indices = np.random.choice(
            len(train_data), size=self.num_train_fire_init_pos, replace=False
        )
        train_subset = train_data[train_indices]
        pos_used = algorithm.workers.foreach_worker(
            lambda w: w.foreach_env(
                lambda env: env._initialize_simfire(
                    data=train_subset,
                    num_envs_per_worker=self._train_envs_per_worker,
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
            len(eval_data), size=self.num_eval_fire_init_pos, replace=False
        )
        eval_subset = eval_data[eval_indices]
        algorithm.evaluation_workers.foreach_worker(
            lambda w: w.foreach_env(
                lambda env: env._initialize_simfire(
                    data=eval_subset,
                    num_envs_per_worker=self._eval_envs_per_worker,
                )
            ),
            local_worker=False,  # FIXME: Should this be True?
        )

        # Put data into the distributed object store, and store the respective refs.
        self.data_object_refs["train"] = ray.put(train_data)
        self.data_object_refs["eval"] = ray.put(eval_data)

    def get_operational_locations(
        self,
    ) -> Tuple[List[BurnMDOperationalLocation], List[BurnMDOperationalLocation]]:
        """Sample operational locations from the BurnMD dataset.

        If the `independent_eval_locations` flag is set to `True` in the provided
        operational locations config, then it is guranteed that the training and
        evaluation locations will be mutually exclusive. Otherwise, the locations used
        for evaluation may overlap with those used for training. In the latter case, it
        is recommended to use mutually exclusive fire initial positions for training and
        evaluation.

        NOTE: If the `independent_eval_locations` flag is not provided, the default
        behavior is to use distinct locations for training and evaluation.

        FIXME: I see 2 options for sampling locs when `independent_eval_locations` is
        False. Option 1 is to ensure eval_locs.issubset(train_locs), ie. we only evaluate
        on locations that we have trained on. Option 2 is to allow eval_locs to be a
        superset of train_locs, ie. we evaluate on locations that we may not have trained
        on. We should decide which option to go with - for now, using option 1. An error
        will be raised if num_eval_locs > num_train_locs.

        TODO: Is above default behavior what we want? Should we enforce the flag?
        TODO: Add support for sampling locations from a custom dataset.
        TODO: Maybe move this method to `simharness2.environments.utils`?

        Returns:
            train_locs: List of `BurnMDOperationalLocation` objects for training.
            eval_locs: List of `BurnMDOperationalLocation` objects for evaluation.
        """
        # Load BurnMD data to use for sampling random operational locations.
        burnmd_fp = self.op_locs_cfg.get("burnmd_dataset_path")
        logger.info(f"Loading BurnMD data from {burnmd_fp}")
        with open(burnmd_fp, "r", encoding="utf-8") as j:
            burnmd_op_locs = json.loads(j.read())

        # Get the total number of locations to sample
        independent_eval_locs = self.op_locs_cfg.get("independent_eval_locations", True)
        if independent_eval_locs:
            total_locations = self.num_train_locations + self.num_eval_locations
        else:
            # Address FIXME in docstr; for now, only evaluate on locations we train on
            # and instead, let different fire initial positions create data diversity.
            if self.num_eval_locations > self.num_train_locations:
                raise ValueError(
                    "The number of evaluation locations cannot exceed the number of "
                    "training locations when `independent_eval_locations` is False. This "
                    "ensures that the evaluation locations are a subset of the training "
                    "locations."
                )
            # Use the maximum of the two values to ensure we have enough locations.
            total_locations = max(self.num_train_locations, self.num_eval_locations)

        # Randomly sample keys from the dictionary
        logger.info(f"Sampling {total_locations} operational locations from BurnMD...")
        sampled_keys = random.sample(list(burnmd_op_locs.keys()), total_locations)

        # Split the sampled keys into train and eval sets
        if independent_eval_locs:
            train_keys = sampled_keys[: self.num_train_locations]
            eval_keys = sampled_keys[self.num_train_locations :]
        else:
            train_keys = sampled_keys
            # Set eval keys to be randomly sampled from train keys.
            eval_keys = random.sample(train_keys, self.num_eval_locations)

        # Extract the corresponding values from the dictionary
        train_locations = {key: burnmd_op_locs[key] for key in train_keys}
        eval_locations = {key: burnmd_op_locs[key] for key in eval_keys}

        # Build the BurnMDOperationalLocation objects
        train_locs = [
            BurnMDOperationalLocation(uid=uid, **loc_data)
            for uid, loc_data in train_locations.items()
        ]
        eval_locs = [
            BurnMDOperationalLocation(uid=uid, **loc_data)
            for uid, loc_data in eval_locations.items()
        ]
        logger.info(f"Sampled training locations: \n{pformat(train_locs)}")
        logger.info(f"Sampled evaluation locations: \n{pformat(eval_locs)}")
        return train_locs, eval_locs

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
                len(train_data), size=self.num_train_fire_init_pos, replace=False
            )
            train_subset = train_data[train_indices]
            pos_used = algorithm.workers.foreach_worker(
                lambda w: w.foreach_env(
                    lambda env: env._initialize_simfire(
                        data=train_subset,
                        num_envs_per_worker=self._train_envs_per_worker,
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
    def num_train_fire_init_pos(self) -> int:
        """Number of fire initial positions to sample for each training location."""
        return self.fire_pos_cfg.get("sampler").get("sample_size").get("train")

    @property
    def num_eval_fire_init_pos(self) -> int:
        """Number of fire initial positions to sample for each evaluation location."""
        return self.fire_pos_cfg.get("sampler").get("sample_size").get("eval")

    @property
    def num_train_locations(self) -> int:
        """The number of operational locations to sample from for training."""
        return self.op_locs_cfg.get("sample_size").get("train")

    @property
    def num_eval_locations(self) -> int:
        """The number of operational locations to sample from for evaluation."""
        return self.op_locs_cfg.get("sample_size").get("eval")

    @property
    def total_train_scenarios(self) -> int:
        """The total number of fire scenarios to use for each training iteration."""
        return self.num_train_fire_init_pos * self.num_train_locations

    @property
    def total_eval_scenarios(self) -> int:
        """The total number of fire scenarios to use for each evaluation iteration."""
        return self.num_train_fire_init_pos * self.num_eval_locations

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
                "number of training scenarios ({}). This will result in some "
                "scenarios appearing more than once in collected sample batches of "
                "experiences. Consider increasing the number of locations "
                "(`simulation.operational_location.sample_size.train`) or the number of "
                "scenarios per location "
                "(`simulation.fire_initial_position.sampler.sample_size.train) to "
                "increase the total number of training scenarios."
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
                "number of evaluation scenarios ({}). This will result in some "
                "scenarios appearing more than once in collected sample batches of "
                "experiences. Consider increasing the number of locations "
                "(`simulation.operational_location.sample_size.eval`) or the number of "
                "scenarios per location "
                "(`simulation.fire_initial_position.sampler.sample_size.eval) to "
                "increase the total number of evaluation scenarios."
            ).format(eval_envs, self.total_eval_scenarios)
            logger.warning(msg)
