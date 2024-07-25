"""
TODO: Maintain naming consistency throughout this file. Ie. for operational locations,
we should fix a convention and ensure this is documented (op vs. operational), etc.
TODO: Decide if we are happy with current usage of properties. If not, refactor.
"""

from typing import TYPE_CHECKING, Dict, Any, List, Literal, Optional, Tuple, Union
import logging
from itertools import chain
from pprint import pformat
import os
import json

import numpy as np
import ray
from ray import ObjectRef
from ray.rllib.algorithms.callbacks import DefaultCallbacks

from simharness2.utils import utils
from simharness2.utils import simfire as simfire_utils
from simharness2.environments import utils as env_utils

if TYPE_CHECKING:
    from ray.rllib.algorithms.algorithm import Algorithm
    from ray.rllib.evaluation.worker_set import WorkerSet
    from simharness2.environments.utils import BurnMDOperationalLocation, EnvFireContext
    from numpy.random import Generator


logger = logging.getLogger(__name__)


class InitializeSimfire(DefaultCallbacks):
    """Enables robust initialization for a `FireSimulation` object.

    This callback is intended to be used to initialize and reset the `FireSimulation`
    object stored under `ReactiveHarness.sim`.
    """

    def __init__(self, legacy_callbacks_dict: Dict[str, callable] = None):
        super().__init__(legacy_callbacks_dict=legacy_callbacks_dict)

        # This will be updated with the user provided value from the config file.
        self.fire_pos_cfg: Dict[str, Any] = None
        self.op_locs_cfg: Dict[str, Any] = None

        # For each operational location's UID, this will store each sampled fire
        # position - the value will be the number of times it has been sampled
        # (ie. total episodes trained w/ the op loc + fire start pos pair).
        self.fire_pos_counter: Dict[str, Dict[str, int]] = {}
        # This will store the output directory used to save counter data in JSON format.
        self.fire_pos_counter_save_dir: str = None

        # This will store each operational location's UID (see BurnMDOperationalLocation)
        # and the value will be a dict with "train" and "eval" keys. The value for each
        # key will be the number of envs that have been seeded with this loc.
        self.op_locs_counter: Dict[str, Dict[str, int]] = {}
        # This will store the output directory used to save counter data in JSON format.
        self.op_locs_counter_save_dir: str = None

        # This will be used to access the correct index of the 2D fire pos array.
        self.op_loc_to_fire_array_idx: Dict[str, Dict[str, int]] = {
            "train": None,
            "eval": None,
        }
        # This will be used to get data arrays from the distributed object store.
        self.fire_init_pos_array_object_refs: Dict[str, ObjectRef] = {
            "train": None,
            "eval": None,
        }

        # PRNGs for each random process in the callback.
        self.prng_dict: Dict[str, "Generator"] = {
            "fire_initial_position": None,
            "operational_locations": None,
        }

        # Define the structure used to store the operational locations data.
        self.op_locations: Dict[
            str,
            Union[
                Dict[str, List["BurnMDOperationalLocation"]],
                List["BurnMDOperationalLocation"],
            ],
        ] = {
            "train": {
                "population": None,
                "current": None,
            },
            "eval": None,
        }

        # Define map to store the current
        self._env_id_to_env_fire_context: Dict[
            str, Dict[Tuple[int, int], "EnvFireContext"]
        ] = {
            "train": {},
            "eval": {},
        }

        # Helper attributes to track local worker existence and env count per worker.
        self._has_local_worker: Dict[str, bool] = {"train": None, "eval": None}
        self._envs_per_worker: Dict[str, int] = {"train": None, "eval": None}

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
          training fires will be updated after every `fire_init_pos_resample_interval`
          episodes.

        NOTE: This method is called at the end of `Algorithm.setup()`, after all the
        initialization is done, and before training actually starts.

        Arguments:
            algorithm: Reference to the Algorithm instance.
            kwargs: Forward compatibility placeholder.
        """
        # Validate the configuration and set the necessary variables.
        logdir = algorithm.logdir
        env_config = algorithm.config.env_config
        self.validate_and_set_config(env_config)
        self._check_sample_size_vs_workers(algorithm)
        self.prepare_context_from_algorithm(algorithm)
        train_workers = algorithm.workers
        eval_workers = algorithm.evaluation_workers

        # Perform setup behavior wrt operational locations.
        self._prepare_operational_locations(logdir)
        self._set_operational_location_foreach_env(train_workers, env_type="train")
        self._set_operational_location_foreach_env(eval_workers, env_type="eval")

        # Perform setup behavior wrt fire initial positions.
        train_data, eval_data = self._prepare_fire_initial_positions(env_config, logdir)
        self._set_fire_initial_position_foreach_env(train_data, train_workers, "train")
        self._set_fire_initial_position_foreach_env(eval_data, eval_workers, "eval")

        # Store each sub env's simfire config, in case we need to restart the env.
        self._store_fire_context_foreach_env(algorithm.workers, "train")
        self._store_fire_context_foreach_env(algorithm.evaluation_workers, "eval")

        # Put data into the distributed object store, and store the respective refs.
        self.fire_init_pos_array_object_refs["train"] = ray.put(train_data)
        self.fire_init_pos_array_object_refs["eval"] = ray.put(eval_data)

        # Prepare save paths for the counter data, then save the initial state.
        self._prepare_save_paths(logdir)
        self._save_counter_data()

    def on_train_result(
        self,
        *,
        algorithm: "Algorithm",
        result: dict,
        **kwargs,
    ) -> None:
        """Updates initial state of each `ReactiveHarness.sim` object across rollouts.

        Approach:
        - Check whether `fire_init_pos_resample_interval` episodes have past, and if so, get a (new)
          sample of size `sample_size` from the train dataset. Then, distribute chosen
          scenarios across the training rollouts.

        Arguments:
            algorithm: Current Algorithm instance.
            result: Dict of results returned from Algorithm.train() call.
                You can mutate this object to add additional metrics.
            kwargs: Forward compatibility placeholder.
        """
        # FIXME: Need to comb through logic and make sure everything is sound.
        curr_iter = algorithm.iteration
        logger.debug(f"Current algorithm iteration: {curr_iter}")

        # Handle resampling of operational locations.
        resample_fire_pos = False
        new_fire_scenario = False
        if self.locations_resample_interval == -1:
            logger.debug(
                "The `locations_resample_interval` is set to -1, so the current train "
                "scenarios will be used for the entire training process."
            )
        elif curr_iter % self.locations_resample_interval == 0:
            logger.info(
                f"Re-sampling operational locations after training iter: {curr_iter}"
            )
            self._set_operational_location_foreach_env(
                algorithm.workers, env_type="train"
            )
            # If locations are resampled, force resampling of fire initial positions.
            resample_fire_pos = True
            new_fire_scenario = True

        # Handle resampling of fire initial positions.
        if self.fire_init_pos_resample_interval == -1:
            logger.debug(
                "The `fire_init_pos_resample_interval` is set to -1, so the current train scenarios "
                "will be used for the entire training process."
            )
        elif resample_fire_pos or curr_iter % self.fire_init_pos_resample_interval == 0:
            logger.info(
                f"Re-sampling each fire init pos after training iter: {curr_iter}"
            )
            train_data = ray.get(self.fire_init_pos_array_object_refs["train"])
            self._set_fire_initial_position_foreach_env(
                train_data, algorithm.workers, "train"
            )
            # Put data back into the distributed object store and store the ref.
            self.fire_init_pos_array_object_refs["train"] = ray.put(train_data)
            new_fire_scenario = True

        if new_fire_scenario:
            logger.info("Storing the updated fire context for each sub environment...")
            self._store_fire_context_foreach_env(algorithm.workers, "train")
            logger.info("Saving the updated counter data...")
            self._save_counter_data()

    def on_workers_recreated(
        self,
        *,
        algorithm: "Algorithm",
        worker_set: "WorkerSet",
        worker_ids: List[int],
        is_evaluation: bool,
        **kwargs,
    ) -> None:
        """Callback run after one or more workers have been recreated.

        This method is called when one or more workers in the worker set have been
        recreated. It allows for custom logic to be executed on the recreated workers,
        such as setting properties or reinitializing environments.

        Method logic specific to InitializeSimfire callback:
        1. Determine the env type based on whether it is an evaluation worker set.
        2. Set the `rllib_env_context` for each environment within the worker set.
        3. Update the sub-environment context to indicate that the worker has been
           recreated. This is done because of an apparent bug in RLlib code.
        4. Reinitialize the simulation for the recreated environments using the
           configuration snapshot.

        Arguments:
            algorithm: Reference to the Algorithm instance.
            worker_set: The WorkerSet object in which the workers in question reside.
                You can use the
                `worker_set.foreach_worker(remote_worker_ids=..., local_worker=False)`
                method call to execute custom code on the recreated (remote) workers.
                Note that the local worker is never recreated as a failure of this would
                also crash the Algorithm.
            worker_ids: The list of (remote) worker IDs that have been recreated.
            is_evaluation: Whether `worker_set` is the evaluation WorkerSet (located in
                `Algorithm.evaluation_workers`) or not.
            **kwargs: Additional keyword arguments.
        """
        env_type = "eval" if is_evaluation else "train"
        # Set `rllib_env_context` for each env (needed w/in `env._set_fire_initial_position`).
        self._set_rllib_context_foreach_env(
            worker_set, self._has_local_worker[env_type], worker_ids=worker_ids
        )
        # Update the sub env context; it seems rllib has a bug where the recreated workers
        # have the boolean as false, although they've been recreated.
        worker_set.foreach_worker(
            lambda w: w.foreach_env(
                lambda env: setattr(env.rllib_env_context, "recreated_worker", True)
            ),
            local_worker=False,
            remote_worker_ids=worker_ids,
        )

        # Use config "snapshot" to reinitialize simulation for the recreated (sub) envs.
        worker_set.foreach_worker(
            lambda w: w.foreach_env(
                lambda env: env._configure_fire_for_recreated_worker(
                    self._env_id_to_env_fire_context[env_type]
                )
            ),
            local_worker=self._has_local_worker[env_type],
            remote_worker_ids=worker_ids,
        )

    def prepare_context_from_algorithm(self, algorithm: "Algorithm") -> None:
        """Configures the `rllib_env_context` for each train (eval) env.

        Arguments:
            algorithm: Reference to the Algorithm instance.
        """
        # Determine if local worker is present for training (evaluation) WorkerSet.
        self._initialize_worker_flags(algorithm)

        # Set `rllib_env_context` for each env (needed w/in `env._set_fire_initial_position`).
        self._set_rllib_context_foreach_env(
            algorithm.workers, self._has_local_worker["train"]
        )
        self._set_rllib_context_foreach_env(
            algorithm.evaluation_workers, self._has_local_worker["eval"]
        )

    def validate_and_set_config(self, env_config: Dict[str, Any]) -> None:
        # Validate the configuration for the `FireSimulation` object.
        self._validate_fire_simulation_config(env_config)

        # Set the operational locations config.
        self.op_locs_cfg = InitializeSimfire._prepare_operational_locations_config(
            env_config
        )

        self.fire_pos_cfg = InitializeSimfire._prepare_fire_initial_position_config(
            env_config
        )
        self._initialize_prngs()

    def _prepare_save_paths(self, logdir: str = None) -> None:
        if self.op_locs_cfg.get("save_op_locs_counter"):
            # Prepare the output directory.
            if self.op_locs_cfg.get("save_subdir"):
                outdir = os.path.join(logdir, self.op_locs_cfg["save_subdir"])
            else:
                outdir = os.path.join(logdir, "initialize_simfire")
            os.makedirs(outdir, exist_ok=True)
            logger.info(f"Operational locations counter will be saved to: {outdir}")
            self.op_locs_counter_save_dir = outdir

        if self.fire_pos_cfg["sampler"].get("save_fire_pos_counter"):
            # Prepare the output directory.
            if self.fire_pos_cfg["sampler"].get("save_subdir"):
                outdir = os.path.join(logdir, self.fire_pos_cfg["sampler"]["save_subdir"])
            else:
                outdir = os.path.join(logdir, "initialize_simfire")
            os.makedirs(outdir, exist_ok=True)
            logger.info(f"Fire initial positions counter will be saved to: {outdir}")
            self.fire_pos_counter_save_dir = outdir

    def _save_counter_data(self) -> None:
        # Write current state of op_locs_counter to a file.
        if self.op_locs_cfg.get("save_op_locs_counter"):
            fpath = os.path.join(self.op_locs_counter_save_dir, "op_locs_counter.json")
            logger.info("Dumping current state of `self.op_locs_counter` to JSON file.")
            with open(fpath, "w", encoding="utf-8") as f:
                json.dump(self.op_locs_counter, f, indent=4)

        # Write current state of fire_pos_counter to a file.
        if self.fire_pos_cfg["sampler"].get("save_fire_pos_counter"):
            fpath = os.path.join(self.fire_pos_counter_save_dir, "fire_pos_counter.json")
            logger.info("Dumping current state of `self.fire_pos_counter` to JSON file.")
            with open(fpath, "w", encoding="utf-8") as f:
                json.dump(self.fire_pos_counter, f, indent=4)

    def _prepare_operational_locations(self, logdir: str = None) -> None:
        """Prepare the operational locations 'dataset' for training and evaluation."""
        # TODO: Add check to ensure each location is valid. For more info, see:
        # https://github.com/mitrefireline/simfire/blob/0d46451db183a58d209ef789c509f00eca0daedf/simfire/utils/config.py#L306
        # FIXME: Document the usage of a hard-coded year.
        train_locs, eval_locs = env_utils.get_operational_locations(
            cfg=self.op_locs_cfg,
            num_train_locs=self.num_train_locations_in_population,
            num_eval_locs=self.num_eval_locations,
            prng=self.prng_dict["operational_locations"],
            fire_year=2020,
        )
        # Store the prepared operational locations for later use.
        self.op_locations["train"]["population"] = train_locs
        self.op_locations["eval"] = eval_locs

        # Optionally save the operational locations to disk.
        if self.op_locs_cfg.get("save_json_data") and logdir is not None:
            # Prepare the output directory.
            if self.op_locs_cfg.get("save_subdir"):
                outdir = os.path.join(logdir, self.op_locs_cfg["save_subdir"])
            else:
                outdir = os.path.join(logdir, "initialize_simfire")
            os.makedirs(outdir, exist_ok=True)
            locs_fpath = os.path.join(outdir, "operational_locations.json")

            # Map the locations to a human-readable format and save to disk.
            population_train_locs_hr = self._get_operational_location_dict("train")
            eval_locs_hr = self._get_operational_location_dict("eval")
            all_locs_hr = {"train": population_train_locs_hr, "eval": eval_locs_hr}

            with open(locs_fpath, "w", encoding="utf-8") as f:
                json.dump(all_locs_hr, f, indent=4)

            logger.info(f"Saved operational locations to: {locs_fpath}")

    def _get_operational_location_dict(
        self, env_type: Literal["train", "eval"]
    ) -> List[Dict]:
        """Get a dict representation of the operational locations for the env_type."""
        locations = self.op_locations[env_type]
        if env_type == "train":
            locations = locations["population"]

        return [
            {"uid": loc.uid, "latitude": loc.latitude, "longitude": loc.longitude}
            for loc in locations
        ]

    def _set_operational_location_foreach_env(
        self,
        worker_set: "WorkerSet",
        env_type: Literal["train", "eval"],
        worker_ids: Optional[List[int]] = None,
        locs_to_use: Optional[List["BurnMDOperationalLocation"]] = None,
    ) -> None:
        # Prepare inputs to use based on the respective env type.
        if env_type == "train":
            num_fire_pos_in_sample = self.num_train_fire_init_pos_in_sample
        else:
            num_fire_pos_in_sample = self.num_eval_fire_init_pos

        # Sample locations from the population, if not provided.
        if locs_to_use is None:
            locs_to_use = self._sample_locations_from_population(env_type)

        # Duplicate the locations to use for each fire position in the sample.
        # This is crucial to ensure (relatively) even distribution across envs!
        duplicated_locs = InitializeSimfire._duplicate_items(
            locs_to_use, num_fire_pos_in_sample
        )

        # Set the operational location for each sub environment, then update counters.
        logger.info(f"Setting operational locations for {env_type} environments...")
        locs_used = worker_set.foreach_worker(
            lambda w: w.foreach_env(
                lambda env: env._set_operational_location(
                    locations=duplicated_locs,
                    num_envs_per_worker=self._envs_per_worker[env_type],
                )
            ),
            local_worker=self._has_local_worker[env_type],
            remote_worker_ids=worker_ids,
        )

        self._update_op_locs_counter(locs_used, env_type)

    def _sample_locations_from_population(
        self, env_type: Literal["train", "eval"]
    ) -> List["BurnMDOperationalLocation"]:
        # Eval locations are fixed, so just return them. No need to sample!
        if env_type == "eval":
            logger.info("Using fixed evaluation locations...")
            return self.op_locations["eval"]

        # Sample from the training population.
        num_locs = self.num_train_locations_in_sample
        logger.info(
            f"Sampling {num_locs} operational locations from the training population..."
        )
        prng = self.prng_dict["operational_locations"]
        locs_to_use = self.op_locations["train"]["population"]
        sampled_locs = prng.choice(locs_to_use, num_locs).tolist()
        self.op_locations["train"]["current"] = sampled_locs
        return sampled_locs

    def _prepare_fire_initial_positions(
        self,
        env_config: Dict[str, Any],
        logdir: str = None,
    ) -> Dict[np.ndarray, np.ndarray]:
        # Get the op locations for train (eval), and define the ndarrays to return.
        population_train_locs = self.op_locations["train"]["population"]
        eval_locs = self.op_locations["eval"]
        train_arrays = {}
        eval_arrays = {}

        # Prepare the fire initial position data for each operational location.
        for loc in population_train_locs + eval_locs:
            logger.info(f"Preparing data for location: {loc}")
            train_data, eval_data = env_utils.prepare_fire_map_data(
                env_config["sim_init_cfg"],
                self.fire_pos_cfg,
                location=loc,
                return_train_data=loc in population_train_locs,
                return_eval_data=loc in eval_locs,
            )
            if train_data is not None:
                train_arrays[loc.uid] = train_data
            if eval_data is not None:
                eval_arrays[loc.uid] = eval_data

        # Stack the arrays and store the loc to idx mapping.
        train_array, train_loc_to_idx = self._stack_arrays(train_arrays)
        self.op_loc_to_fire_array_idx["train"] = train_loc_to_idx
        eval_array, eval_loc_to_idx = self._stack_arrays(eval_arrays)
        self.op_loc_to_fire_array_idx["eval"] = eval_loc_to_idx

        # Optionally save the raw data arrays to disk.
        outdir = ""
        if self.fire_pos_cfg["sampler"].get("save_raw_data") and logdir is not None:
            # Prepare the output directory.
            if self.fire_pos_cfg["sampler"].get("save_subdir"):
                outdir = os.path.join(logdir, self.fire_pos_cfg["sampler"]["save_subdir"])
            else:
                outdir = os.path.join(logdir, "initialize_simfire")
            os.makedirs(outdir, exist_ok=True)

            raw_train_fpath = os.path.join(outdir, "train_arr.npy")
            raw_eval_fpath = os.path.join(outdir, "eval_arr.npy")
            np.save(raw_train_fpath, train_data)
            logger.info(f"Saved train data array to: {raw_train_fpath}")
            np.save(raw_eval_fpath, eval_data)
            logger.info(f"Saved eval data array to: {raw_eval_fpath}")

        # Optionally save the fire initial position data in a human-readable format.
        if self.fire_pos_cfg["sampler"].get("save_json_data") and logdir is not None:
            # Prepare and save fire initial position data.
            train_fire_data_hr = self._get_fire_initial_position_dict(train_arrays)
            eval_fire_data_hr = self._get_fire_initial_position_dict(eval_arrays)

            all_fire_data_hr = {"train": train_fire_data_hr, "eval": eval_fire_data_hr}
            fire_data_fpath = os.path.join(outdir, "fire_initial_positions.json")
            with open(fire_data_fpath, "w", encoding="utf-8") as f:
                json.dump(all_fire_data_hr, f, indent=4)

        return train_array, eval_array

    def _get_fire_initial_position_dict(
        self, data_arrays: Dict[str, np.ndarray]
    ) -> Dict[str, Tuple[int, int]]:
        """Get a dict representation of the fire initial positions for the env_type."""
        fire_data = {}
        for loc_uid, loc_data in data_arrays.items():
            fire_data[loc_uid] = [f"({int(pos.x)}, {int(pos.y)})" for pos in loc_data]

        return fire_data

    def _set_fire_initial_position_foreach_env(
        self,
        data: np.ndarray,
        worker_set: "WorkerSet",
        env_type: Literal["train", "eval"],
    ) -> None:
        # Prepare inputs to use based on the respective env type.
        if env_type == "train":
            num_fire_pos_in_sample = self.num_train_fire_init_pos_in_sample
        else:
            num_fire_pos_in_sample = self.num_eval_fire_init_pos

        # Generate new indices randomly, w/o replacement, then create the array subset.
        pos_prng = self.prng_dict["fire_initial_position"]
        indices = pos_prng.choice(
            data.shape[-1],
            size=num_fire_pos_in_sample,
            replace=False,
        )
        data_subset = data[:, indices]

        # Set the fire initial position for each sub environment, then update counters.
        fire_pos_used = worker_set.foreach_worker(
            lambda w: w.foreach_env(
                lambda env: env._set_fire_initial_position(
                    data=data_subset,
                    num_envs_per_worker=self._envs_per_worker[env_type],
                    loc_to_idx=self.op_loc_to_fire_array_idx[env_type],
                )
            ),
            local_worker=self._has_local_worker[env_type],
        )

        # FIXME: Design of fire_pos_counter dict doesn't distinguish b/w train and eval.
        # For now, only update counter for training envs.
        if env_type == "train":
            self._update_fire_pos_counter(fire_pos_used, env_type)

    def _store_fire_context_foreach_env(
        self, worker_set: "WorkerSet", env_type: Literal["train", "eval"]
    ) -> None:
        """Store a snapshot of the fire config used for each sub environment.

        By subset, we mean the config options that are relevant to this callback, ie.
        `operational` settings and `fire` settings. In the future, there may be other
        settings that are relevant to the callback.

        Arguments:
            worker_set: The WorkerSet obj containing the sub environments of interest.
            env_type: The type of environment, either "train" or "eval".
        """
        env_configs = worker_set.foreach_worker(
            lambda w: w.foreach_env(lambda env: env._build_fire_context()),
            local_worker=self._has_local_worker[env_type],
        )

        # Update mapping of env to current fire init pos. Useful for restarts.
        start_val = 0 if self._has_local_worker[env_type] else 1
        for w_idx, w_cfgs in enumerate(env_configs, start=start_val):
            for v_idx, cfg in enumerate(w_cfgs):
                self._env_id_to_env_fire_context[env_type][(w_idx, v_idx)] = cfg

    def _update_op_locs_counter(
        self,
        locs_used: List[List["BurnMDOperationalLocation"]],
        env_type: Literal["train", "eval"],
    ) -> None:
        """Update self.op_locs_counter based on the locations set in the envs.

        Arguments:
            locs_used: Contains the operational location used by each train (eval) sub
                environment.
            env_type: The type of environment, either "train" or "eval".
        """
        logger.info(
            f"Updating operational locations counter for {env_type} environments..."
        )
        other_env_type = "eval" if env_type == "train" else "train"
        for loc in chain(*locs_used):
            if self.op_locs_counter.get(loc):
                self.op_locs_counter[loc][env_type] += 1
            else:
                self.op_locs_counter[loc] = {env_type: 1, other_env_type: 0}

    def _update_fire_pos_counter(
        self,
        fire_pos_used: List[Tuple[str, Tuple[int, int]]],
        env_type: Literal["train", "eval"],
    ) -> None:
        """Update self.fire_pos_counter based on the fire positions set in the envs.

        Arguments:
            fire_pos_used: Contains the fire positions used by each train (eval) sub
                environment.
            env_type: The type of environment, either "train" or "eval".
        """
        logger.info(
            f"Updating fire initial positions counter for {env_type} environments..."
        )
        # TODO: Optimize this to scale when sample size is large.
        for loc_uid, pos in chain(*fire_pos_used):
            # Ensure key is a str - JSON serialization requires this!
            pos = str(pos)
            # Update the counter, assuming the loc UID has been added.
            if self.fire_pos_counter.get(loc_uid):
                if self.fire_pos_counter[loc_uid].get(pos):
                    self.fire_pos_counter[loc_uid][pos] += 1
                else:
                    self.fire_pos_counter[loc_uid][pos] = 1
            else:
                self.fire_pos_counter[loc_uid] = {pos: 1}

        logger.debug(f"self.fire_pos_counter: \n{pformat(self.fire_pos_counter)}")

    def _initialize_prngs(self) -> None:
        """Initialize the PRNGs for each random process in the callback.

        Returns:
            A dictionary with the PRNGs for each random process.
        """
        self.prng_dict["operational_locations"] = np.random.default_rng(
            self.op_locs_cfg["seed"]
        )

        self.prng_dict["fire_initial_position"] = np.random.default_rng(
            self.fire_pos_cfg["sampler"]["seed"]
        )

    def _initialize_worker_flags(self, algorithm: "Algorithm"):
        self._has_local_worker["train"] = utils.has_local_worker(algorithm.config)
        self._has_local_worker["eval"] = utils.has_local_worker(
            algorithm.evaluation_config
        )

    def _set_rllib_context_foreach_env(
        self, worker_set: "WorkerSet", local_worker: bool, worker_ids: List[int] = None
    ):
        worker_set.foreach_worker(
            lambda w: w.foreach_env_with_context(env_utils.set_harness_env_context),
            local_worker=local_worker,
            remote_worker_ids=worker_ids,
        )

    @staticmethod
    def _stack_arrays(arrays: Dict[str, np.ndarray]) -> Tuple[np.ndarray, Dict[str, int]]:
        """Stack provided arrays into single array along the first axis.

        Arguments:
            arrays: A dictionary of arrays, where the keys are unique identifiers (uids)
                and the values are numpy ndarrays.

        Returns:
            A tuple containing the stacked array and a dictionary that maps
                uids to their corresponding indices in the stacked array.
        """
        data_list = []
        uid_to_idx = {}
        for uid, arr in arrays.items():
            data_list.append(arr)
            uid_to_idx[uid] = len(data_list) - 1

        return np.stack(data_list, axis=0), uid_to_idx

    @staticmethod
    def _prepare_operational_locations_config(
        env_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        # NOTE: We are not doing any validation of the provided op_locs config.
        try:
            op_locs_cfg = env_config["operational_locations"]
        except KeyError as e:
            logger.error(f"Missing key in environment configuration: {e}")
            raise ValueError("Invalid environment configuration") from e

        op_locs_cfg = InitializeSimfire._verify_seed(op_locs_cfg)
        return op_locs_cfg

    @staticmethod
    def _prepare_fire_initial_position_config(
        env_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        try:
            fire_pos_cfg = env_config["fire_initial_position"]
            sim_config_dict = env_config["sim_init_cfg"]["config_dict"]
            sim_map_size = simfire_utils.get_fire_map_size(sim_config_dict)
            # Validate the provided fire initial position config.
            fire_pos_cfg = env_utils.validate_fire_init_config(fire_pos_cfg, sim_map_size)
        except KeyError as e:
            logger.error(f"Missing key in environment configuration: {e}")
            raise ValueError("Invalid environment configuration") from e

        fire_pos_cfg["sampler"] = InitializeSimfire._verify_seed(fire_pos_cfg["sampler"])
        return fire_pos_cfg

    @staticmethod
    def _verify_seed(input: Dict[str, Any]) -> Dict[str, Any]:
        """Verify that a seed is provided in the input, and set a default if not.

        Arguments:
            input: The dictionary containing the configuration options.

        Returns:
            The updated configuration dictionary, guranteed to have a value for 'seed'.
        """
        if "seed" not in input.keys():
            default_seed = utils.get_default_seed()
            input["seed"] = default_seed
            logger.warning(
                "Seed not provided in the configuration. "
                f"Using the default seed: {default_seed}"
            )
        else:
            if input["seed"] is None:
                logger.warning(
                    "The provided seed is None (null). "
                    "The result is non-deterministic sampling!"
                )

        return input

    @staticmethod
    def _validate_fire_simulation_config(env_config: Dict[str, Any]) -> None:
        try:
            sim_config_dict = env_config["sim_init_cfg"]["config_dict"]
        except KeyError as e:
            logger.error(f"Missing key in environment configuration: {e}")
            raise ValueError("Invalid environment configuration") from e

        try:
            logger.debug("Validating `Terrain` config options...")
            env_utils.check_terrain_is_operational(sim_config=sim_config_dict)
            logger.debug("Validating `Fire Initial Position` config options...")
            env_utils.check_fire_init_pos_is_static(sim_config=sim_config_dict)
        except ValueError as e:
            logger.error(f"Configuration validation error: {e}")
            raise

    @staticmethod
    def _duplicate_items(items: List, num_copies: int) -> List:
        """Duplicate each item in the list `num_copies` times."""
        duplicated_items = []
        for item in items:
            duplicated_items.extend([item] * num_copies)

        return duplicated_items

    @property
    def fire_init_pos_resample_interval(self) -> int:
        """The number of training iters between resampling fire initial positions."""
        return self.fire_pos_cfg.get("sampler").get("resample_interval")

    @property
    def num_train_fire_init_pos_in_population(self) -> int:
        """Number of fire initial positions to sample for each training location."""
        return self.fire_pos_cfg.get("sampler").get("population_size")

    @property
    def num_train_fire_init_pos_in_sample(self) -> int:
        """Number of fire initial positions to sample for each training location."""
        return self.fire_pos_cfg.get("sampler").get("sample_size").get("train")

    @property
    def num_eval_fire_init_pos(self) -> int:
        """Number of fire initial positions to sample for each evaluation location."""
        return self.fire_pos_cfg.get("sampler").get("sample_size").get("eval")

    @property
    def locations_resample_interval(self) -> int:
        """The number of training iters between resampling operational locations."""
        return self.op_locs_cfg.get("resample_interval")

    @property
    def num_train_locations_in_population(self) -> int:
        """The total number of operational locations in the population."""
        if self.op_locations["train"]["population"] is not None:
            return len(self.op_locations["train"]["population"])
        else:
            return self.op_locs_cfg.get("population_size")

    @property
    def num_train_locations_in_sample(self) -> int:
        """The number of operational locations to sample for training envs."""
        return self.op_locs_cfg.get("sample_size").get("train")

    @property
    def num_eval_locations(self) -> int:
        """The number of operational locations to use for evaluation envs."""
        return self.op_locs_cfg.get("sample_size").get("eval")

    @property
    def total_train_scenarios(self) -> int:
        """The total number of fire scenarios to use for each training iteration."""
        fire_pos_per_loc = self.num_train_fire_init_pos_in_sample
        locs_per_sample = self.num_train_locations_in_sample
        return fire_pos_per_loc * locs_per_sample

    @property
    def total_eval_scenarios(self) -> int:
        """The total number of fire scenarios to use for each evaluation iteration."""
        return self.num_eval_fire_init_pos * self.num_eval_locations

    def _check_sample_size_vs_workers(self, algorithm: "Algorithm") -> None:
        """Ensure the sample size is valid wrt the number of workers/envs.

        NOTE: Currently, this method only checks the sample size wrt the number of
        expected total workers for training and evaluation. All workers are assumed to
        be healthy. Next iteration should leverage `WorkerSet.num_healthy_workers()`.
        """
        # Get the total number of training and evaluation envs.
        train_env_count = utils.get_total_envs(algorithm.workers)
        train_envs = train_env_count.total_envs
        self._envs_per_worker["train"] = train_env_count.envs_per_worker

        eval_env_count = utils.get_total_envs(algorithm.evaluation_workers)
        eval_envs = eval_env_count.total_envs
        self._envs_per_worker["eval"] = eval_env_count.envs_per_worker

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
