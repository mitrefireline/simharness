from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING
import time
import logging
import json
import random
import numpy as np
from pprint import pformat
from dataclasses import dataclass, field

from simfire.utils.config import Config
from simfire.sim.simulation import Simulation
from simharness2.utils import fire_data
from simharness2.utils.utils import get_default_seed

if TYPE_CHECKING:
    from ray.rllib.env.env_context import EnvContext
    from simfire.sim.simulation import FireSimulation
    from simfire.utils.config import OperationalConfig
    from simharness2.environments.fire_harness import FireHarness
    from numpy.random import Generator


logger = logging.getLogger(__name__)
logger.propagate = False


# TODO: Add this constant to a more relevant place; fine to keep here for now.
SCREEN_SIZE_TO_OPERATIONAL_HW = {
    64: 1920,
    128: 3840,
    256: 7680,
    512: 15360,
    1024: 30720,
}


@dataclass(frozen=True)
class BurnMDOperationalLocation:
    """Dataclass to store the operational location of a BurnMD fire scenario."""

    uid: str
    state: str = field(repr=False)
    year: int = field(repr=False)
    fire_name: str = field(repr=False)
    latitude: float
    longitude: float

    @property
    def lat_lon(self) -> Tuple[float, float]:
        """Return the latitude and longitude of the operational location."""
        return self.latitude, self.longitude


@dataclass
class EnvFireContext:
    """Dataclass to store the context of the environment's fire config."""

    # Environment identification fields
    worker_index: int
    vector_index: int

    # Fire environment context fields
    fire_initial_position: Tuple[int, int]
    burnmd_operational_location: Optional[BurnMDOperationalLocation] = None
    operational_config: Optional["OperationalConfig"] = None

    @property
    def operational_year_used(self) -> int:
        """Return the year of the operational data used for the current environment.

        This is for convenience, but also to highlight that the operational year used
        may differ from the year of the BurnMD fire scenario data. The reason is that,
        by default, the operational data year is set to:
        - the year prior to the BurnMD fire scenario year, ie. year - 1.
        """
        return self.operational_config.year

    @property
    def env_uid(self) -> Tuple[int, int]:
        """Return the unique identifier for the current environment.

        The unique identifier is a tuple of the worker and vector indices. Technically,
        this uid is not unique across train and eval envs, but it will be unique within
        the context of the env type (ie. the truth of self.in_evaluation).
        """
        return self.worker_index, self.vector_index


@dataclass
class RLlibEnvContextMetadata:
    worker_index: int
    vector_index: int
    remote: bool
    num_workers: int
    recreated_worker: bool


def get_adjacent_points(
    row: int, col: int, shape: Tuple[int, int], include_diagonals: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """Return points adjacent to the provided point (excluding point itself).

    The current implementation considers the 4 cardinal directions (N, S, E, W) as
    adjacent points. If `include_diagonals` is set to True, the diagonal points
    (NE, NW, SE, SW) are also considered as adjacent points.

    Arguments:
        row: The row index of the current point.
        col: The column index of the current point.
        shape: A 2-tuple representing the shape of the map.
        include_diagonals: A boolean indicating whether to include diagonal points
            as adjacent points. Defaults to True.

    Returns:
        A tuple containing two numpy arrays, adjacent rows and adjacent columns. The
        returned arrays can be used as an advanced index to access the adjacent
        points, ex: `fire_map[adj_rows, adj_cols]`.
    """
    # TODO: Logic below is copied from a method in simfire, namely
    # simfire.game.managers.fire.FireManager._get_new_locs(). It would be good to
    # refactor this logic into a utility function in simfire, and then call it here.
    x, y = col, row
    # Generate all possible adjacent points around the current point.
    if include_diagonals:
        new_locs = (
            (x + 1, y),
            (x + 1, y + 1),
            (x, y + 1),
            (x - 1, y + 1),
            (x - 1, y),
            (x - 1, y - 1),
            (x, y - 1),
            (x + 1, y - 1),
        )
    else:
        new_locs = (
            (x + 1, y),
            (x, y + 1),
            (x - 1, y),
            (x, y - 1),
        )

    col_coords, row_coords = zip(*new_locs)
    adj_array = np.array([row_coords, col_coords], dtype=np.int32)

    # Clip the adjacent points to ensure they are within the map boundaries
    row_max, col_max = [dim - 1 for dim in shape]
    adj_array = np.clip(adj_array, a_min=[[0], [0]], a_max=[[row_max], [col_max]])
    # Remove the point itself from the list of adjacent points, if it exists.
    adj_array = adj_array[:, ~np.all(adj_array == [[row], [col]], axis=0)]

    return adj_array[0], adj_array[1]


def set_harness_env_context(harness: "FireHarness", env_context: "EnvContext"):
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


def get_operational_locations(
    cfg: Dict[str, Any],
    num_eval_locs: int,
    num_train_locs: int = None,
    prng: "Generator" = None,
    fire_year: int = None,
) -> Tuple[List[BurnMDOperationalLocation], List[BurnMDOperationalLocation]]:
    """Sample operational locations from the BurnMD dataset for training and evaluation.

    This function samples a specified number of operational locations from the BurnMD
    dataset, which can be filtered by a specific fire year. The function supports
    sampling with or without ensuring that evaluation locations are independent
    (mutually exclusive) from training locations.

    Notes:
        - If `independent_eval` is set to `True` in the `cfg` dictionary, the function
          ensures that the training and evaluation locations are mutually exclusive.
          Otherwise, evaluation locations may overlap with training locations.
        - The behavior when `num_train_locs` is None is to use all available locations
          for training after reserving the specified number for evaluation.
        - Specifying a `fire_year` filters the dataset to only include locations from
          that year.
        - The function logs the sampled train and eval locations for transparency.

    TODO (Test Cases to implement):
        - num_eval_locs > num_train_locs with independent_eval == False should raise a
          ValueError.
        - Correct behavior when num_train_locs is None (use all locations for training).
        - Correct behavior when fire_year is specified, and when it is None.
        - Specifying a seed should result in reproducible sampling.
        - independent_eval == False should result in eval_locs being a subset of
          train_locs.

    Parameters:
        cfg: A dictionary containing configuration parameters, including the path to the
            BurnMD dataset.
        num_eval_locs: The number of locations to sample for evaluation.
        num_train_locs: The number of locations to sample for training. If None, all
            remaining locations after sampling for evaluation are used for training.
        seed: An optional seed for the random number generator to ensure reproducibility.
        fire_year: An optional year to filter the operational locations by fire year.

    Returns:
        A tuple containing two lists:
            - The first contains `BurnMDOperationalLocation` objects for training.
            - The second contains `BurnMDOperationalLocation` objects for evaluation.

    Raises:
        ValueError: If the number of evaluation locations requested exceeds the number of
            training locations when independent evaluation is not enabled. Also raised if
            the total number of locations to sample exceeds the number of available
            locations in the BurnMD dataset.

    """
    if prng is None:
        prng = np.random.default_rng(get_default_seed())

    # Load BurnMD data to use for sampling random operational locations.
    burnmd_fp = cfg.get("burnmd_dataset_path")
    logger.info(f"Loading BurnMD data from {burnmd_fp}")
    with open(burnmd_fp, "r", encoding="utf-8") as j:
        burnmd_op_locs = json.loads(j.read())

    # Downsample BurnMD by year, if specified
    if fire_year is not None:
        logger.info(f"Filtering BurnMD data for year {fire_year}...")
        burnmd_op_locs = {
            uid: loc_data
            for uid, loc_data in burnmd_op_locs.items()
            if loc_data["year"] == fire_year
        }
        logger.info(f"Number of locations for year {fire_year}: {len(burnmd_op_locs)}")

    # Get the total number of locations to sample
    independent_eval_locs = cfg.get("independent_eval", True)
    # We need enough locations for both training and eval.
    if num_train_locs is None and independent_eval_locs:
        num_train_locs = len(burnmd_op_locs) - num_eval_locs
    # We only need enough locations for training (since eval is a subset).
    elif num_train_locs is None and not independent_eval_locs:
        num_train_locs = len(burnmd_op_locs)

    if independent_eval_locs:
        total_locations = num_train_locs + num_eval_locs
    else:
        # Address FIXME in docstr; for now, only evaluate on locations we train on
        # and instead, let different fire initial positions create data diversity.
        if num_eval_locs > num_train_locs:
            raise ValueError(
                "The number of evaluation locations cannot exceed the number of "
                "training locations when `independent_eval` is False. This "
                "ensures that the evaluation locations are a subset of the training "
                "locations."
            )

        total_locations = num_train_locs

    # Validate the number of locations to sample
    if total_locations > len(burnmd_op_locs):
        raise ValueError(
            f"Total number of locations to sample ({total_locations}) exceeds the "
            f"number of available locations in BurnMD ({len(burnmd_op_locs)})."
        )

    # Randomly sample keys from the dictionary
    logger.info(f"Sampling {total_locations} operational locations from BurnMD...")
    sampled_keys = prng.choice(list(burnmd_op_locs.keys()), total_locations).tolist()

    # Split the sampled keys into train and eval sets
    if independent_eval_locs:
        train_keys = sampled_keys[:num_train_locs]
        eval_keys = sampled_keys[num_train_locs:]
    else:
        train_keys = sampled_keys
        # Set eval keys to be randomly sampled from train keys.
        eval_keys = random.sample(train_keys, num_eval_locs)

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


def set_operational_location(
    sim: "FireSimulation", location: BurnMDOperationalLocation
) -> "FireSimulation":
    """Set the operational location for the provided FireSimulation object, sim."""
    if location.year != 2020:
        logger.warning(
            f"Location {location.uid} is from year {location.year}, but currently we "
            "only support using BurnMD data from wildfires in the year 2020. This "
            "requirement ensures that we can load operational data collected in a year "
            "PRIOR to the BurnMD fire year. The hope is that using data from a previous "
            "year will provide a more realistic operational environment. This will be "
            "addressed in a future MR."
        )

    # TODO: Create MR for simfire to add `set_operational_location` method and
    # optimize/update the logic of `reset_terrain()`.
    # For now, we just recreate the SimFire Config object and reset the simulation.
    # Overwrite the operational settings in the SimFire config.
    sim_cfg = sim.config.yaml_data

    # FIXME: Logic for setting "height" and "width" is convoluted, need better approach.
    # Use yaml_data bc simfire _load_area() overwrites this value w/ op data.
    screen_height, screen_width = sim.config.yaml_data["area"]["screen_size"]
    op_height = SCREEN_SIZE_TO_OPERATIONAL_HW[screen_height]
    op_width = SCREEN_SIZE_TO_OPERATIONAL_HW[screen_width]
    sim_cfg["operational"].update(
        {
            "latitude": location.latitude,
            "longitude": location.longitude,
            # NOTE: Forcing year to be the year prior to the BurnMD data year, ie. 2019.
            "year": str(location.year - 1),
            # Setting H and W to "correct" operational values to prevent user error.
            "height": op_height,
            "width": op_width,
        }
    )
    logger.info(
        f"Updated SimFire operational settings:\n{pformat(sim_cfg['operational'])}"
    )

    # Overwrite the Config object, then reset to ensure the changes take effect.
    logger.info("Recreating the SimFire Config object with updated operational settings.")
    updated_config = Config(config_dict=sim_cfg)
    del sim.config
    sim.config = updated_config

    sim.reset()

    # Check that the operational location was set correctly.
    sim_lat_lon = sim.config.operational.latitude, sim.config.operational.longitude
    if sim_lat_lon != location.lat_lon:
        msg = (
            f"Error setting operational location for {location.uid}. Expected "
            f"latitude, longitude: {location.lat_lon}, but got: {sim_lat_lon}."
        )
        raise ValueError(msg)

    return sim


def prepare_fire_map_data(
    sim_init_cfg: Dict[str, Any],
    fire_pos_cfg: Dict[str, Any],
    location: BurnMDOperationalLocation,
    return_train_data: bool = True,
    return_eval_data: bool = True,
    logdir: str = None,
) -> Tuple[np.recarray, np.recarray]:
    """Prepare the fire map data for the environment.

    The values of return_train_data and return_eval_data should be determined by
    the usage of the provided location, ie. is it used for training, evaluation, or
    both. For example, if the location is only used for training, then
    return_eval_data should be set to False.

    Arguments:
        sim_init_cfg: The configuration used to initialize the FireSimulation object.
            This dictionary should contain 2 keys: `simfire_cls` and `config_dict`.
        fire_pos_cfg: The configuration for the fire initial position data.
        location: The operational location to use for the simulation.
        return_train_data: Whether to return the training data.
        return_eval_data: Whether to return the evaluation data.
        logdir: The directory to save the data to.
    """
    generator_cfg = fire_pos_cfg.get("generator")
    sampler_cfg = fire_pos_cfg.get("sampler")

    # Use provided simulation info to create a simulation object.
    sim = create_fire_simulation_from_config(sim_init_cfg)
    # FIXME: Ideally we would pass the sim_init_cfg into set_operational_location, but
    # just build the object here for now. It's redundant, but avoids further refactoring
    # until we have experimentation results for the benchmark paper.
    # Set the operational location for the simulation.
    sim = set_operational_location(sim, location)
    try:
        # Generate the dataset using the provided configuration for `generator`.
        start_time = time.time()
        fire_df = fire_data.generate_fire_initial_position_data(sim, **generator_cfg)
        end_time = time.time()
        total_runtime = end_time - start_time
        logger.debug(f"Total generator runtime: {total_runtime} seconds.")
        logger.debug(f"Total generator runtime: {total_runtime/60:.2f} minutes")

        # Down sample the dataset using the provided configuration for `sampler`.
        train_data, eval_data = fire_data.filter_fire_initial_position_data(
            fire_df=fire_df,
            logdir=logdir,
            return_train_data=return_train_data,
            return_eval_data=return_eval_data,
            **sampler_cfg,
        )
    except ValueError as e:
        logger.error(f"Error generating fire map data for {location.uid}: {e}")
        train_data, eval_data = None, None

    return train_data, eval_data


def create_fire_simulation_from_config(sim_init_cfg: Dict[str, Any]) -> "FireSimulation":
    """Create a FireSimulation object from the provided configuration."""
    if sim_init_cfg is None:
        logger.warning("The simulation configuration is not provided. Returning None.")
        return
    sim_cls = sim_init_cfg.get("simfire_cls")
    if sim_cls is None:
        raise ValueError(
            "The simulation class must be present in the `sim` "
            "dictionary. This is usually specified via the "
            "`simfire_cls` key in `environment.env_config.sim`."
        )
    elif not issubclass(sim_cls, Simulation):
        raise ValueError("The simulation class must be a subclass of `Simulation`.")
    sim_cfg = sim_init_cfg.get("config_dict")
    return sim_cls(Config(config_dict=sim_cfg))


def check_fire_init_pos_is_static(sim_config: Dict[str, Any]) -> None:
    """Ensure the `fire.fire_initial_position.type` is static."""
    fire_init_pos_type = sim_config["fire"]["fire_initial_position"]["type"]
    if fire_init_pos_type != "static":
        msg = (
            "Invalid value for `fire.fire_initial_position.type`: "
            f"{fire_init_pos_type}. The value must be `static`."
        )
        raise ValueError(msg)


def check_terrain_is_operational(sim_config: Dict[str, Any]) -> None:
    """Ensure `topography.type` and `fuel.type` are operational for `terrain`."""
    terrain_cfg = sim_config["terrain"]
    fuel_type = terrain_cfg["fuel"]["type"]
    topo_type = terrain_cfg["topography"]["type"]
    layer_types = [fuel_type, topo_type]
    if not all(l_type == "operational" for l_type in layer_types):
        msg = (
            "Invalid value for `terrain.topography.type` or `terrain.fuel.type`: "
            f"{layer_types}. The values must BOTH be `operational`."
        )
        raise ValueError(msg)


def validate_fire_init_config(
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
