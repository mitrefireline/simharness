from typing import Any, Dict, Set, Tuple, TYPE_CHECKING
import time
import logging
import numpy as np

import simharness2.utils.fire_data as fire_data
from simharness2.environments.harness import RLlibEnvContextMetadata
from simharness2.environments.fire_harness import BurnMDOperationalLocation

if TYPE_CHECKING:
    from ray.rllib.env.env_context import EnvContext
    from simfire.sim.simulation import FireSimulation
    from simharness2.environments.fire_harness import FireHarness


logger = logging.getLogger(__name__)


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


def prepare_fire_map_data(
    sim: "FireSimulation",
    fire_pos_cfg: Dict[str, Any],
    train_locations: Set[BurnMDOperationalLocation],
    eval_locations: Set[BurnMDOperationalLocation],
    logdir str = None,
) -> Tuple[np.recarray, np.recarray]:
    """Prepare the fire map data for the environment."""
    generator_cfg = fire_pos_cfg.get("generator")
    sampler_cfg = fire_pos_cfg.get("sampler")

    # Prepare fire position data for each location in the list of locations.
    for location in locations:
        # Set the location for the simulation.
        # TODO: Create MR for simfire to add `set_operational_location` method and
        # optimize/update the logic of `reset_terrain()`.
        # FIXME: We have access to the "year" of the fire, but are not using it here.
        sim.config.reset_terrain(location=location.lat_lon)

        # Generate the dataset using the provided configuration for `generator`.
        start_time = time.time()
        fire_df = fire_data.generate_fire_initial_position_data(sim, **generator_cfg)
        end_time = time.time()
        total_runtime = end_time - start_time
        logger.debug(f"Total generator runtime: {total_runtime} seconds.")
        logger.debug(f"Total generator runtime: {total_runtime/60:.2f} minutes")

        # Down sample the dataset using the provided configuration for `sampler`.
        # FIXME: One idea is to always return train_data, eval_data, but if the location
        # is only used for training, then eval_data is None or an empty recarray.
        train_data, eval_data = fire_data.filter_fire_initial_position_data(
            fire_df=fire_df, logdir=logdir, **sampler_cfg
        )


def check_fire_init_pos_is_static(sim: "FireSimulation") -> None:
    """Ensure the `fire.fire_initial_position.type` is static."""
    fire_init_pos_type = sim.config.yaml_data["fire"]["fire_initial_position"]["type"]
    if fire_init_pos_type != "static":
        msg = (
            "Invalid value for `fire.fire_initial_position.type`: "
            f"{fire_init_pos_type}. The value must be `static`."
        )
        raise ValueError(msg)


def check_terrain_is_operational(sim: "FireSimulation") -> None:
    """Ensure `topography.type` and `fuel.type` are operational for `terrain`."""
    layer_types = sim.get_layer_types()
    if not all(l_type == "operational" for l_type in layer_types.values()):
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
