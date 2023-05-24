from typing import List, Tuple, Union, Any, Optional
import logging

import numpy as np
from omegaconf import DictConfig
from hydra.utils import instantiate

from simfire.utils.config import Config
from simfire.sim.simulation import FireSimulation


# class EvaluationFireScenarios:
#     """FIXME: Class docstring for EvaluationFireScenarios."""

#     def __init__(
#         self,
#         seed: int,
#         num_operational_locations: int = None,
#         operational_locations: List[Tuple[float, float]] = None,
#         num_fire_positions: int = None,
#         fire_initial_positions: Union[List[Tuple[int, int]], List[int]] = None,
#         num_agent_positions: int = None,
#         agent_initial_positions: List[Tuple[int, int]] = None,
#     ):
#         if operational_locations:
#             self.operational_locations = operational_locations
#         elif num_operational_locations:
#             self._generate_operational_locations(num_operational_locations, seed)
#         else:
#             raise ValueError(
#                 "Must specify either operational_locations or num_operational_locations"
#             )

#     def _generate_operational_locations(
#         self, num_operational_locations: int, seed: int
#     ) -> List[Tuple[float, float]]:
#         """FIXME: Docstring for _generate_operational_locations."""


# def prepare_operational_locations(
#     sim_config: Config,
#     operational_locations: List[Tuple[float, float]] = None,
#     operational_seeds: List[int] = None,
# ) -> List[Tuple[float, float]]:
#     """FIXME: Docstring for prepare_operational_locations."""
#     if not operational_locations and not operational_seeds:
#         raise ValueError("Must specify either operational_locations or operational_seeds")
#     elif operational_locations and operational_seeds:
#         raise ValueError(
#             "Cannot specify both operational_locations and operational_seeds"
#         )
#     elif operational_locations:
#         # If operational locations specified, verify that they return a valid simulation
#         for op_loc in operational_locations:
#             sim_config.operational.latitude = op_loc[0]
#             sim_config.operational.longitude = op_loc[1]
#             sim = FireSimulation(sim_config)


def get_default_operational_fires(
    cfg: DictConfig,
    seed: int = None,
) -> Tuple[Tuple[float, float], Tuple[int, int], Tuple[int, int]]:
    """FIXME: Docstring for get_default_operational_fires."""
    # FIXME hard-coded values for now
    num_locations = 6
    # NOTE: logic below assumes num_fire_positions == num_agent_positions
    num_fire_positions = 10
    num_agent_positions = 10
    # FIXME mine as well just use the seed from the config?
    seed = seed or cfg.debugging.seed
    screen_size = cfg.simulation.train.area.screen_size
    output_shape = (screen_size, screen_size)
    # total_fires = cfg.evaluation.get("evaluation_duration", 0)

    # FIXME update seeding procedure
    rng, seed = np_random(seed)
    op_location_seeds = rng.integers(low=1, high=1000, size=num_locations, dtype=int)

    sim = instantiate(cfg.environment.env_config.simulation)
    sim_yaml_data = sim.config.yaml_data
    del sim

    # Generate valid locations to use for operational fires
    op_locations: List[Tuple[float, float]] = []
    for op_seed in op_location_seeds:
        sim = _try_to_create_sim(sim_yaml_data, op_seed, rng)

        # Reset the sim to ensure new fire map is loaded
        # sim.reset()
        # Extract the operational location
        op_location = sim.config.lat_long_box.center

        # Make sure fuel and topography layers have correct shape
        fuel_shape = sim.terrain.fuels.shape
        topo_shape = sim.terrain.elevations.shape
        if fuel_shape == output_shape and topo_shape == output_shape:
            op_locations.append(op_location)

    # Generate valid positions for fire start
    fire_positions = [
        tuple(rng.integers(0, screen_size, size=2, dtype=int))
        for _ in range(num_fire_positions)
    ]
    # Generate valid positions for agent start
    agent_positions = [
        tuple(rng.integers(0, screen_size, size=2, dtype=int))
        for _ in range(num_agent_positions)
    ]

    # Create operational fire scenarios using the generated locations and positions
    op_fires = {}
    num_op_fires = 0
    for op_loc in op_locations:
        fire_positions = rng.permutation(fire_positions)
        agent_positions = rng.permutation(agent_positions)
        for idx in range(len(fire_positions)):
            op_fires[num_op_fires] = (
                op_loc,
                tuple(fire_positions[idx]),
                tuple(agent_positions[idx]),
            )
            num_op_fires += 1

    return op_fires


def np_random(seed: int) -> Tuple[np.random.Generator, Any]:
    """Generates a random number generator from the seed and returns the Generator and seed.

    Args:
        seed: The seed used to create the generator

    Returns:
        The generator and resulting seed

    Raises:
        Error: Seed must be a non-negative integer or omitted
    """
    seed_seq = np.random.SeedSequence(seed)
    np_seed = seed_seq.entropy
    rng = np.random.Generator(np.random.PCG64(seed_seq))
    return rng, np_seed


def _try_to_create_sim(
    sim_yaml_data: DictConfig, seed: int, rng: np.random.Generator
) -> FireSimulation:
    try:
        logging.getLogger(__name__).info(f"Trying to create sim with seed {seed}")
        sim_yaml_data["operational"].update({"seed": seed})
        sim = FireSimulation(Config(config_dict=sim_yaml_data))
        return sim
    except FileNotFoundError as e:
        logging.getLogger(__name__).error(e)
        new_seed = rng.integers(low=1, high=1000, size=1, dtype=int)[0]
        return _try_to_create_sim(sim_yaml_data, new_seed, rng)
