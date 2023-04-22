from typing import List
from dataclasses import dataclass
from abc import ABC, abstractmethod
from numpy import ndarray


class SimulationMetricsTracker(ABC):
    """TODO Add class docstring."""

    @abstractmethod
    def update(self, sim_map: ndarray):
        """TODO Add docstring."""
        raise NotImplementedError


# @dataclass
class SimpleSimulationMetricsTracker(SimulationMetricsTracker):
    # Indicates whether the simulation has reached the end.
    active: bool = True
    # Number of times the simulation has been run (ie. `FireSimulation.run()`).
    # num_steps: int = 0
    # Number of spaces in `sim.fire_map` that have `BurnStatus.BURNING`.
    num_burning: int = 0
    # Array
    # num_burning_per_step: ndarray

    def update(self, sim_map: ndarray):
        pass


@dataclass
class SimulationMetricsTracker:
    active: bool # = True
    num_steps: int # = 0
    # Number of spaces in sim_map that are not equal to BurnStatus.UNBURNED
    num_damaged_per_step: ndarray
    # num new squares damaged (since last timestep)
    # num_damaged_diff: int
    # num "burned" squares
    num_burned: int
    num_new_burned: int
    # - num new "burned" squares (since last timestep)
    num_currently_burning: int
    # - num "currently burning" squares
    num_new_currently_burning: int
    # - num new "currently burning" squares (since last timestep)
    num_mitigation_lines: int
    # - num mitigation lines
    num_new_mitigation_lines: int
    # - num new mitigation lines (since last timestep)


@dataclass
class AgentMetricsTracker:
    num_steps: int # = 0
    # TODO decide how to initialize array
    # - Maybe an empty array with size == to max steps for a given episode?
    # - Otherwise, we will have to append to the array on each timestep. Costly??
    is_burning_per_step: ndarray
    in_burned_area_per_step: ndarray
    near_burned_area_per_step: ndarray

    def update(self, agent_pos: List[int], sim_map: ndarray, interaction: bool):
        self.num_steps += 1

        # TODO put num_mitigations and recent_mitigations in the SimMetricsTracker?
        if interaction:
            self.recent_mitigations += 1

        # TODO track this to add big negative reward
        self.agent_burning = False
        # TODO subtract from self.recent_mitigations if the mitigation was placed in a burned area
        self.agent_in_burned_area = False
        # TODO track this to add negative reward for Agent being too close to fire
        self.agent_near_burning_area = _nearby_fire(sim_map, agent_pos)

    def _nearby_fire(sim_map: ndarray, agent_pos: List[int]) -> bool:
        # TODO
        return False
