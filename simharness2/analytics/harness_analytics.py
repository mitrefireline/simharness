"""FIXME: A one line summary of the module or program.

Leave one blank line.  The rest of this docstring should contain an
overall description of the module or program.  Optionally, it may also
contain a brief description of exported classes and functions and/or usage
examples.

Typical usage example:

  foo = ClassFoo()
  bar = foo.FunctionBar()
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, List

import numpy as np
from simfire.enums import BurnStatus
from simfire.sim.simulation import Simulation

# NOTE on Terminology:
#  - "Run" == a single experiment with a single set of parameters
#  - "Episode" == a single episode within a run
#  - "Timestep" == a single timestep within an episode


# class BaseTracker(ABC):
# Other names: `AnalyticsData`, RLHarnessDataStore`, `RLHarnessData`?
class AnalyticsTracker(ABC):
    """TODO Add class docstring."""

    def __init__(
        self,
        # TODO add type anns to input arguments
        sim_data,
        sim_screen_size: int,
        run_data=None,
        agent_data=None,
        bench_sim_data=None,
    ):
        """TODO (afennelly): Add docstring.

        Expected usage:
            TODO

        Arguments:
            sim_data: TODO
            sim_screen_size: An int representing how large the simulation is in pixels.
              The screen_size sets both the height and the width of the screen.
            run_data: TODO
            agent_data: TODO
            bench_sim_data: TODO
        """
        # Required attributes that track simulation data across each episode in a run.
        self.sim_data = sim_data
        self.sim_screen_size = sim_screen_size
        # Optional attributes that track additional data for the RLHarness.
        self.run_data = run_data
        self.agent_data = agent_data
        self.bench_sim_data = bench_sim_data

    @abstractmethod
    def update_after_one_simulation_step(
        self,
        sim_map: np.ndarray,
        sim_active: bool,
        bench_sim_map: np.ndarray,
        bench_sim_active: bool,
        **kwargs,
    ):
        """TODO Add docstring."""
        raise NotImplementedError

    @abstractmethod
    def update_after_one_agent_step(
        self,
        agent_pos: List[int],
        sim_map: np.ndarray,
        **kwargs,
    ):
        """TODO Add docstring."""
        raise NotImplementedError


"""The status of each pixel in a `fire_map`

Current statuses are:
    - UNBURNED
    - BURNING
    - BURNED
    - FIRELINE
    - SCRATCHLINE
    - WETLINE

UNBURNED: int = 0
BURNING: int = auto()
BURNED: int = auto()
FIRELINE: int = auto()
SCRATCHLINE: int = auto()
WETLINE: int = auto()
"""


class SimpleAnalyticsTracker(AnalyticsTracker):
    """TODO Add class docstring."""

    def __init__(self, sim_data, sim_screen_size: int):
        """TODO (afennelly): Add docstring.

        Expected usage:
            TODO

        Arguments:
            sim_data: TODO
            sim_screen_size: An int representing how large the simulation is in pixels.
              The screen_size sets both the height and the width of the screen.
        """
        super().__init__(sim_data, sim_screen_size)

    def update_after_one_simulation_step(
        self, timestep: int, sim_map: np.ndarray, **kwargs
    ):
        # Assuming we use `SimpleReward`, what variables do we need to calculate the reward?
        # - num_burning_per_step
        self.sim_data.update(timestep, sim_map, self.sim_screen_size)


# class SimpleAnalyticsTrackerWithAgentData(SimpleAnalyticsTracker):
#     # previously named `AgentStep_FEAR_Update()`
#     # better name I think: `update_variables_tracked_after_one_agent_step()`
#     def update_after_one_agent_step(
#         self,
#         agent_pos: List[int],
#         sim_map: np.ndarray,
#         interaction: bool,
# #     ):
#         """Run after one "agent step" is taken within the simulation.

#         We consider one "agent step" to be when the agent performs one movement,
#         followed by one interaction with the environment. Recall that "don't move" and
#         "don't interact" are valid agent movements and interactions, respectively, so
#         the agent's position may not change after one "agent step".

#         NOTE: Currently, one "agent step" is taken for every call to the environment's
#         `step()` method.

#         Arguments: FIXME update docstring with new arguments
#             timestep: An integer representing the current timestep of the simulation.
#             mitigation_placed: A boolean indicating whether agent's interaction with the
#               environment resulted in a mitigation being placed.
#             nearby_fire: A boolean indicating whether the agent is in a square that is
#                 currently burning or has been burned in the past. This is used to determine
#                 whether the agent is in a "danger zone" and should be penalized for not.

#         """
#         self.agent_data.update(agent_pos, sim_map, interaction)
