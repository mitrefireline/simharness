"""TODO: A brief description of the module and its purpose.

TODO: Add a list of any classes, exception, functions, and any other objects exported by
the module.
"""
import math
from abc import ABC, abstractmethod
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype
from simfire.enums import BurnStatus
from simfire.sim.simulation import FireSimulation
from simharness2.analytics.utils import reset_df


class AgentAnalytics(ABC):
    """Interface used to monitor the behavior of an agent within the simulation.

    Attributes:
        sim: TODO
        movement_types: TODO
        interaction_types: TODO
        df: TODO
        df_cols: TODO
        df_dtypes: TODO
        df_index: TODO

    TODO: Add section for anything related to the interface for subclassers.
    """

    def __init__(
        self,
        *,
        sim: FireSimulation,
        movement_types: List[str],
        interaction_types: List[str],
        danger_level: int = 2,
    ):
        """TODO: A brief description of what the method is and what it's used for.

        TODO: Add any side effects that occur when executing the method.
        TODO: Add any exceptions that are raised.
        TODO: Add any restrictions on when the method can be called.

        Arguments:
            sim: A `FireSimulation` object. FIXME!
            movement_types: A list of strings indicating the available movements for the
                agent.
            interaction_types: A list of strings indicating the available interactions
                for the agent.
            danger_level: An integer indicating the minimum distance (in tiles) between
                the agent and a tile with value `BurnStatus.BURNING` for the agent to be
                considered "in danger" (maybe a better name is `near_fire_threshold`?).
        """
        # Store a reference to the agent's world, the `FireSimulation` object.
        self.sim = sim

        # Store the movement and interaction types that are available to the agent.
        self.movement_types = movement_types
        self.interaction_types = interaction_types

        # Store the danger level (in tiles) for the agent.
        self._danger_level = danger_level

        # Define stubs for the class attributes.
        self.df: pd.DataFrame = None
        self.df_cols: List[str]
        self.df_dtypes: Dict[str, Any]
        self.df_index: str

        self.prepare_df_metadata()

        # NOTE: `self.df` will be initialized in `self._reset_df()`.
        self.reset()

    @abstractmethod
    def reset_after_one_simulation_step(self) -> None:
        """Reset values that are tracked between each simulation step."""
        pass

    @abstractmethod
    def update(
        self, timestep: int, movement: int, interaction: int, agent_pos: List[int]
    ) -> None:
        """Update the AgentMetricsTracker object variables after each agent action."""
        pass

    @abstractmethod
    def prepare_df_metadata(self):
        """Define the metadata (column names, dtypes, etc.) used for `self.df`."""
        pass

    def agent_near_fire(self, fire_map: np.ndarray, agent_pos: List[int]) -> bool:
        """Check if the agent is adjacent to a space that is currently burning.

        Returns:
            near_fire: A boolean indicating if there is a burning space adjacent to the
            agent.
        """
        # Define the range of rows and columns to check for burning spaces.
        row_start = max(0, agent_pos[0] - self._danger_level)
        row_end = min(fire_map.shape[0], agent_pos[0] + self._danger_level + 1)
        col_start = max(0, agent_pos[1] - self._danger_level)
        col_end = min(fire_map.shape[1], agent_pos[1] + self._danger_level + 1)
        near_agent_arr = fire_map[row_start:row_end, col_start:col_end]

        # Check if any tiles surrounding the agent have value `BurnStatus.BURNING`.
        return np.any(near_agent_arr == BurnStatus.BURNING)

    def reset(self):
        """Reset the attributes of `BaseAgentAnalytics` to initial values.

        Note that this is intended to be called within `FireSimulationAnalytics.reset()`.
        """
        # Reset attributes used to store the agent's behavior across a single episode.
        self.df = reset_df(self.df, self.df_cols, self.df_dtypes, self.df_index)
        # Reset attributes used to store the agent's behavior between each sim step.
        self.reset_after_one_simulation_step()


class ReactiveAgentAnalytics(AgentAnalytics):
    """Use `ReactiveAgentAnalytics` to monitor an agent in a `reactive` fire scenario.

    Attributes:
        sim: TODO
        movement_types: TODO
        interaction_types: TODO
        df: TODO
        df_cols: TODO
        df_dtypes: TODO
        df_index: TODO

    TODO: Add section for anything related to the interface for subclassers.
    """

    def __init__(
        self,
        *,
        sim: FireSimulation,
        movement_types: List[str],
        interaction_types: List[str],
        danger_level: int = 2,
    ):
        """TODO: A brief description of what the method is and what it's used for.

        TODO: Add any side effects that occur when executing the method.
        TODO: Add any exceptions that are raised.
        TODO: Add any restrictions on when the method can be called.

        NOTE: `self.num_interatctions_since_last_sim_step` and
        `self.num_movements_since_last_sim_step` are initialized within the call to
        `self.reset_after_one_simulation_step()`, which is called within `self.reset()`.

        Arguments:
            sim: A `FireSimulation` object. FIXME!
            movement_types: A list of strings indicating the available movements for the
                agent.
            interaction_types: A list of strings indicating the available interactions
                for the agent.
            danger_level: An integer indicating the minimum distance (in tiles) between
                the agent and a tile with value `BurnStatus.BURNING` for the agent to be
                considered "in danger" (maybe a better name is `near_fire_threshold`?).
        """
        super().__init__(
            sim=sim,
            movement_types=movement_types,
            interaction_types=interaction_types,
            danger_level=danger_level,
        )

    def prepare_df_metadata(self):
        """Prepares the metadata (column names, dtypes, etc.) for the agent dataframe.

        Within this method, the default names and dtypes for the columns of the agent
        dataframe are defined and stored in `self.df_cols` and
        `self.df_dtypes`, respectively. These values are used to initialize the
        `self.df` dataframe within the `reset_df()` method.

        Columns used to store the agent's behavior:
            - `timestep`: current timestep in the episode.
            - `movement`: string id for the movement that the agent selected.
            - `interaction`: string id for the interaction that the agent selected.
            - `near_fire`: bool indicating if the agent is near the fire.
            - `burn_status`: name (str) of the BurnStatus value at the agent's current
                position.
            - `x_pos`: int indicating the x-value of the agent's position within the sim.
            - `y_pos`: int indicating the y-value of the agent's position within the sim.
        """
        # Define the columns that will be used to store the agent's behavior.
        self.df_cols = [
            "timestep",
            "movement",
            "interaction",
            "near_fire",
            "burn_status",
            "x_pos",
            "y_pos",
        ]

        movement_types = CategoricalDtype(categories=self.movement_types)
        interaction_types = CategoricalDtype(categories=self.interaction_types)
        status_types = CategoricalDtype(categories=[s.name for s in BurnStatus])
        self.df_dtypes = {
            "timestep": np.uint16,
            "movement": movement_types,
            "interaction": interaction_types,
            "near_fire": "boolean",
            "burn_status": status_types,
            "x_pos": np.uint8,
            "y_pos": np.uint8,
        }

        self.df_index = "timestep"

    def update(
        self,
        timestep: int,
        movement: int,
        interaction: int,
        agent_pos: List[int],
    ) -> None:
        """Update the AgentAnalytics object variables after each agent action.

        Arguments:
            timestep: The current timestep in the episode.

        """
        # NOTE: These are stored in the corresponding `FireSimulationData.df`.
        if self.interaction_types[interaction] != "none":
            self.num_interactions_since_last_sim_step += 1
        if self.movement_types[movement] != "none":
            self.num_movements_since_last_sim_step += 1

        fire_map, agent_pos = self.sim.fire_map, agent_pos

        # Add the current timestep's data to the dataframe.
        # TODO: Is there a better alternative to this df build approach?
        agent_data = [
            [timestep],
            [self.movement_types[movement]],
            [self.interaction_types[interaction]],
            [self.agent_near_fire(fire_map, agent_pos)],
            [BurnStatus(fire_map[agent_pos[0], agent_pos[1]]).name],
            [agent_pos[1]],
            [agent_pos[0]],
        ]
        agent_data_dict = dict(zip(self.df_cols, agent_data))
        timestep_df = (
            pd.DataFrame(agent_data_dict).astype(self.df_dtypes).set_index(self.df_index)
        )
        self.df = pd.concat([self.df, timestep_df])

    def reset_after_one_simulation_step(self) -> None:
        """Reset values that are tracked between each simulation step."""
        # For debugging, and potentially, timestep intermediate reward calculation?
        self.num_interactions_since_last_sim_step = 0
        self.num_movements_since_last_sim_step = 0


class AgentMetricsTracker:
    """Monitors and tracks the behavior of a single agent within the simulation."""

    def __init__(self, sim: FireSimulation):
        """TODO: Docstring for __init__."""
        # Store a reference to the agent's world, the `FireSimulation` object.
        self._sim = sim
        self.reset()

        # number of actions that the agent has taken within this timestep
        self.num_agent_actions = 0

        # bool for if the agent has placed a mitigation within this timestep
        self.mitigation_placed = False

        # the number of mitigations that the agent has places within this timestep
        self.new_mitigations = 0

        # bool for if the agent is currently within the burning squares (is on fire)
        self.agent_is_burning = False

        # bool for if the agent is currently operating within area that is already burned
        self.agent_in_burned_area = False

        # bool for if the agent is nearby the active fire
        self.agent_near_fire = False

        # TODO create a tracker variable to store the latest action(s) taken

        # TODO create a tracker variable to store the list of actions taken with respect to the timesteps

        # ----------------------

    def update(
        self,
        mitigation_placed: bool,
        movements: List[str],
        movement: int,
        interaction: int,
        agent_pos: List[int],
        agent_pos_is_empty_space: bool,
    ) -> None:
        """Update the AgentMetricsTracker object variables after each agent action"""
        # NOTE: Attribute (s) useful for debugging; may be removed later.
        if mitigation_placed:
            self.num_interactions_since_last_sim_step += 1
            self.mitigation_placed = True
        if movements[movement] != "none":
            self.num_movements_since_last_sim_step += 1

        # Update interaction-specific attributes.
        self.agent_interactions.append(interaction)

        # Update movement-specific attributes.
        self.agent_movements.append(movement)
        self.agent_positions.append(agent_pos)

        fire_map, agent_pos = self._sim.fire_map, agent_pos
        self.agent_is_burning = self._agent_is_burning(fire_map, agent_pos)
        self.agent_in_burned_area = self._agent_in_burned_area(fire_map, agent_pos)
        self.agent_nearby_fire = self._agent_nearby_fire(fire_map, agent_pos)

        # NOTE: We may want to penalize the agent for moving into a non-empty space.
        self.agent_pos_is_empty_space = agent_pos_is_empty_space

    def reset_after_one_simulation_step(self) -> None:
        """Reset values that are tracked between each simulation step."""
        self.num_agent_actions = 0
        # For debugging, and potentially, timestep intermediate reward calculation.
        self.num_interactions_since_last_sim_step = 0
        self.num_movements_since_last_sim_step = 0

        # Movement-specific attributes that can be utilized during reward calculation.
        self.agent_is_burning = False
        self.agent_in_burned_area = False
        self.agent_near_fire = False
        self.agent_pos_is_empty_space = False

    def reset(self):
        """Reset the AgentMetricsTracker to initial values."""
        # reset the agent_datas previous reset func
        self.reset_after_one_simulation_step()

        # Attributes used to store the agent's behavior across a single episode.
        self.agent_interactions: List[int] = []
        self.agent_movements: List[int] = []
        self.agent_positions: List[List[int]] = []

    def _agent_nearby_fire(self, fire_map: np.ndarray, agent_pos: List[int]) -> bool:
        """Check if the agent is adjacent to a space that is currently burning.

        Returns:
            nearby_fire: A boolean indicating if there is a burning space adjacent to the
            agent.
        """
        nearby_locs = []
        screen_size = math.sqrt(fire_map.shape[0])
        # Get all spaces surrounding agent - here we are setting 2 as the danger level distance in squares
        for i in range(agent_pos[0] - 1, agent_pos[0] + 2):
            for j in range(agent_pos[1] - 1, agent_pos[1] + 2):
                if (
                    i < 0
                    or i >= screen_size
                    or j < 0
                    or j >= screen_size
                    or [i, j] == agent_pos
                ):
                    pass
                else:
                    nearby_locs.append((i, j))

        for i, j in nearby_locs:
            if fire_map[i][j] == BurnStatus.BURNING:
                return True
        return False

    def _agent_is_burning(self, fire_map: np.ndarray, agent_pos: List[int]) -> bool:
        # return true if the agent is in a burning square
        if (fire_map[agent_pos[1], agent_pos[0]]) == BurnStatus.BURNING:
            return True

        return False

    def _agent_in_burned_area(self, fire_map: np.ndarray, agent_pos: List[int]) -> bool:
        # return true if the agent is in a burning square
        if (fire_map[agent_pos[1], agent_pos[0]]) == BurnStatus.BURNED:
            return True

        return False
