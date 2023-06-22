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


class BaseAgentAnalytics(ABC):
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
        """
        # Store a reference to the agent's world, the `FireSimulation` object.
        self.sim = sim

        # Store the movement and interaction types that are available to the agent.
        self.movement_types = movement_types
        self.interaction_types = interaction_types

        # Define stubs for the class attributes.
        self.df: pd.DataFrame = None
        self.df_cols: List[str]
        self.df_dtypes: Dict[str, Any]
        self.df_index: str

        self._prepare_df_metadata()

        # NOTE: `self.df` will be initialized in `self._reset_df()`.
        self.reset()

    def reset(self):
        """Reset the attributes of `BaseAgentAnalytics` to initial values.

        Note that this is intended to be called within `FireSimulationAnalytics.reset()`.
        """
        # Reset attributes used to store the agent's behavior across a single episode.
        self._reset_df()
        # Reset attributes used to store the agent's behavior between each sim step.
        self.reset_after_one_simulation_step()

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
    def _prepare_df_metadata(self):
        """Define the metadata (column names, dtypes, etc.) used for `self.df`."""
        pass

    def _reset_df(self):
        """Resets the episode dataframe, `self.df`, to its initial state."""
        if self.df is not None:
            # FIXME convert to usage of df.iat, if possible
            self.df = self.df.iloc[0:0]
        else:
            self.df = (
                pd.DataFrame(columns=self.df_cols)
                .astype(self.df_dtypes)
                .set_index(self.df_index)
            )

    def _agent_nearby_fire(self, fire_map: np.ndarray, agent_pos: List[int]) -> bool:
        """Check if the agent is adjacent to a space that is currently burning.

        Returns:
            nearby_fire: A boolean indicating if there is a burning space adjacent to the
            agent.
        """
        # FIXME: Debug this function to verify that it returns the correct boolean!!
        nearby_locs = []
        screen_size = math.sqrt(fire_map.shape[0])
        # Get all spaces surrounding agent - here we are setting 2 as the danger level
        # distance in squares
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


class AgentAnalytics(BaseAgentAnalytics):  # noqa: D101
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
        """
        super().__init__(
            sim=sim,
            movement_types=movement_types,
            interaction_types=interaction_types,
        )

    def _prepare_df_metadata(self):
        """Prepares the metadata (column names, dtypes, etc.) for the agent dataframe.

        Within this method, the default names and dtypes for the columns of the agent
        dataframe are defined and stored in `self.agent_df_cols` and
        `self.agent_df_dtypes`, respectively. These values are used to initialize the
        `self.agent_df` dataframe within the `self._reset_df()` method.

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
        """Update the AgentMetricsTracker object variables after each agent action."""
        # NOTE: These are stored in the corresponding `FireSimulationData.sim_df`.
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
            [self._agent_nearby_fire(fire_map, agent_pos)],
            [BurnStatus(fire_map[agent_pos[1], agent_pos[0]]).name],
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
