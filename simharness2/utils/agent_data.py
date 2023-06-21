from typing import List
import math
import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype

from simfire.sim.simulation import FireSimulation
from simfire.enums import BurnStatus


class AgentData:
    """Stores data relvant to the behavior of a single agent within the simulation."""

    def __init__(
        self,
        *,
        sim: FireSimulation,
        movement_types: List[str],
        interaction_types: List[str],
    ):
        """TODO: Docstring for __init__.

        Arguments:
            sim: A `FireSimulation` object. FIXME!
            movement_types: A list of strings indicating the available movements for the
                agent.
            interaction_types: A list of strings indicating the available interactions
                for the agent.
        """
        # Store a reference to the agent's world, the `FireSimulation` object.
        self._sim = sim

        # Store the movement and interaction types that are available to the agent.
        self.movement_types = movement_types
        self.interaction_types = interaction_types

        self.agent_df: pd.DataFrame = None  # FIXME use better name?
        # NOTE: `self.agent_df` will be initialized in `self._prepare_agent_df()`.
        self._prepare_df_metadata()

        self.reset()

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
        self.agent_df_cols = [
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
        self.agent_df_dtypes = {
            "timestep": np.uint16,
            "movement": movement_types,
            "interaction": interaction_types,
            "near_fire": "boolean",
            "burn_status": status_types,
            "x_pos": np.uint8,
            "y_pos": np.uint8,
        }

        self.agent_df_index = "timestep"

    def _reset_df(self):
        """Resets the agent dataframe (`self.agent_df`) to its initial state."""
        if self.agent_df is not None:
            tmp_df = self.agent_df.iloc[0:0]
            del self.agent_df
            self.agent_df = tmp_df
        else:
            self.agent_df = (
                pd.DataFrame(columns=self.agent_df_cols)
                .astype(self.agent_df_dtypes)
                .set_index(self.agent_df_index)
            )

    def update(
        self,
        timestep: int,
        movement: int,
        interaction: int,
        agent_pos: List[int],
    ) -> None:
        """Update the AgentMetricsTracker object variables after each agent action"""
        # NOTE: These are stored in the corresponding `FireSimulationData.sim_df`.
        if self.interaction_types[interaction] != "none":
            self.num_interactions_since_last_sim_step += 1
        if self.movement_types[movement] != "none":
            self.num_movements_since_last_sim_step += 1

        fire_map, agent_pos = self._sim.fire_map, agent_pos

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
        agent_data_dict = dict(zip(self.agent_df_cols, agent_data))
        timestep_df = (
            pd.DataFrame(agent_data_dict)
            .astype(self.agent_df_dtypes)
            .set_index(self.agent_df_index)
        )
        self.agent_df = pd.concat([self.agent_df, timestep_df])

    def reset_after_one_simulation_step(self) -> None:
        """Reset values that are tracked between each simulation step."""
        # For debugging, and potentially, timestep intermediate reward calculation?
        self.num_interactions_since_last_sim_step = 0
        self.num_movements_since_last_sim_step = 0

    def reset(self):
        """Reset the attributes of `AgentData` to initial values.

        Note that this is intended to be called within `FireSimulationData.reset()`.
        """
        # Reset attributes used to store the agent's behavior across a single episode.
        self._reset_df()
        # Reset attributes used to store the agent's behavior between each sim step.
        self.reset_after_one_simulation_step()

    def _agent_nearby_fire(self, fire_map: np.ndarray, agent_pos: List[int]) -> bool:
        """Check if the agent is adjacent to a space that is currently burning.

        Returns:
            nearby_fire: A boolean indicating if there is a burning space adjacent to the
            agent.
        """
        # FIXME: Debug this function to verify that it returns the correct boolean!!
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
