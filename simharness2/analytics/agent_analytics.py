"""TODO: A brief description of the module and its purpose.

TODO: Add a list of any classes, exception, functions, and any other objects exported by
the module.
"""
import logging
import os
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import InitVar, dataclass
from typing import List, Dict, Any

import numpy as np
import pandas as pd

from simharness2.agents import ReactiveAgent
from simfire.enums import BurnStatus
from simfire.sim.simulation import FireSimulation

logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
handler.setFormatter(
    logging.Formatter("%(asctime)s\t%(levelname)s %(filename)s:%(lineno)s -- %(message)s")
)
logger.addHandler(handler)
logger.propagate = False


@dataclass
class AgentData:
    """Docstring.

    FIXME: Alt. names - `AgentBehavior`, `AgentEpisodeBehavior`, etc. ??
    """

    agent_id: str
    save_history: InitVar[bool] = False

    def __post_init__(self, save_history):
        """TODO"""
        # Create a deque that is (optionally) used to aggregate data across timesteps.
        if save_history:
            self._history = deque()
        else:
            self._history = None

    def update(self, timestep_dict):
        """
        Regardless of whether we save history, we want to store:
        - movement
        - interaction
        - moved_off_map
        - connected_mitigation (TODO later)
        - near_fire
        - burn_status
        - timestep
        """
        # Store the current timestep's data in the history deque.
        if self._history is not None:
            self._history.append(timestep_dict)

        # Update the attributes that store the agent's behavior.
        self.movement = timestep_dict["movement"]
        self.interaction = timestep_dict["interaction"]
        self.moved_off_map = timestep_dict["moved_off_map"]
        # self.connected_mitigation = timestep_dict["connected_mitigation"]
        self.near_fire = timestep_dict["near_fire"]
        self.burn_status = timestep_dict["burn_status"]

    def save_episode_history(self, output_dir: str, total_eval_iters: int) -> None:
        """Save episode history to CSV file."""
        if self._history is None:
            return

        # TODO: Add logic to save history from multiple episodes (run concurrently).
        # Maybe we can use the PID to create a unique file name for each episode?
        # Prepare to save
        subdir = os.path.join("agent_data", self.agent_id)

        # TODO: Update logic to handle saving history from training episodes too.
        data_save_path = os.path.join(
            output_dir, subdir, f"eval_iter_{total_eval_iters}.csv"
        )
        # Converts deque to list of dicts, then to DataFrame.
        df = pd.DataFrame(list(self._history))
        # Write to CSV file.
        logger.info(f"Saving episode history to {data_save_path}...")
        os.makedirs(os.path.dirname(data_save_path), exist_ok=True)
        df.to_csv(data_save_path, index=False)


class AgentAnalytics(ABC):
    """Interface used to monitor the behavior of an agent within the simulation.

    Attributes:
        sim: TODO
        movement_types: TODO
        interaction_types: TODO

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
        self._movement_types = movement_types
        self._interaction_types = interaction_types

        # Store the danger level (in tiles) for the agent.
        self._danger_level = danger_level

    @abstractmethod
    def reset(self, env_is_rendering: bool = False):
        """Reset the attributes of `AgentAnalytics` to initial values."""
        pass

    @abstractmethod
    def reset_after_one_simulation_step(self) -> None:
        """Reset values that are tracked between each simulation step."""
        pass

    @abstractmethod
    def update(
        self,
        timestep: int,
    ) -> None:
        """Update the `AgentAnalytics` object after each agent action."""
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

        logging.debug(f"near_agent_arr:\n {near_agent_arr}")
        # Check if any tiles surrounding the agent have value `BurnStatus.BURNING`.
        return np.any(near_agent_arr == BurnStatus.BURNING).astype(bool)

    def connected_mitigation(self, fire_map: np.ndarray, agent_pos: List[int]) -> bool:
        """TODO docstring."""
        pass


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

    # NOTE: This is called from within the init of FireSimulationAnalytics !!
    def __init__(
        self,
        *,
        sim: FireSimulation,
        movement_types: List[str],
        interaction_types: List[str],
        agent_ids: set,
        danger_level: int = 2,
        save_history: bool = False,
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
            agent_ids: TODO
            danger_level: An integer indicating the minimum distance (in tiles) between
                the agent and a tile with value `BurnStatus.BURNING` for the agent to be
                considered "in danger" (maybe a better name is `near_fire_threshold`?).
            save_history: TODO
        """
        super().__init__(
            sim=sim,
            movement_types=movement_types,
            interaction_types=interaction_types,
            danger_level=danger_level,
        )

        # Indicates if data from each timestep will be stored across the entire episode.
        self.save_history = save_history
        self._agent_ids = agent_ids
        # TODO: Cleaner to store all agents within a single `AgentData` class?
        self.data = {ag_id: AgentData(ag_id, save_history) for ag_id in self._agent_ids}

    def update(
        self,
        timestep: int,
        agents: Dict[Any, ReactiveAgent],
    ) -> None:
        """Update the AgentAnalytics object variables after each agent action.

        Args:
            timestep: An integer indicating the current timestep of the episode.
            agents: TODO
        """
        # FIXME: Update for MARL case.
        # NOTE: These are stored in the corresponding FireSimulationAnalytics.data obj.
        # if self._interaction_types[interaction] != "none":
        #     self.num_interactions_since_last_sim_step += 1
        # if self._movement_types[movement] != "none":
        #     self.num_movements_since_last_sim_step += 1
        for ag_id, agent in agents.items():
            agent_pos = (agent.row, agent.col)
            # Prepare current timestep data.
            agent_timestep_dict = {
                "timestep": timestep,
                "movement": self._movement_types[agent.latest_movement],
                "interaction": self._interaction_types[agent.latest_interaction],
                "moved_off_map": agent.moved_off_map,
                # "connected_mitigation": self.connected_mitigation(fire_map, agent_pos),
                "near_fire": self.agent_near_fire(self.sim.fire_map, agent_pos),
                "burn_status": BurnStatus(self.sim.fire_map[agent_pos]).name,
            }
            # Store agent behavior for the current timestep in the dataclass
            self.data[ag_id].update(agent_timestep_dict)

    def reset_after_one_simulation_step(self) -> None:
        """Reset values that are tracked between each simulation step."""
        # For debugging, and potentially, timestep intermediate reward calculation?
        # self.num_interactions_since_last_sim_step = 0
        # self.num_movements_since_last_sim_step = 0
        pass

    def reset(self, env_is_rendering: bool = False):
        """Reset the attributes of `BaseAgentAnalytics` to initial values.

        Note that this is intended to be called within `FireSimulationAnalytics.reset()`.
        """
        # Reset attributes used to store the agent's behavior across a single episode.
        # NOTE: either create new object or use dataclasses.replace()
        save_history = env_is_rendering and self.save_history
        self.data = {ag_id: AgentData(ag_id, save_history) for ag_id in self._agent_ids}
        # Reset attributes used to store the agent's behavior between each sim step.
        self.reset_after_one_simulation_step()
