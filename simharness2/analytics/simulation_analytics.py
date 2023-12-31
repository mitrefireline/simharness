"""TODO: A brief description of the module and its purpose.

TODO: Add a list of any classes, exception, functions, and any other objects exported by
the module.
"""
import logging
import os
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import InitVar, dataclass
from functools import partial
from typing import Optional

import numpy as np
import pandas as pd
from simfire.enums import BurnStatus
from simfire.sim.simulation import FireSimulation

from simharness2.analytics.agent_analytics import ReactiveAgentAnalytics

logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
handler.setFormatter(
    logging.Formatter("%(asctime)s\t%(levelname)s %(filename)s:%(lineno)s -- %(message)s")
)
logger.addHandler(handler)
logger.propagate = False


@dataclass
class SimulationData:
    """Docstring.

    FIXME: Alt. names - `SimulationBehavior`, `SimEpisodeBehavior`, etc. ??
    """

    is_benchmark: bool = False
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
        - burned_total
        - unburned_total
        - burning_total
        - mitigated_total (all mitigation types, can break up later if desired)
        - timestep and/or sim_step (not sure if we actually need this though?)
        Additionally, it might be useful to store (mostly for debug purposes??):
        - num_new_interactions (sum !"none" interactions since last call to sim.run(1))
        - num_new_movements (total non "none" movements since last call to sim.run(1))
        """
        # Store the current timestep's data in the history deque.
        if self._history is not None:
            self._history.append(timestep_dict)

        # Update the attributes that store the simulation's behavior.
        self.burned = timestep_dict["burned"]
        self.unburned = timestep_dict["unburned"]
        self.burning = timestep_dict["burning"]

        if not self.is_benchmark:
            self.mitigated = timestep_dict["mitigated"]
            # self.agent_interactions = timestep_dict["agent_interactions"]
            # self.agent_movements = timestep_dict["agent_movements"]

    def save_episode_history(self, output_dir: str, total_eval_iters: int) -> None:
        """Save episode history to CSV file."""
        if self._history is None:
            return

        # TODO: Add logic to save history from multiple episodes (run concurrently).
        # Maybe we can use the PID to create a unique file name for each episode?
        # Prepare to save
        if self.is_benchmark:
            subdir = "benchsim_data"
        else:
            subdir = "sim_data"

        # TODO: Update logic to handle saving history from training episodes too.
        sim_data_save_path = os.path.join(
            output_dir, subdir, f"eval_iter_{total_eval_iters}.csv"
        )
        # Converts deque to list of dicts, then to DataFrame.
        df = pd.DataFrame(list(self._history))
        # Write to CSV file.
        logger.info(f"Saving episode history to {sim_data_save_path}...")
        os.makedirs(os.path.dirname(sim_data_save_path), exist_ok=True)
        df.to_csv(sim_data_save_path, index=False)


class SimulationAnalytics(ABC):
    """Interface used to monitor metrics using the `fire_map` within a `FireSimulation`.

    Attributes:
        sim: TODO
        agent_analytics: TODO
        is_benchmark: TODO

    TODO: Add section for anything related to the interface for subclassers.
    """

    def __init__(
        self,
        sim: FireSimulation,
        agent_analytics_partial: partial,
        is_benchmark: bool = False,
    ):
        """TODO Add docstring.

        Arguments:
            sim: The `FireSimulation` object that will be tracked.
            agent_analytics_partial: A `functools.partial` object that defines the class
            that will be used to monitor and track agent (s) behavior within `self.sim`.
            is_benchmark: TODO

        NOTE: In the MARL case, we can use a dictionary of AgentAnalytics objects,
        where the key is the agent ID. This would change the type of `agent_analytics`.
        """
        # Store a reference to the `FireSimulation` object that is being tracked.
        self.sim = sim
        # Indicates whether this object will track a `benchmark` simulation.
        self.is_benchmark = is_benchmark
        self.agent_analytics: Optional[ReactiveAgentAnalytics] = None

        # TODO: Update for MARL case.
        if not self.is_benchmark:
            # Agents only exist in the main simulation.
            self.agent_analytics = agent_analytics_partial(sim=self.sim)

    @abstractmethod
    def reset(self, env_is_rendering: bool = False):
        """Reset the attributes of `FireSimulationData` to initial values."""
        pass

    @abstractmethod
    def update(self, timestep: int) -> None:
        """TODO Add docstring."""
        pass


class FireSimulationAnalytics(SimulationAnalytics):
    """Use `FireSimulationAnalytics` to monitor the `fire_map` within a `FireSimulation`.

    Attributes:
        sim: TODO
        is_benchmark: TODO
        agent_analytics: TODO
        num_agents: TODO
        df: TODO
        df_cols: TODO
        df_dtypes: TODO
        df_index: TODO
        num_sim_steps: TODO
        active: TODO

    TODO: Add section for anything related to the interface for subclassers.
    """

    def __init__(
        self,
        sim: FireSimulation,
        agent_analytics_partial: partial,
        is_benchmark: bool = False,
        save_history: bool = False,
        log_to_file: bool = False,
        file_type: str = "csv",
        custom_file_name: Optional[str] = None,
    ):
        """TODO: A brief description of what the method is and what it's used for.

        TODO: Add any side effects that occur when executing the method.
        TODO: Add any exceptions that are raised.
        TODO: Add any restrictions on when the method can be called.

        Arguments:
            sim: The `FireSimulation` object that will be tracked.
            agent_analytics_partial: A `functools.partial` object that defines the class
                that will be used to monitor and track agent (s) behavior within
                `self.sim`.
            is_benchmark: TODO
            save_data: TODO
        """
        super().__init__(sim, agent_analytics_partial, is_benchmark)

        self._is_benchmark = is_benchmark
        # Indicates if data from each timestep will be stored across the entire episode.
        self.save_history = save_history
        self.data = SimulationData(is_benchmark, save_history)

        # Helper attributes used to control the saving of data to a file.
        self.log_to_file = log_to_file
        self.file_type = file_type
        self.custom_file_name = custom_file_name

    def update(self, timestep: int) -> None:
        """TODO Add docstring."""
        # Only access `active` attribute if the sim has been updated at least once.
        if self.sim.elapsed_steps != 0:
            self.active = self.sim.active

        # Prepare current timestep data.
        fire_map = self.sim.fire_map
        burned_total = np.sum(fire_map == BurnStatus.BURNED)
        burning_total = np.sum(fire_map == BurnStatus.BURNING)
        unburned_total = np.sum(fire_map == BurnStatus.UNBURNED)

        sim_timestep_dict = {
            "sim_step": self.num_sim_steps,
            "timestep": timestep,
            "burned": burned_total,
            "burning": burning_total,
            "unburned": unburned_total,
        }

        if not self.is_benchmark:
            non_mitigated_total = burned_total + burning_total + unburned_total
            sim_timestep_dict.update(
                {
                    "mitigated": fire_map.size - non_mitigated_total,
                    # FIXME: Update for MARL case.
                    # "agent_interactions": self.agent_analytics.num_interactions_since_last_sim_step,  # noqa: E501
                    # "agent_movements": self.agent_analytics.num_movements_since_last_sim_step,  # noqa: E501
                }
            )

        # Update the dataclass that stores the simulation's behavior.
        self.data.update(sim_timestep_dict)

        self.num_sim_steps += 1  # increment AFTER method logic is performed (convention).

    def reset(self, env_is_rendering: bool = False):
        """Reset the attributes of `FireSimulationData` to initial values."""

        # NOTE: either create new object or use dataclasses.replace()
        save_history = env_is_rendering and self.save_history
        self.data = SimulationData(self._is_benchmark, save_history)

        # Reset attributes used to store simulation behavior across a single episode.
        self.num_sim_steps = 0
        self.active = True

        # If we are tracking agent behavior, reset the `agent_analytics` object.
        if self.agent_analytics:
            self.agent_analytics.reset(env_is_rendering)
