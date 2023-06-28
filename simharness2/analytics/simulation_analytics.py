"""TODO: A brief description of the module and its purpose.

TODO: Add a list of any classes, exception, functions, and any other objects exported by
the module.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, List
from functools import partial

import numpy as np
import pandas as pd

from simfire.enums import BurnStatus
from simfire.sim.simulation import FireSimulation
from simharness2.analytics.agent_analytics import AgentAnalytics, AgentMetricsTracker


class SimulationAnalytics(ABC):
    """Interface used to monitor metrics using the `fire_map` within a `FireSimulation`.

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
    ):
        """TODO Add docstring.

        Arguments:
            sim: The `FireSimulation` object that will be tracked.
            agent_analytics_partial: A `functools.partial` object that defines the class that
                will be used to monitor and track agent (s) behavior within `self.sim`.
            is_benchmark: TODO

        NOTE: In the MARL case, we can use a dictionary of AgentAnalytics objects,
        where the key is the agent ID. This would change the type of `agent_analytics`.

        NOTE: `self.df` will be initialized in `self._reset_df()`, which is called within
        `self.reset()`.
        """
        # Store a reference to the `FireSimulation` object that is being tracked.
        self.sim = sim
        # Indicates whether this object will track a `benchmark` simulation.
        self.is_benchmark = is_benchmark
        self.agent_analytics: AgentAnalytics = None

        if not self.is_benchmark:
            # Agents only exist in the main simulation.
            self.agent_analytics = agent_analytics_partial(sim=self.sim)

        # Define stubs for the class attributes.
        self.df: pd.DataFrame = None
        self.df_cols: List[str]
        self.df_dtypes: Dict[str, Any]
        self.df_index: str

        self.prepare_df_metadata()
        self.reset()

    def reset(self):
        """Reset the attributes of `FireSimulationData` to initial values."""
        # Reset attributes used to store simulation behavior across a single episode.
        self._reset_df()
        self.num_sim_steps = 0
        self.active = True

        # If we are tracking agent behavior, reset the `agent_analytics` object.
        if self.agent_analytics:
            self.agent_analytics.reset()

    @abstractmethod
    def prepare_df_metadata(self):
        """Prepares the metadata (column names, dtypes, etc.) for the sim dataframe."""
        pass

    @abstractmethod
    def update(self, timestep: int) -> None:
        """TODO Add docstring."""
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


class FireSimulationAnalytics(SimulationAnalytics):
    """Use `FireSimulationAnalytics` to monitor `fire_map` the within a `FireSimulation`.

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
        """
        super().__init__(sim, agent_analytics_partial, is_benchmark)

    def prepare_df_metadata(self):
        """Prepares the metadata (column names, dtypes, etc.) for the sim dataframe.

        Within this method, the default names and dtypes for the columns of the agent
        dataframe are defined and stored in `self.df_cols` and
        `self.df_dtypes`, respectively. These values are used to initialize the
        `self.df` dataframe within the `self._reset_df()` method.

        Columns used to store the simulation's behavior:
            - `sim_step`: current simulation step in the episode.
            - `timestep`: current timestep in the episode.
            - `agent_interactions`: total number of interactions that were performed by
                the agent (s) since the last simulation step (`sim_step - 1`).
            - `agent_movements`: total number of movements that were performed by the
                agent (s) since the last simulation step (`sim_step - 1`).
            - `unburned_total`: total number of tiles in `self._sim.fire_map` that have
                `BurnStatus.UNBURNED`.
            - `burned_total`: total number of tiles in `self._sim.fire_map` that have
                `BurnStatus.BURNED`.
            - `burning_total`: total number of tiles in `self._sim.fire_map` that have
                `BurnStatus.BURNING`.
            - `mitigations_total`: total number of tiles in `self._sim.fire_map` that
                contain a mitigation line. This equates to tiles that are any of
                `BurnStatus.FIRELINE`, `BurnStatus.WETLINE`, `BurnStatus.SCRATCHLINE`.
        """
        # Define the columns that will be used to store the simulation's behavior.
        self.df_cols: List[str] = [
            "sim_step",
            "timestep",
            "unburned_total",
            "burned_total",
            "burning_total",
        ]

        # NOTE: Last 3 columns are not applicable to the benchmark simulation.
        self.df_dtypes = {
            "sim_step": np.uint16,
            "timestep": np.uint16,
            "unburned_total": np.uint16,
            "burned_total": np.uint16,
            "burning_total": np.uint16,
        }
        # Insert columns that are only applicable to the main simulation.
        if not self.is_benchmark:
            self.df_cols.extend(
                [
                    "agent_interactions",
                    "agent_movements",
                    # TODO: do we want to distinguish each mitigation type?
                    "mitigations_total",
                ]
            )
            self.df_dtypes.update(
                {
                    "agent_interactions": np.uint8,
                    "agent_movements": np.uint8,
                    "mitigations_total": np.uint16,
                }
            )
        # FIXME: do we want to index using "timestep" or "sim_step"?
        self.df_index = "sim_step"

    def update(self, timestep: int) -> None:
        """TODO Add docstring."""
        # NOTE: We can also get sim_steps with self._sim.elapsed_steps
        self.active = self.sim.active

        # Add the current timestep's data to the dataframe.
        # TODO: Is there a better alternative to this df build approach?
        fire_map = self.sim.fire_map
        burned_total = np.sum(fire_map == BurnStatus.BURNED)
        burning_total = np.sum(fire_map == BurnStatus.BURNING)
        unburned_total = np.sum(fire_map == BurnStatus.UNBURNED)
        sim_data = [
            [self.num_sim_steps],
            [timestep],
            [unburned_total],
            [burned_total],
            [burning_total],
        ]
        if not self.is_benchmark:
            non_mitigated_total = burned_total + burning_total + unburned_total
            sim_data.extend(
                [
                    [self.agent_analytics.num_interactions_since_last_sim_step],
                    [self.agent_analytics.num_movements_since_last_sim_step],
                    [fire_map.size - non_mitigated_total],
                ]
            )

        sim_data_dict = dict(zip(self.df_cols, sim_data))
        timestep_df = (
            pd.DataFrame(sim_data_dict).astype(self.df_dtypes).set_index(self.df_index)
        )
        self.df = pd.concat([self.df, timestep_df])

        self.num_sim_steps += 1 # increment AFTER method logic is performed (convention).

    def reset(self):
        """Reset the attributes of `FireSimulationData` to initial values."""
        # Reset attributes used to store simulation behavior across a single episode.
        self._reset_df()
        self.num_sim_steps = 0
        self.active = True

        # If we are tracking agent behavior, reset the `agent_analytics` object.
        if self.agent_analytics:
            self.agent_analytics.reset()


class FireSimulationMetricsTracker:
    """FIXME: Docstring for FireSimulationMetricsTracker class.

    metrics tracked after the simulation updates
    """

    def __init__(
        self,
        sim: FireSimulation,
        agent_analytics_partial: partial,
        is_benchmark: bool = False,
    ):
        """TODO Add docstring.

        Arguments:
            agent_analytics_partial: A `functools.partial` object that defines the class that
                will be used to monitor and track agent (s) behavior within `self.sim`.

        """
        self._sim = sim
        # Indicates whether this object will track a `benchmark` simulation.
        self.is_benchmark = is_benchmark
        self.agent_analytics: AgentMetricsTracker = None

        # NOTE: In the MARL case, we can use a dictionary of AgentMetricsTracker objects,
        # where the key is the agent ID. This would replace the `agent_analytics` below.
        if not self.is_benchmark:
            # Agents only exist in the main simulation.
            self.agent_analytics = agent_analytics_partial(sim=self._sim)

        self.reset()

    def update(self) -> None:
        """TODO Add docstring."""
        # run this tracker update function after the agents actions and right after the
        # simulation has updated
        # track the current timestep
        self.num_sim_steps += 1

        # update the simulation update counter
        self.active = self._sim.active

        # Calculate the number of currently burned (burning) squares in this timestep.
        num_currently_burned = np.sum(self._sim.fire_map == BurnStatus.BURNED)
        num_currently_burning = np.sum(self._sim.fire_map == BurnStatus.BURNING)

        # Calculate the number of newly burned (burning) squares in this timestep.
        self.num_new_burned = num_currently_burned - self.num_burned
        self.num_new_burning = num_currently_burning - self.num_burning

        # FIXME refactor into a separate method?
        # Set values to 0 if they are negative (indicates no new burned/burning squares).
        if self.num_new_burning < 0:
            self.num_new_burning = 0
        if self.num_new_burned < 0:
            self.num_new_burned = 0

        self.num_burned = num_currently_burned
        self.num_burning = num_currently_burning

        # Update values for attributes tracking mitigation lines.
        if self.agent_analytics:
            self.num_new_mitigations = (
                self.agent_analytics.num_interactions_since_last_sim_step
            )
            self.num_mitigations_total += self.num_new_mitigations

        # Calculate the number of currently undamaged squares in this timestep.
        # TODO: verify that `UNBURNED` is the correct `BurnStatus` to use here.
        num_currently_undamaged = np.sum(self._sim.fire_map == BurnStatus.UNBURNED)
        self.num_new_damaged = self.num_undamaged - num_currently_undamaged

        # FIXME refactor into a separate method?
        # Set values to 0 if they are negative (though this really shouldn't happen).
        if self.num_new_damaged < 0:
            self.num_new_damaged = 0

        self.num_damaged_per_step.append(self.num_new_damaged)

        # Now can update the self.num_undamaged with its new value
        self.num_undamaged = num_currently_undamaged

        # Finally reset the agent_analytics object for the next timestep
        # TODO: Should this be moved elsewhere to make calculating the reward easier when using agent_metrics
        # This is currently moved into the larger AnalyticsTracker class
        # self.agent_analytics.reset()

        return

    def reset(self):
        """TODO Add docstring."""
        # reset the SimulationMetricsTracker object variables at the end of each episode
        self.active = True
        self.num_sim_steps: int = 0
        self.num_burned = 0
        self.num_new_burned = 0
        self.num_burning = 0
        self.num_new_burning = 0
        self.num_undamaged = 0
        self.num_new_damaged = 0
        self.num_damaged_per_step = [0]

        # We do not need to track mitigation lines in the benchmark simulation.
        self.num_mitigations_total: int = 0 if not self.is_benchmark else None

        self.num_new_mitigations: int = 0 if not self.is_benchmark else None

        # TODO: Indicate (maybe in docstring?) that `agent_analytics` is reset here.
        if self.agent_analytics:
            self.agent_analytics.reset()
