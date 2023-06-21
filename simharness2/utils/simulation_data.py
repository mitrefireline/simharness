from typing import List
from functools import partial
import numpy as np
import pandas as pd

from simfire.sim.simulation import FireSimulation
from simfire.enums import BurnStatus

from simharness2.utils.agent_data import AgentData


class FireSimulationData:
    """FIXME: Docstring for FireSimulationData class.

    metrics tracked after the simulation updates

    """

    def __init__(
        self,
        sim: FireSimulation,
        agent_data_partial: partial,
        is_benchmark: bool = False,
        num_agents: int = 1,
    ):
        """TODO Add docstring.

        Arguments:
            agent_data_partial: A `functools.partial` object that defines the class that
                will be used to monitor and track agent (s) behavior within `self.sim`.

        """
        self._sim = sim
        # Indicates whether this object will track a `benchmark` simulation.
        self.is_benchmark = is_benchmark
        self.agent_data: AgentData = None

        # NOTE: In the MARL case, we can use a dictionary of AgentMetricsTracker objects,
        # where the key is the agent ID. This would replace the `agent_data` below.
        if not self.is_benchmark:
            # Agents only exist in the main simulation.
            self.agent_data = agent_data_partial(sim=self._sim)

        self.sim_df: pd.DataFrame = None  # FIXME use better name?
        # NOTE: `self.sim_df` will be initialized in `self._prepare_agent_df()`.
        self._prepare_df_metadata()

        self.reset()

    def _prepare_df_metadata(self):
        """Prepares the metadata (column names, dtypes, etc.) for the sim dataframe.

        Within this method, the default names and dtypes for the columns of the agent
        dataframe are defined and stored in `self.sim_df_cols` and
        `self.sim_df_dtypes`, respectively. These values are used to initialize the
        `self.sim_df` dataframe within the `self._reset_df()` method.

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
        self.sim_df_cols: List[str] = [
            "sim_step",
            "timestep",
            "unburned_total",
            "burned_total",
            "burning_total",
        ]

        # NOTE: Last 3 columns are not applicable to the benchmark simulation.
        self.sim_df_dtypes = {
            "sim_step": np.uint16,
            "timestep": np.uint16,
            "unburned_total": np.uint16,
            "burned_total": np.uint16,
            "burning_total": np.uint16,
        }
        # Insert columns that are only applicable to the main simulation.
        if not self.is_benchmark:
            self.sim_df_cols.extend(
                [
                    "agent_interactions",
                    "agent_movements",
                    # TODO: do we want to distinguish each mitigation type?
                    "mitigations_total",
                ]
            )
            self.sim_df_dtypes.update(
                {
                    "agent_interactions": np.uint8,
                    "agent_movements": np.uint8,
                    "mitigations_total": np.uint16,
                }
            )
        # FIXME: do we want to index using "timestep" or "sim_step"?
        self.sim_df_index = "sim_step"

    def _reset_df(self):
        """Resets the simulation dataframe (`self.sim_df`) to its initial state."""
        if self.sim_df is not None:
            tmp_df = self.sim_df.iloc[0:0]
            del self.sim_df
            self.sim_df = tmp_df
        else:
            self.sim_df = (
                pd.DataFrame(columns=self.sim_df_cols)
                .astype(self.sim_df_dtypes)
                .set_index(self.sim_df_index)
            )

    def update(self, timestep: int) -> None:
        """TODO Add docstring."""
        # NOTE: We can also get sim_steps with self._sim.elapsed_steps
        self.num_sim_steps += 1
        self.active = self._sim.active

        # Add the current timestep's data to the dataframe.
        # TODO: Is there a better alternative to this df build approach?
        fire_map = self._sim.fire_map
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
                    [self.agent_data.num_interactions_since_last_sim_step],
                    [self.agent_data.num_movements_since_last_sim_step],
                    [fire_map.size - non_mitigated_total],
                ]
            )

        sim_data_dict = dict(zip(self.sim_df_cols, sim_data))
        timestep_df = (
            pd.DataFrame(sim_data_dict)
            .astype(self.sim_df_dtypes)
            .set_index(self.sim_df_index)
        )
        self.sim_df = pd.concat([self.sim_df, timestep_df])

    def reset(self):
        """Reset the attributes of `FireSimulationData` to initial values."""
        # Reset attributes used to store simulation behavior across a single episode.
        self._reset_df()
        self.num_sim_steps = 0
        self.active = True

        # If we are tracking agent behavior, reset the `agent_data` object.
        if self.agent_data:
            self.agent_data.reset()
