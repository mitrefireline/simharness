"""

Base AnalyticsTracker for SimHarness and BaseReward

    -dgandikota, afennelly


"""
from abc import ABC, abstractmethod
from typing import List
from dataclasses import dataclass
import numpy as np
import math
from functools import partial
import logging

from simfire.enums import BurnStatus
from simfire.sim.simulation import FireSimulation


@dataclass
class BestEpisodePerformance:
    """Stores best performance (wrt reactive fire scenario) across all episodes in trial.

    Attributes:
        max_unburned: An int storing the maximum number of tiles in the main
            `FireSimulation.fire_map` that are `BurnStatus.UNBURNED` across all episodes
            in a trial. At the end of each episode, this value is updated if the number
            of `BurnStatus.UNBURNED` tiles is greater than the current value.
        max_unburned_rescaled: A float storing a rescaled value of `max_unburned`, where min-max
            normalization is used to perform the rescaling. The rescaled value will fall
            between 0 and 1 (inclusive). Intuitively, this value represents the
            maximum proportion of "land" that was "saved" by the agent's actions.
        num_sim_steps: An int storing the number of simulation steps that occurred in the
            episode with the maximum number of `BurnStatus.UNBURNED` tiles.
        episode: An int storing the episode number that corresponds to the best episode
            performance.
        reward: A float storing the cumulative reward achieved after the "best" episode.
    """

    max_unburned: int
    sim_area: int
    num_sim_steps: int
    episode: int
    reward: float

    def __post_init__(self):
        self.max_unburned_rescaled = self.max_unburned / self.sim_area


class RLHarnessData(ABC):
    """Base class with several built in methods."""

    def __init__(
        self,
        *,
        sim: FireSimulation,
        sim_data_partial: partial,
        benchmark_sim: FireSimulation = None,
    ) -> None:
        """TODO: Add docstring."""
        # Store objects used to track simulation data within each episode in a run.
        try:
            self.sim_data = sim_data_partial(sim=sim)
            if benchmark_sim:
                self.benchmark_sim_data = sim_data_partial(
                    sim=benchmark_sim, is_benchmark=True
                )
            else:
                self.benchmark_sim_data = None
        except TypeError as e:
            raise e

        self.best_episode_performance: BestEpisodePerformance = None

    @abstractmethod
    def update_after_one_simulation_step(self, *, timestep: int) -> None:
        """See subclass for docstring."""
        pass

    @abstractmethod
    def update_after_one_agent_step(
        self,
        *,
        timestep: int,
        movement: int,
        interaction: int,
        agent_pos: List[int],
    ) -> None:
        """See subclass for docstring."""
        pass

    @abstractmethod
    def update_after_one_harness_step(
        self, sim_run: bool, terminated: bool, reward: float, *, timestep: int
    ) -> None:
        """See subclass for docstring."""
        pass

    @abstractmethod
    def reset(self):
        """See subclass for docstring."""
        pass


class ReactiveHarnessData(RLHarnessData):
    """TODO add docstring"""

    def __init__(
        self,
        *,
        sim: FireSimulation,
        sim_data_partial: partial,
        benchmark_sim: FireSimulation = None,
    ) -> None:
        """TODO Add summary line.

        Arguments:
            sim: The underlying `FireSimulation` object that contains the agent (s) that
                are being trained. The agent (s) will place mitigation lines, and the
                simulation will spread the fire. An episode terminates when the fire is
                finished spreading.
            sim_data_partial: A `functools.partial` object that defines the class that will
                be used to monitor and track `self.sim`, and `self.benchmark_sim`, if the
                optional `benchmark_sim` is provided. The user is expected to provide the
                `agent_data_partial` keyword argument, along with a valid value.
            benchmark_sim: A separate `FireSimulation` object, identical to
                `sim` (after initialization). No mitigation lines will be placed in this
                simulation, as it does not contain any agent (s).

        Raises:
            TypeError: If `sim_data_partial.keywords` does not contain a
            `agent_data_partial` key with value of type `functools.partial`.

        """
        super().__init__(
            sim=sim, sim_data_partial=sim_data_partial, benchmark_sim=benchmark_sim
        )
        # Define attributes that are needed/accessed within `ComprehensiveReward` class.
        # TODO: Address where these attributes should be stored, see
        # https://gitlab.mitre.org/fireline/reinforcementlearning/simharness2/-/merge_requests/6#note_1504742
        if self.benchmark_sim_data:
            self.bench_timesteps: int = 0
            self.bench_damage: int = 0
            self.bench_estimated: bool = False

        # Track the latest episode reward
        # TODO is this the reward for the latest timestep or the latest episode?
        # FIXME: Decide how and where this attribute is/should be used.
        self.latest_reward = 0.0

        self.episodes_total = 0

    def update_after_one_agent_step(
        self,
        *,
        timestep: int,
        movement: int,
        interaction: int,
        agent_pos: List[int],
    ) -> None:
        """Updates `self.sim_data.agent_data`, if agents are in the sim.

        This method is intended to be called directly after the call to
        `ReactiveHarness._do_one_agent_step()` (within `ReactiveHarness.step()`).

        Arguments:
            timestep: An integer indicating the current timestep of the episode.
            movement: An integer indicating the index of the latest movement that the
                agent selected.
            interaction: An integer indicating the index of the latest interaction that
                the agent selected.
            agent_pos: A list of integers indicating the current position of the agent.
        """
        if self.sim_data.agent_data:
            self.sim_data.agent_data.update(timestep, movement, interaction, agent_pos)

    def update_after_one_simulation_step(self, *, timestep: int) -> None:
        """Updates `self.sim_data` (and `self.benchmark_sim_data`, if exists).

        This method is intended to be called directly after the call to
        `ReactiveHarness._do_one_simulation_step()` (within `ReactiveHarness.step()`).

        Arguments:
            timestep: An integer indicating the current timestep of the episode.
        """
        self.sim_data.update(timestep)

        if self.benchmark_sim_data:
            self.benchmark_sim_data.update(timestep)

        sim_area = self.sim_data._sim.fire_map.size
        # FIXME mention in docstring that this logic is performed. need to condense!!
        benchsim_active = self.benchmark_sim_data.active
        benchsim_undamaged = self.benchmark_sim_data.sim_df.iloc[-1]["unburned_total"]
        # Use this to update the self.bench_timesteps and the self.bench_damage
        if benchsim_active == False and self.bench_estimated == False:
            # if the benchsim has reached it's end, then use this to set the values of the variables
            self.bench_timesteps = self.benchmark_sim_data.num_sim_steps
            self.bench_damage = sim_area - benchsim_undamaged
            self.bench_estimated = True

        # use this to initialize the self.bench_timesteps and the self.bench_damage if the bench_sim has not ended before the main_sim yet
        # TODO make this more efficient or just have the benchsim run once before the agent makes any actions
        elif self.bench_estimated == False:
            if self.benchmark_sim_data.num_sim_steps > self.bench_timesteps:
                self.bench_timesteps = self.benchmark_sim_data.num_sim_steps + 1

            if sim_area - benchsim_undamaged > self.bench_damage:
                self.bench_damage = sim_area - benchsim_undamaged + 1

    def update_after_one_harness_step(
        self, sim_run: bool, terminated: bool, reward: float, *, timestep: int
    ) -> None:
        # Reset any attributes that monitor agent behavior between each simulation step.
        if sim_run and self.sim_data.agent_data:
            self.sim_data.agent_data.reset_after_one_simulation_step()

        # Once episode has terminated, check if episode performance is the best so far.
        if terminated:
            self.episodes_total += 1
            log = logging.getLogger(__name__)
            # log.info(f"Episode {self.episodes_total} has terminated.")

            sim_df = self.sim_data.sim_df
            current_unburned = (
                0 if len(sim_df) == 0 else sim_df.iloc[-1]["unburned_total"]
            )
            update_best_episode_performance = True
            if self.best_episode_performance:
                max_unburned = self.best_episode_performance.max_unburned
                if current_unburned <= max_unburned:
                    update_best_episode_performance = False

            if update_best_episode_performance:
                self.best_episode_performance = BestEpisodePerformance(
                    max_unburned=current_unburned,
                    sim_area=self.sim_data._sim.fire_map.size,
                    num_sim_steps=self.sim_data.num_sim_steps,
                    episode=self.episodes_total,
                    reward=reward,
                )

    def reset(self):
        """Resets attributes that track data within each episode.

        This method is intended to be called within after the call to
        `ReactiveHarness._do_one_agent_step()` (within `ReactiveHarness.step()`).

        """
        self.sim_data.reset()

        if self.benchmark_sim_data:
            self.benchmark_sim_data.reset()


class FireSimulationMetricsTracker:
    """FIXME: Docstring for FireSimulationMetricsTracker class.

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
        self.agent_data: AgentMetricsTracker = None

        # NOTE: In the MARL case, we can use a dictionary of AgentMetricsTracker objects,
        # where the key is the agent ID. This would replace the `agent_data` below.
        if not self.is_benchmark:
            # Agents only exist in the main simulation.
            self.agent_data = agent_data_partial(sim=self._sim)

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
        if self.agent_data:
            self.num_new_mitigations = (
                self.agent_data.num_interactions_since_last_sim_step
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

        # Finally reset the agent_data object for the next timestep
        # TODO: Should this be moved elsewhere to make calculating the reward easier when using agent_metrics
        # This is currently moved into the larger AnalyticsTracker class
        # self.agent_data.reset()

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

        # TODO: Indicate (maybe in docstring?) that `agent_data` is reset here.
        if self.agent_data:
            self.agent_data.reset()


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
