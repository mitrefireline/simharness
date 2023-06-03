"""

Base AnalyticsTracker for SimHarness and BaseReward

    -dgandikota, afennelly


"""
from abc import ABC, abstractmethod
from typing import Any, Dict, List
import numpy as np
from simfire.enums import BurnStatus
from simfire.sim.simulation import FireSimulation
import math


class RLAnalyticsTracker(ABC):
    """Base class with several built in methods."""

    @abstractmethod
    def __init__(self) -> None:
        """Subclasses must implement there own `__init__` method."""
        pass

    @abstractmethod
    def update_after_one_simulation_step(self):
        """See subclass for docstring."""
        pass

    @abstractmethod
    def update_after_one_agent_step(
        self,
        *,
        mitigation_placed: bool,
        movements: List[str],
        movement: int,
        interaction: int,
        agent_pos: List[int],
        agent_pos_is_empty_space: bool,
    ) -> None:
        """See subclass for docstring."""
        pass

    @abstractmethod
    def update_after_one_simulation_step_and_reward(self):
        """TODO Add docstring."""
        raise NotImplementedError

    @abstractmethod
    def update_after_one_episode(self, reward):
        """See subclass for docstring."""
        pass


class ReactiveAnalyticsTracker(RLAnalyticsTracker):
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

        # Store objects used to track simulation data within each episode in a run.
        try:
            self.sim_data: FireSimulationMetricsTracker = sim_data_partial(sim=sim)
            if benchmark_sim:
                self.benchmark_sim_data: FireSimulationMetricsTracker = sim_data_partial(
                    sim=benchmark_sim, is_benchmark=True
                )
        except TypeError as e:
            raise e

        self.reset()

    def update_after_one_agent_step(
        self,
        *,
        mitigation_placed: bool,
        movements: List[str],
        movement: int,
        interaction: int,
        agent_pos: List[int],
        agent_pos_is_empty_space: bool,
    ) -> None:
        """Calls `self.sim_data.agent_tracker.update()`, if agents are in the sim.

        This method is intended to be called directly after the
        `_do_one_agent_step()` method defined in the `ReactiveHarness` class.

        Arguments: 
            mitigation_placed: A boolean indicating if the agent placed a mitigation line
                during this timestep.
            movements: A list of strings indicating the available movements for the agent.
            movement: An integer indicating the index of the movement that the agent
                selected.
            interaction: An integer indicating the index of the interaction that the agent
                selected.
            agent_pos: A list of integers indicating the current position of the agent.
            agent_pos_is_empty_space: A boolean indicating if the agent is currently in an
                empty space.
        """
        if self.sim_data.agent_tracker:
            self.sim_data.agent_tracker.update(
                mitigation_placed=mitigation_placed,
                movements=movements,
                movement=movement,
                interaction=interaction,
                agent_pos=agent_pos,
                agent_pos_is_empty_space=agent_pos_is_empty_space,
            )

    def update_after_one_simulation_step(self):
        """Calls `update()` on `self.sim_data` (`self.benchmark_sim_data`, if exists).

        This method is intended to be called directly after the
        `_do_one_simulation_step()` method defined in the `ReactiveHarness` class.
        """
        self.sim_data.update()

        if self.benchmark_sim_data:
            self.benchmark_sim_data.update()

        benchsim_active = self.benchmark_sim_data.active
        # Use this to update the self.bench_timesteps and the self.bench_damage
        if benchsim_active == False and self.bench_estimated == False:
            # if the benchsim has reached it's end, then use this to set the values of the variables
            self.bench_timesteps = self.benchmark_sim_data.num_sim_steps
            sim_area = self.sim_data._sim.config.area.screen_size**2
            self.bench_damage = sim_area - self.benchmark_sim_data.num_undamaged
            self.bench_estimated = True

        # use this to initialize the self.bench_timesteps and the self.bench_damage if the bench_sim has not ended before the main_sim yet
        # TODO make this more efficient or just have the benchsim run once before the agent makes any actions
        elif self.bench_estimated == False:
            if self.benchmark_sim_data.num_sim_steps > self.bench_timesteps:
                self.bench_timesteps = self.benchmark_sim_data.num_sim_steps + 1

            if (sim_area - self.benchmark_sim_data.num_undamaged) > self.bench_damage:
                self.bench_damage = (sim_area - self.benchmark_sim_data.num_undamaged) + 1

    # run this reset function AFTER the final reward is calculated for a sim_step & after every sim_step within an episode within a simulation
    def update_after_one_simulation_step_and_reward(self):
        # reset the agent_tracker only after all of the rewards have been calculated for the sim_step
        if self.sim_data.agent_tracker:
            self.sim_data.agent_tracker.reset_after_sim_update()

    def update_after_one_episode(self, reward: float):
        """TODO Add docstring."""
        # run this update function at the end of an episode before the next episode
        # update the number of episodes
        self.episodes_total += 1

        # get the total number of undamaged squares from the sim_tracker object
        sim_undamaged = self.sim_data.num_undamaged

        # Update the highest_undamaged_overall if episode value is higher.
        if sim_undamaged > self.max_episode_unburned_squares:
            self.max_episode_unburned_squares = sim_undamaged

        # Update the lowest total timesteps used to stop fire if episode value is lower.
        if self.sim_data.num_sim_steps < self.min_episode_sim_steps:
            self.min_episode_sim_steps = self.sim_data.num_sim_steps

        # update the latest_reward tracker
        self.latest_reward = reward

        # reset the timestep tracker
        self.timestep = 0

        # Finally reset the tracker objects for the sim and the benchsim
        self.sim_tracker.reset()
        self.benchsim_tracker.reset()


# ------------------------------------------------------------------------------------

    def reset(self):
        """TODO Add docstring."""
        # Define metrics that are tracked across all episodes in a run.
        # Highest number of undamaged squares achieved during a single-episode across the
        # course of a trial.
        # FIXME currently, this doesn't seem to be used for any reward calculations?
        self.max_episode_unburned_squares = -1

        # lowest number of timesteps used to stop fire
        # FIXME currently, this doesn't seem to be used for any reward calculations?
        # FIXME better default value?
        self.min_episode_sim_steps = 9999

        # Stores the current episode
        # NOTE: ray tracks this via `ray.tune.result.EPISODES_TOTAL`
        self.episodes_total = 0

        # Track the current timestep of the episode that we are in
        # Incremented after `update...agent_step` and `update...simulation_step`.
        # FIXME use better variable name: `timesteps_total` or `timesteps_this_episode`?
        # NOTE: ray tracks this via `ray.tune.result.TIMESTEPS_TOTAL`
        # self.timestep = 0

        # Track the latest episode reward
        # TODO is this the reward for the latest timestep or the latest episode?
        self.latest_reward = 0.0

        self.sim_data.reset()

        if self.benchmark_sim_data:
            # Track the avg BenchSim timesteps
            self.bench_timesteps = 0
            # self.min_episode_benchmark_sim_steps = -1

            # Track the avg BenchSim damage total
            self.bench_damage = 0
            # self.max_episode_benchmark_unburned_squares = -1

            # FIXME what is `bench_estimated` trying to represent?
            # Bool to determine if the bench metrics were intialized by the bench sim, or an estimation from the main sim
            self.bench_estimated = True

            self.benchmark_sim_data.reset()

# metrics tracked after the simulation updates
class FireSimulationMetricsTracker:
    """FIXME: Docstring for FireSimulationMetricsTracker class."""

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
        self.agent_tracker: AgentMetricsTracker = None

        # NOTE: In the MARL case, we can use a dictionary of AgentMetricsTracker objects,
        # where the key is the agent ID. This would replace the `agent_tracker` below.
        if not self.is_benchmark:
            # Agents only exist in the main simulation.
            self.agent_tracker = agent_data_partial(self._sim)

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
        if self.agent_tracker:
            self.num_new_mitigations = (
                self.agent_tracker.num_interactions_since_last_sim_step
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

        # Finally reset the agent_tracker object for the next timestep
        # TODO: Should this be moved elsewhere to make calculating the reward easier when using agent_metrics
        # This is currently moved into the larger AnalyticsTracker class
        # self.agent_tracker.reset()

        return

    def reset(self):
        """TODO Add docstring."""
        # reset the SimulationMetricsTracker object variables at the end of each episode
        self.active = True

        self.num_sim_steps: int = 0


        # ---------------------

        self.num_burned = 0

        self.num_new_burned = 0

        self.num_burning = 0

        self.num_new_burning = 0

        self.num_undamaged = 0

        self.num_new_damaged = 0

        self.num_damaged_per_step = [0]

        # ----------------------
        # We do not need to track mitigation lines in the benchmark simulation.
        self.num_mitigations_total: int = 0 if not self.is_benchmark else None

        self.num_new_mitigations: int = 0 if not self.is_benchmark else None

        # ----------------------
        # TODO: Indicate (maybe in docstring?) that `agent_tracker` is reset here.
        if self.agent_tracker:
            self.agent_tracker.reset()

class AgentMetricsTracker:
    """Monitors and tracks the behavior of a single agent within the simulation."""

    def __init__(self, sim: FireSimulation):
        """TODO: Docstring for __init__."""
        self.sim_area = sim_area

        # track the current timestep that the agent is operating within
        self.timestep = 0

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
        # track the current timestep
        self.timestep = timestep

        # track how many actions the agent has taken within the timestep
        self.num_agent_actions += 1

        # update the mitigation_placed bool and the new_mitigations count if the agent has placed a mitigation
        if interaction:
            self.new_mitigations += 1
            self.mitigation_placed = True

        # TODO add tracker variable to store the latest action and then pass that information into the update arguments

        # update the bool if the agent is burning
        self.agent_burning = self._agent_is_burning(fire_map, agent_pos)

        # update the bool if the agent is operating within already burnt area
        self.agent_in_burned_area = self._agent_in_burned_area(fire_map, agent_pos)

        # update the bool if the agent is nearby the fire
        self.agent_near_fire = self._nearby_fire(fire_map, agent_pos)

    # reset a set of the AgentMetricsTracker object variables at the end of each simulation update *after the reward has been calculated

    def reset_after_one_simulation_step(self) -> None:
        """Reset values that are tracked between each simulation step."""
        self.num_agent_actions = 0

        self.mitigation_placed = False

        self.new_mitigations = 0

        self.agent_is_burning = False

        self.agent_in_burned_area = False

        self.agent_near_fire = False

    def reset(self):
        """Reset the AgentMetricsTracker to initial values."""
        # reset the agent_trackers previous reset func
        self.agent_tracker.reset_after_sim_update()

        # reset the timesteps within the agent_tracker at the end of an episode]
        self.agent_tracker.timestep = 0

    def _agent_nearby_fire(self, fire_map: np.ndarray, agent_pos: List[int]) -> bool:
        """Check if the agent is adjacent to a space that is currently burning.

        Returns:
            nearby_fire: A boolean indicating if there is a burning space adjacent to the
            agent.
        """
        nearby_locs = []
        screen_size = math.sqrt(self.sim_area)
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
        if (fire_map[agent_pos[0]][agent_pos[1]]) == BurnStatus.BURNING:
            return True

        return False

    def _agent_in_burned_area(self, fire_map: np.ndarray, agent_pos: List[int]) -> bool:
        # return true if the agent is in a burning square
        if (fire_map[agent_pos[0]][agent_pos[1]]) == BurnStatus.BURNED:
            return True

        return False
