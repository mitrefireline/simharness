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


class BaseAnalyticsTracker(ABC):
    """TODO Add class docstring."""

    def __init__(
        self,
        sim_area,
        agent_speed,
        num_agent=1,
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
            benchmark_sim_data: TODO
        """
        # Required attributes that track simulation data across each episode in a run.
        # self.sim_data = sim_data
        # self.sim_screen_size = sim_screen_size
        # Optional attributes that track additional data for the RLHarness.
        # self.run_data = run_data
        # self.agent_data = agent_data
        # self.benchmark_sim_data = benchmark_sim_data
        raise NotImplementedError

    @abstractmethod
    def update_after_one_simulation_step(
        self,
        timestep,
        fire_map: np.ndarray,
        sim_active: bool,
        bench_fire_map: np.ndarray,
        benchsim_active: bool,
        **kwargs,
    ):
        """TODO Add docstring."""
        raise NotImplementedError

    @abstractmethod
    def update_after_one_agent_step(
        self,
        timestep,
        agent_pos: List[int],
        fire_map: np.ndarray,
        interaction: bool,
        **kwargs,
    ):
        """TODO Add docstring."""
        raise NotImplementedError

    @abstractmethod
    def update_after_one_simulation_step_and_reward(self):
        """TODO Add docstring."""
        raise NotImplementedError

    @abstractmethod
    def update_after_one_episode(self, reward):
        """TODO Add docstring."""
        raise NotImplementedError


class AnalyticsTracker(BaseAnalyticsTracker):
    """TODO add docstring"""

    def __init__(self, sim_area, agent_speed, num_agents=1):
        super().__init__(sim_area, agent_speed, num_agents)

        ## VARIABLES TRACKED ACROSS ALL EPISODES
        # ---------------------

        # lowest number of undamaged squares achieved across all of the simulations in an experiment
        self.lowest_undamaged_overall = sim_area + 1

        # lowest number of timesteps used to stop fire
        self.lowest_timesteps = 9999

        # The Number of experiments/episodes and also the current episode that we are in
        self.episode_num = 0

        # Track the current timestep of the episode that we are in
        self.timestep = 0

        # Track the latest episode reward
        self.latest_reward = 0.0

        ## VARIABLES TRACKED within an episode
        # ---------------------

        # Tracker Object for the Metrics from the main_simulation
        self.sim_tracker = SimulationMetricsTracker(sim_area, agent_speed, num_agents)

        # Tracker Object for the Metrics from the bench_simulation
        self.benchsim_tracker = SimulationMetricsTracker(
            sim_area, agent_speed, num_agents
        )

    # run this update function after every action the agent takes
    def update_after_one_agent_step(
        self, timestep, agent_pos: List[int], fire_map: np.ndarray, interaction: bool
    ):
        self.timestep = timestep

        self.sim_tracker.agent_tracker.update(
            self.timestep, agent_pos, fire_map, interaction
        )

    # run this update function after every sim_step within an episode within a simulation
    def update_after_one_simulation_step(
        self,
        timestep,
        fire_map: np.ndarray,
        sim_active: bool,
        bench_fire_map: np.ndarray,
        benchsim_active: bool,
    ):
        self.timestep = timestep

        # update the main simulation
        self.sim_tracker.update(self.timestep, fire_map, sim_active)

        # update the benchmark simulation
        self.benchsim_tracker.update(self.timstep, bench_fire_map, benchsim_active)

    # run this reset function AFTER the final reward is calculated for a sim_step & after every sim_step within an episode within a simulation
    def update_after_one_simulation_step_and_reward(self):
        # reset the agent_tracker only after all of the rewards have been calculated for the sim_step
        self.sim_tracker.agent_tracker.reset_after_sim_update()

    # run this update function at the end of an episode before the next episode
    def update_after_one_episode(self, reward):
        # update the number of episodes
        self.episode_num += 1

        # get the total number of undamaged squares from the sim_tracker object
        simulation_undamaged = self.sim_tracker.num_undamaged

        # update the lowest_undamaged_overall if this value is lower
        if simulation_undamaged < self.lowest_undamaged_overall:
            self.lowest_undamaged_overall = simulation_undamaged

        # get the total number of timesteps until the fire ended from the simtracker object
        simulation_timesteps = self.sim_tracker.num_timesteps

        # update the lowest_undamaged_overall if this value is lower
        if simulation_timesteps < self.lowest_timesteps:
            self.lowest_timesteps = simulation_timesteps

        # update the latest_reward tracker
        self.latest_reward = reward

        # reset the timestep tracker
        self.timestep = 0

        # Finally reset the tracker objects for the sim and the benchsim
        self.sim_tracker.reset()
        self.benchsim_tracker.reset()


# ------------------------------------------------------------------------------------


# metrics tracked after the simulation updates
class SimulationMetricsTracker:
    def __init__(self, sim_area, agent_speed, num_agents=1):
        # number of squares within the simulation
        self.sim_area = sim_area

        # store the speed of the agent
        self.agent_speed = agent_speed

        # store the number of agents
        self.num_agents = num_agents

        # whether the simulation is active during this sim_step
        self.active = True

        # number of simulation updates
        self.num_sim_updates = 0

        # current operating timestep
        self.timestep = 0

        # ---------------------

        # number of burned squares during this sim_step
        self.num_burned = 0

        # number of new burned squares during this sim_step
        self.num_new_burned = 0

        # number of burning squares during this sim_step
        self.num_burning = 0

        # number of new burning squares during this sim_step
        self.num_new_burning = 0

        # number of undamaged squares during the last sim_step
        self.num_undamaged = 0

        # number of squares damaged (burned + burning + mitigations) during the last sim_step
        self.num_new_damaged = 0

        # TODO decide how to initialize array
        # array that tracks the number of new damaged squares at each simulation sim_step
        self.num_damaged_per_step = [0]

        # ----------------------

        # number of total mitigations places during the simulation
        self.num_mitigations

        # number of new mitigations placed during the recent sim_step
        self.num_new_mitigations

        # ----------------------

        # create an agent tracker object for the simulation's agent
        self.agent_tracker = AgentMetricsTracker(self.sim_area)

        # ----------------------

    # run this tracker update function after the agents actions and right after the simulation has updated
    def update(self, timestep, fire_map: np.ndarray, sim_active: bool):
        # track the current timestep
        self.timestep = timestep

        # update the simulation update counter
        self.num_sim_updates += 1

        # update whether the simulation is active
        self.active = sim_active

        # find the current number of burning and burned squares within the updated fire_map
        burned_tmp = np.count_nonzero(fire_map == BurnStatus.BURNED)
        burning_tmp = np.count_nonzero(fire_map == BurnStatus.BURNING)

        # Use the stored previous values of burning and burned to calculate the num_new_burning and num_new_burned squares in this timestep
        self.num_new_burned = burned_tmp - self.burned
        self.num_new_burning = burning_tmp - self.burning

        # set num_new_burning and num_new_burned to 0 if the are negative (indicating no new burned/burning squares)
        if self.num_new_burning < 0:
            self.num_new_burning = 0
        if self.num_new_burned < 0:
            self.num_new_burned = 0

        # Now we update the class values of burned and burning to match the updated simulation
        self.burned = burned_tmp
        self.burning = burning_tmp

        # update the number of new mitigations from the agent_tracker.recent_mitigations
        self.num_new_mitigations = self.agent_tracker.new_mitigations

        # update the total of the mitigation lines places from self.num_new_mitigations that was updated in the agent_updat func
        self.num_mitigations = self.num_mitigations + self.num_new_mitigations

        # Calculate the number of undamaged squares in this updated simulation
        num_undamaged_tmp = (
            self.sim_area - self.burned - self.burning - self.num_mitigations
        )

        # Calulate the number of recently damaged squares based of the old stored number of undamaged squares
        self.num_new_damaged = self.num_undamaged - num_undamaged_tmp

        # set self.num_new_damaged = 0 if it is negative (though this really shouldn't happen)
        if self.num_new_damaged < 0:
            self.num_new_damaged = 0

        # update self.num_damaged_per_step with the new updated self.num_new_damaged
        self.num_damaged_per_step[self.timesteps] = self.num_new_damaged

        # Now can update the self.num_undamaged with its new value
        self.num_undamaged = num_undamaged_tmp

        # Finally reset the agent_tracker object for the next timestep
        # TODO: Should this be moved elsewhere to make calculating the reward easier when using agent_metrics
        # This is currently moved into the larger AnalyticsTracker class
        # self.agent_tracker.reset()

        return

    # reset all of the SimulationMetricsTracker object variables at the end of each episode
    def sim_reset(self):
        self.active = True

        self.timestep = 0

        self.num_sim_updates = 0

        # ---------------------

        self.num_burned = 0

        self.num_new_burned = 0

        self.num_burning = 0

        self.num_new_burning = 0

        self.num_undamaged = 0

        self.num_new_damaged = 0

        self.num_damaged_per_step = [0]

        # ----------------------

        self.num_mitigations

        self.num_new_mitigations

        # ----------------------

        # reset the agent_tracker
        self.agent_tracker.reset_after_episode()


class AgentMetricsTracker:
    def __init__(self, sim_area):
        self.sim_area = sim_area

        # track the current timestep that the agent is operating within
        self.timestep

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

    # update the AgentMetricsTracker object variables after each agent action
    def update(
        self, timestep, agent_pos: List[int], fire_map: np.ndarray, interaction: bool
    ):
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
        self.agent_in_burned_area = self._agent_is_burning(fire_map, agent_pos)

        # update the bool if the agent is nearby the fire
        self.agent_near_burning_area = self._nearby_fire(fire_map, agent_pos)

    # reset a set of the AgentMetricsTracker object variables at the end of each simulation update *after the reward has been calculated
    def reset_after_sim_update(self):
        self.num_agent_actions = 0

        self.mitigation_placed = False

        self.new_mitigations = 0

        self.agent_is_burning = False

        self.agent_in_burned_area = False

        self.agent_near_fire = False

    # reset the set of the AgentMetricsTracker object variables at the end of each episode
    def reset_after_episode(self):
        # reset the agent_trackers previous reset func
        self.agent_tracker.reset_after_sim_update()

        # reset the timesteps within the agent_tracker at the end of an episode]
        self.agent_tracker.timestep = 0

    def _nearby_fire(self, fire_map: np.ndarray, agent_pos: List[int]) -> bool:
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
