"""

Reward Classes to be called in the main environment that derive rewards from the AnalyticsTracker
Used if the tracker object is housed within the reward_class

     -dgandikota, afennelly

"""
from abc import ABC, abstractmethod

from simharness2.analytics.harness_analytics import RLHarnessData


class BaseReward(ABC):
    """Abstract Class for Reward_Class template with the update functions implemented"""

    def __init__(self, tracker: RLHarnessData):
        """TODO Add constructor docstring."""
        # reference to the tracker object within the environment
        self.tracker = tracker
        # helper variable indicating the total number of squares in the simulation map
        self._sim_area = self.tracker.sim_data._sim.config.area.screen_size**2

    @abstractmethod
    def get_reward(self, timestep: int, sim_run: bool) -> float:
        """TODO Add docstring."""
        pass

    @abstractmethod
    def get_timestep_intermediate_reward(self, timestep: int) -> float:
        """TODO Add docstring."""
        pass


class SimpleReward(BaseReward):
    """TODO Add class docstring."""

    def __init__(self, tracker: RLHarnessData):
        """TODO Add constructor docstring."""
        super().__init__(tracker)

    def get_reward(self, timestep: int, sim_run: bool) -> float:
        """TODO Add function docstring."""
        # Simulation was not run this timestep, so return intermediate reward
        if not sim_run:
            # No intermediate reward calculation used currently, so 0.0 is returned.
            return self.get_timestep_intermediate_reward()

        # Use the data stored in the tracker object to calculate this timesteps reward

        ## set the simplereward to be the number of new_damaged squares in the main simulation
        new_damaged = self.tracker.sim_data.num_new_damaged

        reward = -(new_damaged / self._sim_area) * 100

        # update self.latest_reward and then return the reward
        self.latest_reward = reward
        return reward

    def get_timestep_intermediate_reward(self, timestep: int) -> float:
        """TODO Add function docstring."""
        # Basic Intermediate reward is 0
        return 0.0


class BenchmarkReward(BaseReward):
    def __init__(self, tracker: RLHarnessData):
        """TODO Add constructor docstring."""
        super().__init__(tracker)

    def get_reward(self, timestep: int, sim_run: bool) -> float:
        """TODO Add function docstring."""

        # if Simulation was not run this timestep, return intermediate reward
        if not sim_run:
            # intermediate reward calculation used
            return self.get_timestep_intermediate_reward(timestep)

        ## This Reward will compare the number of new recently damaged squares in the main sim and within the bench sim
        ##       to determine the performance/reward of the agent

        new_damaged_mainsim = self.tracker.sim_tracker.num_new_damaged

        new_damaged_benchsim = self.tracker.benchsim_tracker.num_new_damaged

        # write in the edge case for if the benchsim is not active, but the main sim is still active
        if self.tracker.benchsim_tracker.active == False:
            # setting arbitrary maximum possible burning from the benchsim to be half of the total area
            # in general, it is good for the main sim to last longer than the benchsim so this should hopefully yield positive rewards
            new_damaged_benchsim = (self._sim_area) // 2

        # define the number of squares saved by the agent as the difference between the benchsim and the mainsim
        timestep_number_squares_saved = new_damaged_benchsim - new_damaged_mainsim

        reward = ((timestep_number_squares_saved) / self._sim_area) * 100.0

        # TODO add larger negative reward if agent gets close to fire

        # TODO add very large negative reward if agent steps into fire (or end the simulation)

        # update self.latest_reward and then return the reward
        self.latest_reward = reward
        return reward

    def get_timestep_intermediate_reward(self, timestep: int) -> float:
        """TODO Add function docstring."""

        # TODO add small negative reward if the agent places mitigation within an already burned area

        # start with the intermediate reward just being the same as the previously calculated reward
        inter_reward = self.latest_reward

        # add a slight reward to the agent for placing a mitigation
        if self.tracker.sim_tracker.agent_tracker.mitigation_placed == True:
            inter_reward += 1

        # update self.latest_reward and then return the intermediate reward
        self.latest_reward = inter_reward
        return inter_reward


# Reward Function that mirrors Dhanuj's complex_reward from last year and takes into account agent location to the fire while also
#   inducing a positive reward structure where the agent is rewarded for undamaged squares in the simulation - this reward function is task agnostic
class ComprehensiveReward(BaseReward):
    def __init__(self, tracker: RLHarnessData):
        """TODO Add constructor docstring."""
        super().__init__(tracker)

    def get_reward(self, timestep: int, sim_run: bool) -> float:
        """TODO Add function docstring."""

        # if Simulation was not run this timestep, return intermediate reward
        if not sim_run:
            # intermediate reward calculation used
            return self.get_timestep_intermediate_reward(timestep)

        ## This Reward will compare the number of new recently damaged squares in the main sim and within the bench sim
        ##       to determine the performance/reward of the agent

        undamaged_mainsim = self.tracker.sim_tracker.num_undamaged

        undamaged_benchsim = self.tracker.benchsim_tracker.num_undamaged

        # if the benchsim is no longer active, but the main simulation is still active,
        #   then it is okay to use the last value of the num_undamaged from the benchsim as assumadley our main_sim agent will be rewarded for sustaining the fire longer

        # define the number of squares saved by the agent as the difference between the benchsim and the mainsim
        timestep_number_squares_saved = undamaged_mainsim - undamaged_benchsim

        total = self.tracker.sim_tracker.sim_area

        reward = ((timestep_number_squares_saved) / total) * 100.0

        ## If MAIN SIMULATION ENDS FASTER THAN BENCH SIMULATION
        #   there are either two possibilities
        #   1. The agent's actions sped up the fire
        #           - In which case the agent should be heavily penalized
        #   2. The agent's actions ended the fire faster and saved more squares (The most ideal Situation)
        #           - In which case the agent should be heavily rewarded
        if (
            self.tracker.benchsim_tracker.active == True
            and self.tracker.sim_tracker.active == False
        ):
            # update the value of the undamaged benchsim to be that of the bench simulation if it reached it's end
            undamaged_benchsim = self._sim_area - self.tracker.bench_damage

            timestep_number_squares_saved = undamaged_mainsim - undamaged_benchsim

            # multiply this by the number of timesteps that the main_sim is faster than the bench sim
            if self.tracker.bench_timesteps > self.tracker.timestep:
                timestep_number_squares_saved = timestep_number_squares_saved * (
                    self.tracker.bench_timesteps - self.tracker.timestep
                )

        # this new reward works out well for both of the above cases
        #   For Case 1., this reward will yield a large negative reward
        #   For Case 2., this reward will yield a large positive reward
        reward = ((timestep_number_squares_saved) / self._sim_area) * 100.0

        ## AUGMENT THE REWARD IF AGENT GETS TOO CLOSE TO THE FIRE
        # use static reward so RL easily learns what causes this reward
        # TODO: determine best amount for this reward
        if self.tracker.sim_tracker.agent_tracker.agent_near_fire == True:
            # set the reward to be -50
            reward = -50

        # TODO add very large negative reward if agent steps into fire (or end the simulation)

        # update self.latest_reward and then return the reward
        self.latest_reward = reward
        return reward

    def get_timestep_intermediate_reward(self, timestep: int) -> float:
        """TODO Add function docstring."""

        # start with the intermediate reward just being the same as the previously calculated reward
        inter_reward = self.latest_reward

        # add a slight reward to the agent for placing a mitigation not in a burned area
        if (
            self.tracker.sim_tracker.agent_tracker.mitigation_placed == True
            and self.tracker.sim_tracker.agent_tracker.agent_in_burned_area == False
        ):
            inter_reward += 1

        # update self.latest_reward and then return the intermediate reward
        self.latest_reward = inter_reward
        return inter_reward
