"""

Reward Classes to be called in the main environment that derive rewards from the AnalyticsTracker
Used if the tracker object is housed within the reward_class

     -dgandikota, afennelly

"""
from abc import ABC, abstractmethod
from numpy import ndarray
from typing import List
from ..utils.analytics_tracker import AnalyticsTracker


class BaseReward(ABC):
    """Abstract Class for Reward_Class template with the update functions implemented"""

    def __init__(self, tracker: AnalyticsTracker):
        """TODO Add constructor docstring."""
        #reference to the tracker object within the environment
        self.tracker = tracker

    @abstractmethod
    def get_reward(self, sim_run: bool) -> float:
        """TODO Add docstring."""
        raise NotImplementedError

    @abstractmethod
    def get_timestep_intermediate_reward(self) -> float:
        """TODO Add docstring."""
        raise NotImplementedError

    #---------------------


class SimpleReward(BaseReward):

    def __init__(self, tracker: AnalyticsTracker):
        """TODO Add constructor docstring."""
        super().__init__(tracker)

    def get_reward(self, sim_run: bool) -> float:
        """TODO Add function docstring."""

        # if Simulation was not run this timestep, return intermediate reward
        if not sim_run:
            # No intermediate reward calculation used currently, so 0.0 is returned.
            return self.get_timestep_intermediate_reward()
        
        
        # Use the data stored in the tracker object to calculate this timesteps reward

        ## set the simplereward to be the number of new_damaged squares in the main simulation
        new_damaged = self.tracker.sim_tracker.num_new_damaged
         
        #total = self.simulation.config.area.screen_size**2

        #get the total area from the sim_tracker
        total = self.tracker.sim_tracker.sim_area

        reward = -(new_damaged / total) * 100

        #update self.latest_reward and then return the reward
        self.latest_reward = reward
        return reward

    def get_timestep_intermediate_reward(self, timestep: int) -> float:
        """TODO Add function docstring."""
        #Basic Intermediate reward is 0
        return 0.0


class BenchmarkReward(BaseReward):

    def __init__(self, tracker: AnalyticsTracker):
        """TODO Add constructor docstring."""
        super().__init__(tracker)

    def get_reward(self, sim_run: bool) -> float:
        """TODO Add function docstring."""


        # if Simulation was not run this timestep, return intermediate reward
        if not sim_run:
            # intermediate reward calculation used 
            return self.get_timestep_intermediate_reward()   
        
        ## This Reward will compare the number of new recently damaged squares in the main sim and within the bench sim 
        ##       to determine the performance/reward of the agent

        new_damaged_mainsim = self.tracker.sim_tracker.num_new_damaged

        new_damaged_benchsim = self.tracker.benchsim_tracker.num_new_damaged

    
        #write in the edge case for if the benchsim is not active, but the main sim is still active
        if self.tracker.benchsim_tracker.active == False:
            #setting arbitrary maximum possible burning from the benchsim to be half of the total area
            #in general, it is good for the main sim to last longer than the benchsim so this should hopefully yield positive rewards
            new_damaged_benchsim = (self.tracker.benchsim_tracker.sim_area) // 2

        #define the number of squares saved by the agent as the difference between the benchsim and the mainsim
        timestep_number_squares_saved = new_damaged_benchsim - new_damaged_mainsim


        total = self.tracker.sim_tracker.sim_area

        reward = ((timestep_number_squares_saved)/total) * 100.0


        #TODO add larger negative reward if agent gets close to fire

        #TODO add very large negative reward if agent steps into fire (or end the simulation)

        #update self.latest_reward and then return the reward
        self.latest_reward = reward
        return reward

    def get_timestep_intermediate_reward(self) -> float:
        """TODO Add function docstring."""

        #TODO add small negative reward if the agent places mitigation within an already burned area

        #start with the intermediate reward just being the same as the previously calculated reward
        inter_reward = self.latest_reward

        #add a slight reward to the agent for placing a mitigation
        if self.tracker.sim_tracker.agent_tracker.mitigation_placed == True:
            inter_reward += 1

        #update self.latest_reward and then return the intermediate reward  
        self.latest_reward = inter_reward
        return inter_reward


