# Parent Reward_Calc Class that inherits FEAR Data Class to compute the reward function 
#                          Allows users to define and choose between rewared functions for Config reward capabilites - (dgandikota)
                                                                                                                    
from simharness2.EnvData.Env_Analysis_Data_Class import FEAR_Data

from collections import OrderedDict as ordered_dict
from typing import Dict, List, OrderedDict, Tuple
from copy import deepcopy
import numpy as np
from abc import ABC, abstractmethod
from simfire.sim.simulation import Simulation
from simfire.enums import BurnStatus


class Reward_Calc(FEAR_Data):

    def __init__(self, agent_speed, sim_size, reward_option = "base"):
        
        super().__init__(self, agent_speed, sim_size)

        self.reward_option = reward_option


    def calculate_Reward_after_timestep(self, timestep):

        #Enter the User Made Reward Functions here


        if self.reward_option == "num_burning":

            #number of squares burning at this timstep
            return (self.burning)
        
        elif self.reward_option == "num_undamaged":

            #number of squares remaining at this timestep that are undamaged
            return (self.num_undamaged)
        
        else:

            #based off the benchmark simulation
            # the reward is the difference between the unmitigated # of squares that would be damaged in this timestep and the current # of squares that are damaged in this timestep
            # such that if the mitigations result in more squares saved in a timestep, then we will see a positive reward
            return (self.damaged_per_timestep_benchmarkSim[timestep] - self.recent_damaged)/(self.sim_area)

            #TODO
            #Add positive reward if agent saves more squares than self.lowest_exp_undamaged and then update self.lowest_exp_undamaged
            #Add postive reward if agent ends fire in less timesteps than self.lowest_timesteps with a >= num of undamaged squares to self.lowest_exp_undamaged
            #Add estimated postive rewards if agent's mitigations make the sim last longer than the benchmark sim





