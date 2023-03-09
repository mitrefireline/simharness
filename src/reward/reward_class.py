from collections import OrderedDict as ordered_dict
from typing import Dict, List, OrderedDict, Tuple
from copy import deepcopy
import numpy as np
from abc import ABC, abstractmethod


class Reward_Simharness():
    def __init__(self):
        # VARIABLES TRACKED ACROSS THE ENTIRE EXPERIMENT - ALL THE SIMULATIONS
        # -------------------------------------------------------------------
        
        ##  lowest number of undamaged squares achieved across all of the simulations in an experiment
        self.lowest_exp_undamaged = 0


        # VARIABLES TRACKED ACROSS THE FULL RUN OF A SIMULATION
        # -------------------------------------------------------------------

        ##  list that stores the number of damaged squares during each timestep of the same simulation with no agent mitigation
        self.damaged_per_timestep_benchmarkSim = [0]





        # VARIABLES TRACKED/UPDATED FOR EACH SIMULATION TIMESTEP WITH NO AGENT - BENCHMARK SIMULATION
        # -------------------------------------------------------------------
        
        ##  number of timesteps that have occurred in the current benchmark simulation
        self.num_timesteps_benchmarkSim = 1

        ##  number of new squares that are burning in current timestep of benchmark simulation
        self.recent_damaged_benchmarkSim = 1




        # VARIABLES TRACKED/UPDATED FOR EACH SIMULATION TIMESTEP WITH AGENT - AGENT SIMULATION
        # -------------------------------------------------------------------
        
        ##  number of timesteps that have occurred in the current simulation
        self.num_timesteps = 1

        ##  number of steps the agent has taken in the current simulation
        self.num_agent_steps = 0

        ##  the speed of the agent in the current simulation
        self.agent_speed = 1

        ##  the total number of squares that have been burned in the simulation
        self.num_burned = 0

        ##  the number of squares that are currently burning in the simulation
        self.burning = 0

        ##  the number of squares that were burned in the most recent timestep of the simulation
        self.recent_burned = 0

        ##  the total number of squares that have mitigations placed by the agent(s)
        self.num_mitigations = 0

        ##  the number of mitigations placed in the most recent timestep
        self.recent_mitigations

        ##  the number of remaining squares in the simulation in this timestep that are undamaged
        ###   the number of squares that neither have a mitigation or were burned/burning
        self.num_undamaged

        ##  did the agent place a mitigation in this timestep
        self.mitigation_placed = False

        ##  is the agent in a burning square within this timestep
        self.agent_burning = False

        ##  is the agent in a square that has already been burned within this timestep
        self.agent_in_burned_area = False

        ##  is the agent close to a burning square within this timestep
        self.agent_near_burning_area = False


    # Update the Reward Class Variables after each timestep of the Benchmark Simulation
    def timestep_BenchSim_Reward_Variables_Update(self,timestep, num_burning_Benchmark):
        
        self.num_timesteps_benchmarkSim = timestep

        #add in the value of the # of NEW burning squares into the self.damaged_per_timestep_benchmarkSim list for this timestep
        self.damaged_per_timestep_benchmarkSim[self.num_timesteps_benchmarkSim] = num_burning_Benchmark - self.damaged_per_timestep_benchmarkSim[self.num_timesteps_benchmarkSim - 1]

        #define the self.recent_damaged_benchmarkSim as this value
        self.recent_damaged_benchmarkSim = self.damaged_per_timestep_benchmarkSim[self.num_timesteps_benchmarkSim]

    # Update the Reward Class Variables after each timestep of the Benchmark Simulation
    def timestep_BenchSim_Reward_Variables_Update(self,timestep, num_burning_Benchmark):
        
    
