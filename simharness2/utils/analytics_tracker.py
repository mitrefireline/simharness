# FiReLine Environment Analysis & Reward (FEAR) Data Class to dynamically extract, store and transform key variable data from RL Agent Training Experiments
#                                                                to create Config reward capabilites and offer additional experimental analysis
#                                                                                                                                   - (dgandikota)

from abc import ABC, abstractmethod
# other possible names - FEAR_D : FiReLine Environment Analysis & Reward Data
#                     - FETA : FiReLine Environment Training Analysis
from collections import OrderedDict as ordered_dict
from copy import deepcopy
from typing import Dict, List, OrderedDict, Tuple

import numpy as np
from simfire.enums import BurnStatus
from simfire.sim.simulation import Simulation

# define and update this class object within the environment file code


class AnalyticsTracker:
    def __init__(self, agent_speed, sim_size):

        # VARIABLES TRACKED ACROSS THE ENTIRE EXPERIMENT - ALL THE SIMULATIONS
        # -------------------------------------------------------------------

        ##  lowest number of undamaged squares achieved across all of the simulations in an experiment
        self.lowest_exp_undamaged = 0

        ## lowest number of timesteps used to stop fire
        self.lowest_timesteps

        # VARIABLES TRACKED ACROSS THE FULL RUN OF A SIMULATION
        # -------------------------------------------------------------------

        ##  list that stores the number of damaged squares during each timestep of the same simulation with no agent mitigation
        self.damaged_per_timestep_benchmarkSim = [0]

        ##  the speed of the agent in the current simulation
        self.agent_speed = agent_speed

        ## the simulation size in area (num squares)
        self.sim_area = sim_size

        # VARIABLES TRACKED/UPDATED FOR EACH SIMULATION TIMESTEP WITH NO AGENT - BENCHMARK SIMULATION
        # -------------------------------------------------------------------

        ##  number of timesteps that have occurred in the current benchmark simulation
        self.num_timesteps_benchmarkSim = 1

        ##  number of new squares that are burning in current timestep of benchmark simulation
        self.recent_damaged_benchmarkSim = 1

        ## track whether bench_sim is active
        self.bench_sim_active = True

        # VARIABLES TRACKED/UPDATED FOR EACH SIMULATION TIMESTEP WITH AGENT - AGENT SIMULATION
        # -------------------------------------------------------------------

        ##  number of timesteps that have occurred in the current simulation
        self.num_timesteps = 1

        ##  number of steps the agent has taken in the current simulation
        self.num_agent_steps = 0

        ## track whether simulation with agent is active
        self.sim_active = True

        ##  the total number of squares that have been burned in the simulation
        self.num_burned = 0

        ##  the number of squares that are currently burning in the simulation
        self.burning = 0

        ##  the number of squares that are burned in the most recent timestep of the simulation
        self.recent_burned = 0

        ##  the number of squares that are burning in the most recent timestep of the simulation
        self.recent_burning = 0

        ##  the total number of squares that have mitigations placed by the agent(s)
        self.num_mitigations = 0

        ##  the number of mitigations placed in the most recent timestep
        self.recent_mitigations

        ##  the number of remaining squares in the simulation in this timestep that are undamaged
        ###   the number of squares that neither have a mitigation or were burned/burning
        self.num_undamaged

        ##  number of new squares that are damaged in current timestep of Agent simulation
        self.recent_damaged = 1

        ##  did the agent place a mitigation in this timestep
        self.mitigation_placed = False

        ##  is the agent in a burning square within this timestep
        self.agent_burning = False

        ##  is the agent in a square that has already been burned within this timestep
        self.agent_in_burned_area = False

        ##  is the agent close to a burning square within this timestep
        self.agent_near_burning_area = False

    # Update the FEAR Data Class Variables after each timestep of the Benchmark Simulation
    def timestep_BenchSim_FEAR_Update(
        self, timestep, Benchmark_fire_map, Benchmark_sim_active
    ):

        # TODO integrate functionality for when the bench sim is not active

        self.bench_sim_active = Benchmark_sim_active

        self.num_timesteps_benchmarkSim = timestep

        # Use the fire_map to calculate the number of currently burning squares in the benchmark sim
        num_burning_Benchmark = np.count_nonzero(Benchmark_fire_map == BurnStatus.BURNING)

        # add in the value of the # of NEW burning squares into the self.damaged_per_timestep_benchmarkSim list for this timestep
        self.damaged_per_timestep_benchmarkSim[self.num_timesteps_benchmarkSim] = (
            num_burning_Benchmark
            - self.damaged_per_timestep_benchmarkSim[self.num_timesteps_benchmarkSim - 1]
        )

        # define the self.recent_damaged_benchmarkSim as this value
        self.recent_damaged_benchmarkSim = self.damaged_per_timestep_benchmarkSim[
            self.num_timesteps_benchmarkSim
        ]

    # Update the FEAR Data Class Variables after each Agent action/movement of the Agent Simulation - Useful when Agent Speed is greater than 1
    def AgentStep_FEAR_Update(
        self, timestep, simulation, mitigation_placed=False, nearby_fire=False
    ):

        self.num_timesteps = timestep

        self.num_agent_steps = self.num_agent_steps + 1

        self.mitigation_placed = mitigation_placed

        if self.mitigation_placed:
            self.recent_mitigations = self.recent_mitigations + 1

        # make sure that this fire_map is has the agent's most recent mitigation recorded

        # TODO
        self.agent_burning = False
        # track this to add big negative reward

        # TODO
        self.agent_in_burned_area = False
        # subtract from recent_mitigations if the mitigation was placed in a burned area

        # TODO
        self.agent_near_burning_area = nearby_fire
        # track this to add negative reward for Agent being too close to fire

    # Update the FEAR Data Class Variables after each timestep of the Agent-Simulation
    def timestep_AgentSim_FEAR_Update(
        self, timestep, AgentSim_fire_map, AgentSim_sim_active
    ):

        self.num_timesteps = timestep

        self.sim_active = AgentSim_sim_active

        # make sure that this fire_map is has all the agent's most recent mitigation recorded and has run through the simulation to see the effects
        # get the burned and burning squares from this fire_map
        burned = np.count_nonzero(AgentSim_fire_map == BurnStatus.BURNED)

        burning = np.count_nonzero(AgentSim_fire_map == BurnStatus.BURNING)

        # Use the stored previous values of burning and num_burned to calculate the NEW burning and burned squares in this timestep
        self.recent_burning = burning - self.burning

        self.recent_burned = burned - self.num_burned

        # update the stored values of burned and num_burned from this new fire_map
        self.num_burned = np.count_nonzero(AgentSim_fire_map == BurnStatus.BURNED)

        self.burning = np.count_nonzero(AgentSim_fire_map == BurnStatus.BURNING)

        self.num_mitigations = self.num_mitigations + self.recent_mitigations

        # Calculate number of squares damaged in this timestep and the total undamaged squares left
        self.recent_damaged = (
            self.recent_burning + self.recent_burned + self.recent_mitigations
        )

        self.num_undamaged = (
            self.sim_area - self.burning - self.burned - self.num_mitigations
        )

    # TODO
    # create method to inheret the "VARIABLES TRACKED ACROSS THE ENTIRE EXPERIMENT" when a new reward_class object is defined for each new simulation within an experiment
