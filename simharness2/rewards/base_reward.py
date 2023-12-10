"""Base Reward Class for representing the modular reward function.

Reward Classes to be called in the main environment that derive rewards from the
ReactiveHarnessAnalytics object.
"""
from abc import ABC, abstractmethod
import logging

from simharness2.analytics.harness_analytics import ReactiveHarnessAnalytics

logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
handler.setFormatter(
    logging.Formatter("%(asctime)s\t%(levelname)s %(filename)s:%(lineno)s -- %(message)s")
)
logger.addHandler(handler)
logger.propagate = False


class BaseReward(ABC):
    """Abstract Class for Reward_Class template with the update functions implemented."""

    def __init__(self, harness_analytics: ReactiveHarnessAnalytics):
        """TODO Add constructor docstring."""
        # reference to the harness_analytics object within the environment
        self.harness_analytics = harness_analytics
        # helper variable indicating the total number of squares in the simulation map
        self._sim_area = (
            self.harness_analytics.sim_analytics.sim.config.area.screen_size[0] ** 2
        )

    @abstractmethod
    def get_reward(self, timestep: int, sim_run: bool) -> float:
        """TODO Add docstring."""
        pass

    @abstractmethod
    def get_timestep_intermediate_reward(self, timestep: int) -> float:
        """TODO Add docstring."""
        pass


class SimpleReward(BaseReward):
    """TODO add description."""

    def __init__(self, harness_analytics: ReactiveHarnessAnalytics):
        """TODO Add constructor docstring."""
        super().__init__(harness_analytics)

    def get_reward(self, timestep: int, sim_run: bool) -> float:
        """TODO Add function docstring."""
        if not sim_run:
            # No intermediate reward calculation used currently, so 0.0 is returned.
            return self.get_timestep_intermediate_reward(timestep)

        burning = self.harness_analytics.sim_analytics.data.burning
        reward = -(burning / self._sim_area)

        # update self.latest_reward and then return the reward
        self.latest_reward = reward
        return reward

    def get_timestep_intermediate_reward(self, timestep: int) -> float:
        """TODO Add function docstring."""
        # Basic Intermediate reward is 0
        return 0.0

class AreaSavedPropReward(BaseReward):
    """Reward as described in paper based on the incremental proportion of
        Area saved, or area that was not burned, burning, or mitigated, at each timestep (t), when comparing
        the mitigated simulation (Sim) to the unmitigated benchmark simulation (Bench)"""

    def __init__(self, harness_analytics: ReactiveHarnessAnalytics):
        """TODO Add constructor docstring."""
        super().__init__(harness_analytics)

    def get_reward(self, timestep: int, sim_run: bool) -> float:
        """TODO Add function docstring."""
        if not sim_run:
            # No intermediate reward calculation used currently, so 0.0 is returned.
            return self.get_timestep_intermediate_reward(timestep)

        ## DEFINE VALUES NEEDED FOR REWARD CALCULATION

        #extract the current number of simulation steps in the agent(s) simulation
        sim_steps = self.harness_analytics.sim_analytics.num_sim_steps

        #extract the number of newly damaged squares in the agent(s) simulation
        sim_new_damaged = self.harness_analytics.sim_analytics.data.new_damaged

        #extract the total number of damaged squares in the agent(s) simulation
        sim_total_damaged = self.harness_analytics.sim_analytics.data.total_damaged

        #extract the number of simulation steps that occured in the benchmark simulation
        bench_sim_steps_total = len(self.harness_analytics.benchmark_sim_analytics.data.damaged)

        #extract the total number of damaged squares in the benchmark simulation
        bench_total_damaged = self.harness_analytics.benchmark_sim_analytics.data.damaged[-1]

        #extract the number of newly damaged squares in the benchmark simulation
        bench_new_damaged = 0
        if sim_steps == 1:
            bench_new_damaged = self.harness_analytics.benchmark_sim_analytics.data.damaged[0]
        elif sim_steps <= bench_sim_steps_total:
            bench_new_damaged = self.harness_analytics.benchmark_sim_analytics.data.damaged[(sim_steps - 1)] - self.harness_analytics.benchmark_sim_analytics.data.damaged[(sim_steps - 2)]
        else:
            bench_new_damaged = 0.0

        ## REWARD CALCULATION

        #calculate the reward as the difference in newly damaged squares between the agent(s) simulation and the benchmark simulation at the given timestep

        if sim_steps <= bench_sim_steps_total:
            
            reward = ((bench_new_damaged*1.0 - sim_new_damaged) / bench_total_damaged*1.0)

        else:
            #account for when the agent(s) simulation has lasted longer than the benchmark simulation

            bench_new_damaged = 0.0
            reward = ((bench_new_damaged*1.0 - sim_new_damaged) / bench_total_damaged*1.0)


        #account for if the agent(s) simulation has ended in fewer steps than the benchmark simulation
        if (self.harness_analytics.sim_analytics.active == False) & (sim_steps < bench_sim_steps):

            bench_rest_damaged = self.harness_analytics.benchmark_sim_analytics.data.damaged[-1] - self.harness_analytics.benchmark_sim_analytics.data.damaged[(sim_steps - 1)]

            sim_rest_damaged = 0.0

            reward = reward + (bench_rest_damaged*1.0 - sim_rest_damaged / bench_total_damaged*1.0)

        # update self.latest_reward and then return the reward
        self.latest_reward = reward

        return reward

    def get_timestep_intermediate_reward(self, timestep: int) -> float:
        """Basic Intermediate reward is the last sim step reward
         + a small amount if the agent successfully places a mitigation and the total squares damaged does not exceed the benchmark sim
         """

        #extract the total number of damaged squares in the benchmark simulation
        bench_total_damaged = self.harness_analytics.benchmark_sim_analytics.data.damaged[-1]
        #extract the total number of damaged squares in the agent(s) simulation
        sim_total_damaged = self.harness_analytics.sim_analytics.data.total_damaged
        #calculate the fractional reward partial given for each agent's successful mitigation placement
        mitigation_bonus = 1.0/(self.num_agents * self.agent_speed)

        reward = self.latest_reward

        #if the total area damaged in the agent(s) simulation is less than the total area damaged in the benchmark simulation
        if sim_total_damaged < bench_total_damaged:
            for agent_id, agent in self.agents.items():
                if agent.mitigation_placed == True:
                    #FIXME for multidiscrete action space
                    reward = reward + (mitigation_bonus/bench_total_damaged)

        return reward


class AreaSavedPropRewardV2(BaseReward):
    """Reward as described in paper based on the incremental proportion of
        Area saved, or area that was not burned, burning, or mitigated, at each timestep (t), when comparing
        the mitigated simulation (Sim) to the unmitigated benchmark simulation (Bench)
        
        Additional Conditions are given to add lower bounds of reward = 0 when the maximum possible damage that occurs at each timestep
         in the agent(s) simulation is bounded to the maximum possible damage that occurs at the benchmark simulation timestep. """

    def __init__(self, harness_analytics: ReactiveHarnessAnalytics):
        """TODO Add constructor docstring."""
        super().__init__(harness_analytics)

    def get_reward(self, timestep: int, sim_run: bool) -> float:
        """TODO Add function docstring."""
        if not sim_run:
            # No intermediate reward calculation used currently, so 0.0 is returned.
            return self.get_timestep_intermediate_reward(timestep)

        ## DEFINE VALUES NEEDED FOR REWARD CALCULATION

        #extract the current number of simulation steps in the agent(s) simulation
        sim_steps = self.harness_analytics.sim_analytics.num_sim_steps

        #extract the number of newly damaged squares in the agent(s) simulation
        sim_new_damaged = self.harness_analytics.sim_analytics.data.new_damaged

        #extract the total number of damaged squares in the agent(s) simulation
        sim_total_damaged = self.harness_analytics.sim_analytics.data.total_damaged

        #extract the number of simulation steps that occured in the benchmark simulation
        bench_sim_steps_total = len(self.harness_analytics.benchmark_sim_analytics.data.damaged)

        #extract the total number of damaged squares in the benchmark simulation
        bench_total_damaged = self.harness_analytics.benchmark_sim_analytics.data.damaged[-1]

        #extract the number of newly damaged squares in the benchmark simulation
        bench_new_damaged = 0
        if sim_steps == 1:
            bench_new_damaged = self.harness_analytics.benchmark_sim_analytics.data.damaged[0]
        elif sim_steps <= bench_sim_steps_total:
            bench_new_damaged = self.harness_analytics.benchmark_sim_analytics.data.damaged[(sim_steps - 1)] - self.harness_analytics.benchmark_sim_analytics.data.damaged[(sim_steps - 2)]
        else:
            bench_new_damaged = 0.0

        ## REWARD CALCULATION

        #calculate the reward as the difference in newly damaged squares between the agent(s) simulation and the benchmark simulation at the given timestep

        if sim_steps <= bench_sim_steps_total:

            #ensure that the maximum possible damage that occurs at timestep in the agent(s) simulation is equivalent to the maximum possible damage that occurs at the benchmark simulation timestep
            if sim_new_damaged > bench_new_damaged:
                sim_new_damaged = bench_new_damaged
            
            reward = ((bench_new_damaged*1.0 - sim_new_damaged) / bench_total_damaged*1.0)

        else:
            #account for when the agent(s) simulation has lasted longer than the benchmark simulation

            #if the total area damaged in the agent(s) simulation is less than the total area damaged in the benchmark simulation
            if sim_total_damaged < bench_total_damaged:
                bench_new_damaged = 0.0
                reward = ((bench_new_damaged*1.0 - sim_new_damaged) / bench_total_damaged*1.0)
            
            else:
                #if the total area damaged in the agent(s) simulation now exceeds the area damaged in the benchmark simulation, treat the total reward as 0
                reward = 0.0

        #account for if the agent(s) simulation has ended in fewer steps than the benchmark simulation
        if (self.harness_analytics.sim_analytics.active == False) & (sim_steps < bench_sim_steps):

            #augment the reward with the number of potential squares saved compared to the benchsim
            if sim_total_damaged < bench_total_damaged:

                bench_rest_damaged = self.harness_analytics.benchmark_sim_analytics.data.damaged[-1] - self.harness_analytics.benchmark_sim_analytics.data.damaged[(sim_steps - 1)]

                sim_rest_damaged = 0.0

                reward = reward + (bench_rest_damaged*1.0 - sim_rest_damaged / bench_total_damaged*1.0)

        # update self.latest_reward and then return the reward
        self.latest_reward = reward

        return reward

    def get_timestep_intermediate_reward(self, timestep: int) -> float:
        """Basic Intermediate reward is the last sim step reward
         + a small amount if the agent successfully places a mitigation and the total squares damaged does not exceed the benchmark sim
         OR
         - a small amount if the agent successfully places a mitigation and the total squares damaged exceeds the benchmark sim"""
        

        #extract the total number of damaged squares in the benchmark simulation
        bench_total_damaged = self.harness_analytics.benchmark_sim_analytics.data.damaged[-1]
        #extract the total number of damaged squares in the agent(s) simulation
        sim_total_damaged = self.harness_analytics.sim_analytics.data.total_damaged
        #calculate the fractional reward partial given for each agent's successful mitigation placement
        mitigation_bonus = 1.0/(self.num_agents * self.agent_speed)

        reward = self.latest_reward

        #if the total area damaged in the agent(s) simulation is less than the total area damaged in the benchmark simulation
        if sim_total_damaged < bench_total_damaged:
            for agent_id, agent in self.agents.items():
                if agent.mitigation_placed == True:
                    #FIXME for multidiscrete action space
                    reward = reward + (mitigation_bonus/bench_total_damaged)

        #else if the total area damaged in the agent(s) simulation is more than the total area damaged in the benchmark simulation
        if sim_total_damaged > bench_total_damaged:
            for agent_id, agent in self.agents.items():
                if agent.mitigation_placed == True:
                    #FIXME for multidiscrete action space
                    reward = reward - (mitigation_bonus/bench_total_damaged)

        return reward
        