from simharness2.rewards.base_reward import BaseReward, AreaSavedPropRewardV2
from simharness2.analytics.harness_analytics import ReactiveHarnessAnalytics
from typing import Any, Dict

from simharness2.agents.agent import ReactiveAgent

class ForwardReward(BaseReward):
    def __init__(self, harness_analytics: ReactiveHarnessAnalytics):
        """TODO Add constructor docstring."""
        super().__init__(harness_analytics)
    
    def get_reward(
        self,
        timestep: int,
        sim_run: bool,
        done_episode: bool,
        agents: Dict[Any, ReactiveAgent],
        agent_speed: int
    ) -> float:
        """TODO Add function docstring."""
        movement_reward = 0
        def dist(x, y):
            return abs(x[0] - y[0]) + abs(x[1] - y[1])
        for k in agents:
            agent = agents[k]
            delta_dist = dist(agent.current_position, agent.initial_position) - dist(agent._previous_position, agent.initial_position)
            movement_reward += 0.0001 * delta_dist
        if not sim_run:
            # No intermediate reward calculation used currently, so 0.0 is returned.
            return movement_reward + self.get_timestep_intermediate_reward(timestep)

        burning = self.harness_analytics.sim_analytics.data.burning
        reward = -(burning / self._sim_area)
        reward += movement_reward


        # update self.latest_reward and then return the reward
        self.latest_reward = reward
        return reward

    def get_timestep_intermediate_reward(self, timestep: int) -> float:
        """TODO Add function docstring."""
        # Basic Intermediate reward is 0
        return 0.0

class ForwardRewardV2(AreaSavedPropRewardV2):
    def __init__(self, harness_analytics: ReactiveHarnessAnalytics):
        """TODO Add constructor docstring."""
        super().__init__(harness_analytics)

    def get_reward(
        self,
        timestep: int,
        sim_run: bool,
        done_episode: bool,
        agents: Dict[Any, ReactiveAgent],
        agent_speed: int
    ) -> float:
        reward = super().get_reward(timestep=timestep, sim_run=sim_run, done_episode=done_episode, agents=agents, agent_speed=agent_speed)
        movement_reward = 0
        def dist(x, y):
            return abs(x[0] - y[0]) + abs(x[1] - y[1])
        for k in agents:
            agent = agents[k]
            delta_dist = dist(agent.current_position, agent.initial_position) - dist(agent._previous_position, agent.initial_position)
            movement_reward += 0.0001 * delta_dist
        reward += movement_reward
        return reward