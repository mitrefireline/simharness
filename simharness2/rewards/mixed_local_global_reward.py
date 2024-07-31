import logging
from typing import Any, Dict
import numpy as np

from simfire.enums import BurnStatus
from simharness2.agents.agent import ReactiveAgent
from simharness2.analytics.harness_analytics import ReactiveHarnessAnalytics
from simharness2.rewards.area_saved_reward import AreaSavedReward

logger = logging.getLogger(__name__)


class MixedLocalAreaSavedReward(AreaSavedReward):

    def __init__(
        self,
        harness_analytics: ReactiveHarnessAnalytics,
        mixing_coefficient: float = 1.0,
        **kwargs,
    ):
        """TODO Add constructor docstring."""
        super().__init__(harness_analytics)

        self.mixing_coefficient = mixing_coefficient

        # TODO: Decide if we want to keep this as is.
        if kwargs.get("debug"):
            self._in_debug_mode = True
        else:
            self._in_debug_mode = False

    def get_reward(
        self,
        *,
        timestep: int,
        sim_run: bool,
        done_episode: bool,
        agents: Dict[Any, ReactiveAgent],
        agent_speed: int,
        **kwargs,
    ) -> float:
        """TODO Add function docstring."""
        # Retrieve the global reward from the parent class.
        global_reward = super().get_reward(
            timestep=timestep,
            sim_run=sim_run,
            done_episode=done_episode,
            agents=agents,
            agent_speed=agent_speed,
            **kwargs,
        )

        # If sim did not run, return the global reward for all agents.
        if not sim_run:
            return {ag_id: global_reward[ag_id] for ag_id in agents.keys()}

        agent_rewards = {}
        for agent_id, agent in agents.items():
            # Calculate the local reward for each agent.
            local_reward = self.get_local_reward(agent=agent)
            logger.debug(f"Local reward for agent {agent_id}: {local_reward}")
            agent_rewards[agent_id] = (
                self.mixing_coefficient * global_reward[agent_id]
                + (1 - self.mixing_coefficient) * local_reward
            )

        if self._in_debug_mode:
            from PIL import Image
            from os import makedirs

            for agent_id, agent in agents.items():
                logger.debug(f"Saving adj_matrix for agent: {agent_id}")
                im = Image.fromarray(agent.adj_to_mitigation)
                makedirs(f"adjacent_points/ts_{timestep}", exist_ok=True)
                im.save(f"adjacent_points/ts_{timestep}/adj_matrix_{agent_id}.png")

        return agent_rewards

    def get_local_reward(self, agent: ReactiveAgent) -> float:
        """TODO Add function docstring."""
        # Calculate total squares adjacent to mitigation that are burning.
        adj_arr = agent.adj_to_mitigation
        fire_map = self.harness_analytics.sim_analytics.sim.fire_map
        # TODO: Do we want to include BurnStatus.BURNED in the condition?
        burning = fire_map == BurnStatus.BURNING
        burn_adj_to_mitigation = np.sum(np.logical_and(adj_arr, burning))
        if burn_adj_to_mitigation > 0:
            logger.debug(f"burn_adj_to_mitigation for agent: {burn_adj_to_mitigation}")

        # Prepare local reward to return.
        local_reward = burn_adj_to_mitigation / self._sim_area
        return local_reward


class MixedForwardAreaSavedReward(MixedLocalAreaSavedReward):
    def get_local_reward(self, agent: ReactiveAgent) -> float:
        def _manhatten_dist(p1, p2):
            return np.abs(p1[0] - p2[0]) + np.abs(p1[1] - p2[1])

        cur_pos, prev_pos = agent.current_position, agent.previous_position
        init_pos = agent.initial_position
        return _manhatten_dist(cur_pos, init_pos) - _manhatten_dist(prev_pos, init_pos)


class MixedLocalAreaSavedRewardWithFirePenalty(MixedLocalAreaSavedReward):
    def __init__(
        self,
        harness_analytics: ReactiveHarnessAnalytics,
        mixing_coefficient: float = 1.0,
        fire_penalty: float = 0.1,
        **kwargs,
    ):
        self._fire_penalty = fire_penalty
        super().__init__(
            harness_analytics=harness_analytics,
            mixing_coefficient=mixing_coefficient,
            **kwargs,
        )

    def get_local_reward(self, agent: ReactiveAgent) -> float:
        reward = super().get_local_reward(agent)
        sim_analytics = self.harness_analytics.sim_analytics
        agent_data = sim_analytics.agent_analytics.data[agent.agent_id]
        near_fire = agent_data.near_fire
        if near_fire:
            reward -= self._fire_penalty
        return reward
