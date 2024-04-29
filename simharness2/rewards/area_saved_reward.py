import logging
from typing import Any, Dict

from simharness2.agents.agent import ReactiveAgent
from simharness2.analytics.harness_analytics import ReactiveHarnessAnalytics
from simharness2.rewards.base_reward import BaseReward

logger = logging.getLogger(__name__)


class AreaSavedReward(BaseReward):
    """Basic counterfactual reward"""

    def __init__(self, harness_analytics: ReactiveHarnessAnalytics):
        """TODO Add constructor docstring."""
        super().__init__(harness_analytics)

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
        if not sim_run:
            reward = 0
            # No intermediate reward calculation used currently, so 0.0 is returned.
            # reward = self.get_timestep_intermediate_reward(
            # timestep=timestep, agents=agents, agent_speed=agent_speed
            # )
        else:
            ## DEFINE VALUES NEEDED FOR REWARD CALCULATION

            # extract the current number of simulation steps in the agent(s) simulation
            sim_steps = self.harness_analytics.sim_analytics.num_sim_steps

            # extract the number of newly damaged squares in the agent(s) simulation
            sim_new_damaged = self.harness_analytics.sim_analytics.data.new_damaged

            # extract the total number of damaged squares in the agent(s) simulation
            sim_total_damaged = self.harness_analytics.sim_analytics.data.total_damaged

            # extract the number of simulation steps that occured in the benchmark simulation
            bench_sim_steps_total = len(
                self.harness_analytics.benchmark_sim_analytics.data.damaged
            )

            # extract the total number of damaged squares in the benchmark simulation
            bench_total_damaged = (
                self.harness_analytics.benchmark_sim_analytics.data.damaged[-1]
            )

            # extract the number of newly damaged squares in the benchmark simulation
            bench_new_damaged = 0
            if sim_steps == 1:
                bench_new_damaged = (
                    self.harness_analytics.benchmark_sim_analytics.data.damaged[0]
                )
            elif sim_steps <= bench_sim_steps_total:
                bench_new_damaged = (
                    self.harness_analytics.benchmark_sim_analytics.data.damaged[
                        (sim_steps - 1)
                    ]
                    - self.harness_analytics.benchmark_sim_analytics.data.damaged[
                        (sim_steps - 2)
                    ]
                )
            else:
                bench_new_damaged = 0.0

            ## REWARD CALCULATION

            # calculate the reward as the difference in newly damaged squares between the agent(s) simulation and the benchmark simulation at the given timestep
            reward = (bench_new_damaged - sim_new_damaged) / bench_total_damaged

        if done_episode:
            reward += 1.0
        reward = {k: reward for k in agents}
        return reward

    def get_timestep_intermediate_reward(self, timestep: int, **kwargs) -> float:
        return 0
