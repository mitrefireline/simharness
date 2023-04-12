"""FIXME: A one line summary of the module or program.

Leave one blank line.  The rest of this docstring should contain an
overall description of the module or program.  Optionally, it may also
contain a brief description of exported classes and functions and/or usage
examples.

Typical usage example:

  foo = ClassFoo()
  bar = foo.FunctionBar()
"""
from abc import ABC, abstractmethod

from ..utils.analytics_tracker import AnalyticsTracker


class BaseReward(ABC):
    """TODO Add class docstring."""

    def __init__(self, tracker: AnalyticsTracker):
        """TODO Add constructor docstring."""
        # FIXME is "tracker" a good name to use? update later if needed.
        self.tracker = tracker

    @abstractmethod
    def get_reward(self, timestep: int, sim_run: bool) -> float:
        """TODO Add docstring."""
        raise NotImplementedError

    @abstractmethod
    def get_intermediate_reward(self, timestep: int) -> float:
        """TODO Add docstring."""
        raise NotImplementedError


class SimpleReward(BaseReward):
    """TODO Add class docstring."""

    def __init__(self, tracker: AnalyticsTracker):
        """TODO Add constructor docstring."""
        super().__init__(tracker)

    def get_reward(self, timestep: int, sim_run: bool) -> float:
        """TODO Add function docstring."""
        # Simulation was not run this timestep, so return intermediate reward
        if not sim_run:
            # No intermediate reward calculation used currently, so 0.0 is returned.
            return self.get_intermediate_reward(timestep)
        
        # Use the data stored in the tracker object to calculate this timesteps reward
        # FIXME burning = np.count_nonzero(fire_map == 1)
        burning = self.tracker.sim_data.num_burning_per_step[timestep]
        # burnt = np.count_nonzero(fire_map == 2)

        # diff = burnt - self.num_burned
        # self.num_burned = burnt

        # firelines = np.count_nonzero(fire_map == 3)

        total = self.simulation.config.area.screen_size**2
        reward = -(burning / total) * 10

        return reward

    def get_intermediate_reward(self, timestep: int) -> float:
        """TODO Add function docstring."""
        return 0.0


class BenchmarkReward(BaseReward):
    """TODO Add class docstring."""

    def __init__(self, tracker: AnalyticsTracker):
        """TODO Add constructor docstring."""
        super().__init__(tracker)

    def calculate_reward(self, timestep: int, simulation_step: bool) -> float:
        """TODO Add function docstring."""
        # based off the benchmark simulation
        bench_sim_data, sim_data = self.tracker.bench_sim_data, self.tracker.sim_data
        # the reward is the difference between the unmitigated # of squares that would be damaged in this timestep and the current # of squares that are damaged in this timestep
        # such that if the mitigations result in more squares saved in a timestep, then we will see a positive reward
        damage_diff = (
            self.tracker.agent_data.num_damaged_per_step[
                self.tracker.agent_data.num_steps
            ]
            - self.tracker.agent_data.num_damaged_per_step[
                self.tracker.agent_data.num_steps - 1
            ]
        )
        reward += (
            self.damaged_per_timestep_benchmarkSim[timestep] - self.recent_damaged
        ) / (self.sim_area)

        if not self.sim_active:
            reward += 10
        # if self.agent_near_burning_area:
        #     reward -= 2.0

        # TODO
        # Add positive reward if agent saves more squares than self.lowest_exp_undamaged and then update self.lowest_exp_undamaged
        # Add postive reward if agent ends fire in less timesteps than self.lowest_timesteps with a >= num of undamaged squares to self.lowest_exp_undamaged
        # Add estimated postive rewards if agent's mitigations make the sim last longer than the benchmark sim
