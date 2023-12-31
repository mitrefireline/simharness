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


class BenchmarkReward(BaseReward):
    """TODO: add description."""

    def __init__(self, harness_analytics: ReactiveHarnessAnalytics):
        """TODO Add constructor docstring."""
        super().__init__(harness_analytics)

    def get_reward(self, timestep: int, sim_run: bool) -> float:
        """TODO Add function docstring."""
        # if Simulation was not run this timestep, return intermediate reward
        if not sim_run:
            # intermediate reward calculation used
            return self.get_timestep_intermediate_reward(timestep)

        # This Reward will compare the number of new recently damaged squares in the main
        # sim and within the bench sim to determine the performance/reward of the agent

        new_damaged_mainsim = self.harness_analytics.sim_analytics.num_new_damaged

        new_damaged_benchsim = (
            self.harness_analytics.benchmark_sim_analytics.num_new_damaged
        )

        # write in the edge case for if the benchsim is not active, but the main sim is
        # still active
        if self.harness_analytics.benchmark_sim_analytics.active is False:
            # setting arbitrary maximum possible burning from the benchsim to be half of
            # the total area in general, it is good for the main sim to last longer than
            # the benchsim so this should hopefully yield positive rewards
            new_damaged_benchsim = (self._sim_area) // 2

        # define the number of squares saved by the agent as the difference between the
        # benchsim and the mainsim
        timestep_number_squares_saved = new_damaged_benchsim - new_damaged_mainsim

        reward = ((timestep_number_squares_saved) / self._sim_area) * 100.0

        # TODO add larger negative reward if agent gets close to fire

        # TODO add very large negative reward if agent steps into fire
        # (or end the simulation)

        # update self.latest_reward and then return the reward
        self.latest_reward = reward
        return reward

    def get_timestep_intermediate_reward(self, timestep: int) -> float:
        """TODO Add function docstring."""
        # TODO add small negative reward if the agent places mitigation within an already
        # burned area

        # start with the intermediate reward just being the same as the previously
        # calculated reward
        inter_reward = self.latest_reward

        # add a slight reward to the agent for placing a mitigation
        if self.harness_analytics.sim_analytics.agent_analytics.mitigation_placed is True:
            inter_reward += 1

        # update self.latest_reward and then return the intermediate reward
        self.latest_reward = inter_reward
        return inter_reward


class ComprehensiveReward(BaseReward):
    """TODO add description."""

    def __init__(
        self,
        *,
        harness_analytics: ReactiveHarnessAnalytics,
        fixed_reward: float,
        static_penalty: float,
        invalid_movement_penalty: float,
    ):
        """Induces a postive reward structure using the number of undamaged squares.

        The method used for reward calculation is intended to mirror Dhanuj's
        `complex_reward` from FY22, where the agent is rewarded using the difference in
        the number of `BurnStatus.UNBURNED` tiles between the main and benchmark
        simulations. This reward structure is intended to be task agnostic, and makes an
        effort to penalize the agent for unsafe proximity to the fire (`static_penalty`).

        Attributes:
            tracker: The `ReactiveHarnessAnalytics` object that houses the data used to
                calculate the reward.
            fixed_reward: The fixed reward that is scaled by the number of squares saved
                by the agent.
            static_penalty: The fixed penalty that is applied to the agent if it is
                within a certain distance of the fire.
            invalid_movement_penalty: The fixed penalty that is applied to the agent if
                it attempts to move to a square that is not contained within the bounds
                of the `FireSimulation.fire_map`.
        """
        self.fixed_reward = fixed_reward
        # TODO: rename to `self.near_fire_penalty`
        self.static_penalty = static_penalty
        self.invalid_movement_penalty = invalid_movement_penalty
        super().__init__(harness_analytics)

    def get_reward(self, timestep: int, sim_run: bool) -> float:
        """Rewards the agent for `saving` squares from the fire wrt the benchmark sim.

        A constant scalar reward, `self.fixed_reward`, is scaled by the number of squares
        "saved" by the agent, which equates to the difference in the number of
        `BurnStatus.UNBURNED` tiles between the main `FireSimulation.fire_map` and the
        benchmark `FireSimulation.fire_map`.

        Arguments:
            timestep: The current timestep in the episode.
            sim_run: Whether or not the simulation was run this timestep.

        Returns:
            (Fixed reward * number of squares saved) - (penalty for being near fire)
        """
        if not sim_run:
            return self.get_timestep_intermediate_reward(timestep)

        # This Reward will compare the number of new recently damaged squares in the
        # main sim and within the bench sim

        # Get the number of undamaged squares in the main and benchmark simulations
        # FIXME: should we index with `-1` or `timestep - 1`?
        # NOTE: Convert to int to avoid "overflow in scalar subtract"(cols are
        # np.uint16!!)
        undamaged_mainsim = self.harness_analytics.sim_analytics.data.unburned
        undamaged_benchsim = self.harness_analytics.benchmark_sim_analytics.data.unburned
        # if the benchsim is no longer active, but the main simulation is still active,
        #   then it is okay to use the last value of the num_undamaged from the benchsim
        # as assumadley our main_sim agent will be rewarded for sustaining the fire longer

        # define the number of squares saved by the agent as the difference between the
        # benchsim and the mainsim
        timestep_number_squares_saved = undamaged_mainsim - undamaged_benchsim

        # If MAIN SIMULATION ENDS FASTER THAN BENCH SIMULATION
        #   there are either two possibilities
        #   1. The agent's actions sped up the fire
        #           - In which case the agent should be heavily penalized
        #   2. The agent's actions ended the fire faster and saved more squares (The most
        #           ideal Situation)
        #           - In which case the agent should be heavily rewarded
        if (
            self.harness_analytics.benchmark_sim_analytics.active is True
            and self.harness_analytics.sim_analytics.active is False
        ):
            # update the value of the undamaged benchsim to be that of the bench
            # simulation if it reached it's end
            undamaged_benchsim = self._sim_area - self.harness_analytics.bench_damage

            timestep_number_squares_saved = undamaged_mainsim - undamaged_benchsim

            # multiply this by the number of timesteps that the main_sim is faster than
            # the bench sim
            if (
                self.harness_analytics.bench_timesteps
                > self.harness_analytics.sim_analytics.num_sim_steps
            ):
                timestep_number_squares_saved = timestep_number_squares_saved * (
                    self.harness_analytics.bench_timesteps
                    - self.harness_analytics.sim_analytics.num_sim_steps
                )

        # this new reward works out well for both of the above cases
        #   For Case 1., this reward will yield a large negative reward
        #   For Case 2., this reward will yield a large positive reward
        reward = ((timestep_number_squares_saved) / self._sim_area) * self.fixed_reward

        # FIXME handle MARL case
        agent_id = next(
            iter(self.harness_analytics.sim_analytics.agent_analytics.data.keys())
        )
        # AUGMENT THE REWARD IF AGENT GETS TOO CLOSE TO THE FIRE
        # use static reward so RL easily learns what causes this reward
        # TODO: determine best amount for this reward
        if self.harness_analytics.sim_analytics.agent_analytics.data[agent_id].near_fire:
            # set the reward to be -1.0 * static_penalty
            reward = -self.static_penalty

        # Penalize agent if chosen movement would result in an invalid map position.
        # if not self.harness_analytics.sim_analytics.agent_analytics.df.iloc[-1][
        if self.harness_analytics.sim_analytics.agent_analytics.data[
            agent_id
        ].moved_off_map:
            reward -= self.invalid_movement_penalty

        # TODO add very large negative reward if agent steps into fire
        # (or end the simulation)

        # update self.latest_reward and then return the reward
        self.latest_reward = reward
        return reward

    def get_timestep_intermediate_reward(self, timestep: int) -> float:
        """TODO Add function docstring."""
        # start with the intermediate reward just being the same as the previously
        # calculated reward
        inter_reward = self.harness_analytics.latest_reward

        # add a slight reward to the agent for placing a mitigation not in a burned area
        # FIXME handle MARL case
        # if self.harness_analytics.sim_analytics.agent_analytics.data.near_fire:
        #     inter_reward += 1

        # Penalize agent if chosen movement would result in an invalid map position.
        # FIXME handle MARL case
        # if self.harness_analytics.sim_analytics.agent_analytics.data.moved_off_map:
        #     inter_reward -= self.invalid_movement_penalty

        # update self.latest_reward and then return the intermediate reward
        # self.latest_reward = inter_reward
        return inter_reward


class ComprehensiveRewardV2(BaseReward):
    """TODO add description."""

    def __init__(
        self,
        *,
        harness_analytics: ReactiveHarnessAnalytics,
        fixed_reward: float,  # FIXME use better name
        static_penalty: float,  # FIXME use better name
        invalid_movement_penalty: float,
    ):
        """Induces a postive reward structure using the number of undamaged squares.

        The method used for reward calculation is intended to mirror Dhanuj's
        `complex_reward` from FY22, where the agent is rewarded using the difference in
        the number of `BurnStatus.UNBURNED` tiles between the main and benchmark
        simulations. This reward structure is intended to be task agnostic, and makes an
        effort to penalize the agent for unsafe proximity to the fire (`static_penalty`).

        Attributes:
            harness_analytics: The `ReactiveHarnessAnalytics` object that houses the data
                used to calculate the reward.
            fixed_reward: The fixed reward that is scaled by the number of squares saved
                by the agent.
            static_penalty: The fixed penalty that is applied to the agent if it is
                within a certain distance of the fire.
            invalid_movement_penalty: The fixed penalty that is applied to the agent if
                it attempts to move to a square that is not contained within the bounds
                of the `FireSimulation.fire_map`.
        """
        self.fixed_reward = fixed_reward
        # TODO: rename to `self.near_fire_penalty`
        self.static_penalty = static_penalty
        self.invalid_movement_penalty = invalid_movement_penalty
        super().__init__(harness_analytics)

    def get_reward(self, timestep: int, sim_run: bool) -> float:
        """Rewards the agent for `saving` squares from the fire wrt the benchmark sim.

        A constant scalar reward, `self.fixed_reward`, is scaled by the number of squares
        "saved" by the agent, which equates to the difference in the number of
        `BurnStatus.UNBURNED` tiles between the main `FireSimulation.fire_map` and the
        benchmark `FireSimulation.fire_map`.

        Arguments:
            timestep: The current timestep in the episode.
            sim_run: Whether or not the simulation was run this timestep.

        Returns:
            (Fixed reward * number of squares saved) - (penalty for being near fire)
        """
        if not sim_run:
            return self.get_timestep_intermediate_reward(timestep)

        # This Reward will compare the number of new recently damaged squares in the
        # main sim and within the bench sim

        # Get the number of undamaged squares in the main and benchmark simulations
        # FIXME: should we index with `-1` or `timestep - 1`?
        # NOTE: Convert to int to avoid "overflow in scalar subtract"(cols are
        # np.uint16!!)
        undamaged_mainsim = self.harness_analytics.sim_analytics.data.timestep_data[
            "unburned"
        ]
        undamaged_benchsim = self.harness_analytics.benchmark_sim_analytics.data.unburned
        # if the benchsim is no longer active, but the main simulation is still active,
        #   then it is okay to use the last value of the num_undamaged from the benchsim
        # as assumadley our main_sim agent will be rewarded for sustaining the fire longer

        # define the number of squares saved by the agent as the difference between the
        # benchsim and the mainsim
        timestep_number_squares_saved = undamaged_mainsim - undamaged_benchsim

        # If MAIN SIMULATION ENDS FASTER THAN BENCH SIMULATION
        #   there are either two possibilities
        #   1. The agent's actions sped up the fire
        #           - In which case the agent should be heavily penalized
        #   2. The agent's actions ended the fire faster and saved more squares (The most
        #           ideal Situation)
        #           - In which case the agent should be heavily rewarded
        if (
            self.harness_analytics.benchmark_sim_analytics.active is True
            and self.harness_analytics.sim_analytics.active is False
        ):
            # update the value of the undamaged benchsim to be that of the bench
            # simulation if it reached it's end
            undamaged_benchsim = self._sim_area - self.harness_analytics.bench_damage

            timestep_number_squares_saved = undamaged_mainsim - undamaged_benchsim

            # multiply this by the number of timesteps that the main_sim is faster than
            # the bench sim
            if (
                self.harness_analytics.bench_timesteps
                > self.harness_analytics.sim_analytics.num_sim_steps
            ):
                timestep_number_squares_saved = timestep_number_squares_saved * (
                    self.harness_analytics.bench_timesteps
                    - self.harness_analytics.sim_analytics.num_sim_steps
                )

        # this new reward works out well for both of the above cases
        #   For Case 1., this reward will yield a large negative reward
        #   For Case 2., this reward will yield a large positive reward
        reward = ((timestep_number_squares_saved) / self._sim_area) * self.fixed_reward

        # AUGMENT THE REWARD IF AGENT GETS TOO CLOSE TO THE FIRE
        # use static reward so RL easily learns what causes this reward
        # TODO: determine best amount for this reward
        if self.harness_analytics.sim_analytics.agent_analytics.data.near_fire:
            # set the reward to be -1.0 * static_penalty
            reward = -self.static_penalty

        # Penalize agent if chosen movement would result in an invalid map position.
        # if not self.harness_analytics.sim_analytics.agent_analytics.df.iloc[-1][
        # FIXME handle MARL case
        # if self.harness_analytics.sim_analytics.agent_analytics.data.valid_movement:
        #     reward -= self.invalid_movement_penalty

        # TODO add very large negative reward if agent steps into fire
        # (or end the simulation)

        # update self.latest_reward and then return the reward
        self.latest_reward = reward
        return reward

    def get_timestep_intermediate_reward(self, timestep: int) -> float:
        """TODO Add function docstring."""
        # start with the intermediate reward just being the same as the previously
        # calculated reward
        inter_reward = self.harness_analytics.latest_reward

        # add a slight reward to the agent for placing a mitigation not in a burned area
        # FIXME: should we index with `-1` or `timestep - 1`?
        # FIXME handle MARL case
        # if self.harness_analytics.sim_analytics.agent_analytics.data.near_fire:
        #     inter_reward += 1

        # Penalize agent if chosen movement would result in an invalid map position.
        # FIXME handle MARL case
        # if self.harness_analytics.sim_analytics.agent_analytics.data.valid_movement:
        #     inter_reward -= self.invalid_movement_penalty

        # update self.latest_reward and then return the intermediate reward
        # self.latest_reward = inter_reward
        return inter_reward
