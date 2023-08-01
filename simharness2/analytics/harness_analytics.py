"""Base AnalyticsTracker for SimHarness and BaseReward."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import partial
from typing import List, Optional

from simfire.sim.simulation import FireSimulation

from simharness2.analytics.simulation_analytics import FireSimulationAnalytics


class RLHarnessAnalytics(ABC):
    """Base class with several built in methods."""

    def __init__(
        self,
        *,
        sim: FireSimulation,
        sim_analytics_partial: partial,
        benchmark_sim: FireSimulation = None,
    ) -> None:
        """TODO: Add docstring."""
        # Store objects used to track simulation data within each episode in a run.
        try:
            self.sim_analytics: FireSimulationAnalytics = sim_analytics_partial(sim=sim)
            if benchmark_sim:
                self.benchmark_sim_analytics: FireSimulationAnalytics = (
                    sim_analytics_partial(sim=benchmark_sim, is_benchmark=True)
                )
            else:
                self.benchmark_sim_analytics
        except TypeError as e:
            raise e

        self.best_episode_performance: Optional[BestEpisodePerformance] = None

    @abstractmethod
    def update_after_one_simulation_step(self, *, timestep: int) -> None:
        """See subclass for docstring."""
        pass

    @abstractmethod
    def update_after_one_agent_step(
        self,
        *,
        timestep: int,
        movement: int,
        interaction: int,
        agent_pos: List[int],
    ) -> None:
        """See subclass for docstring."""
        pass

    @abstractmethod
    def update_after_one_harness_step(
        self, sim_run: bool, terminated: bool, reward: float, *, timestep: int
    ) -> None:
        """See subclass for docstring."""
        pass

    @abstractmethod
    def reset(self):
        """See subclass for docstring."""
        pass


class ReactiveHarnessAnalytics(RLHarnessAnalytics):
    """TODO Add description."""

    def __init__(
        self,
        *,
        sim: FireSimulation,
        sim_analytics_partial: partial,
        benchmark_sim: FireSimulation = None,
    ) -> None:
        """TODO Add summary line.

        Arguments:
            sim: The underlying `FireSimulation` object that contains the agent (s) that
                are being trained. The agent (s) will place mitigation lines, and the
                simulation will spread the fire. An episode terminates when the fire is
                finished spreading.
            sim_analytics_partial: A `functools.partial` object that defines the class
                that willbbe used to monitor and track `self.sim`, and
                `self.benchmark_sim`, if the optional `benchmark_sim` is provided. The
                user is expected to provide the `agent_analytics_partial` keyword
                argument, along with a valid value.
            benchmark_sim: A separate `FireSimulation` object, identical to
                `sim` (after initialization). No mitigation lines will be placed in this
                simulation, as it does not contain any agent (s).

        Raises:
            TypeError: If `sim_analytics_partial.keywords` does not contain a
            `agent_analytics_partial` key with value of type `functools.partial`.

        """
        super().__init__(
            sim=sim,
            sim_analytics_partial=sim_analytics_partial,
            benchmark_sim=benchmark_sim,
        )
        # Define attributes that are needed/accessed within `ComprehensiveReward` class.
        # TODO: Address where these attributes should be stored, see
        # https://gitlab.mitre.org/fireline/reinforcementlearning/simharness2/-/merge_requests/6#note_1504742
        if self.benchmark_sim_analytics:
            self.bench_timesteps: int = 0
            self.bench_damage: int = 0
            self.bench_estimated: bool = False

        # Track the latest episode reward
        # TODO is this the reward for the latest timestep or the latest episode?
        # FIXME: Decide how and where this attribute is/should be used.
        self.latest_reward = 0.0

        self.episodes_total = 0

    def update_after_one_agent_step(
        self,
        *,
        timestep: int,
        movement: int,
        interaction: int,
        agent_pos: List[int],
    ) -> None:
        """Updates `self.sim_analytics.agent_analytics`, if agents are in the sim.

        This method is intended to be called directly after the call to
        `ReactiveHarness._do_one_agent_step()` (within `ReactiveHarness.step()`).

        Arguments:
            timestep: An integer indicating the current timestep of the episode.
            movement: An integer indicating the index of the latest movement that the
                agent selected.
            interaction: An integer indicating the index of the latest interaction that
                the agent selected.
            agent_pos: A list of integers indicating the current position of the agent.
        """
        if self.sim_analytics.agent_analytics:
            self.sim_analytics.agent_analytics.update(
                timestep, movement, interaction, agent_pos
            )

    def update_after_one_simulation_step(self, *, timestep: int) -> None:
        """Updates `self.sim_analytics` (and `self.benchmark_sim_analytics`, if exists).

        This method is intended to be called directly after the call to
        `ReactiveHarness._do_one_simulation_step()` (within `ReactiveHarness.step()`).

        Arguments:
            timestep: An integer indicating the current timestep of the episode.
        """
        self.sim_analytics.update(timestep)

        if self.benchmark_sim_analytics:
            self.benchmark_sim_analytics.update(timestep)

        sim_area = self.sim_analytics.sim.fire_map.size
        # FIXME mention in docstring that this logic is performed. need to condense!!
        benchsim_active = self.benchmark_sim_analytics.active
        benchsim_undamaged = self.benchmark_sim_analytics.df.iloc[-1]["unburned_total"]
        # Use this to update the self.bench_timesteps and the self.bench_damage
        if benchsim_active is False and self.bench_estimated is False:
            # if the benchsim has reached it's end, then use this to set the values of
            # the variables
            self.bench_timesteps = self.benchmark_sim_analytics.num_sim_steps
            self.bench_damage = sim_area - benchsim_undamaged
            self.bench_estimated = True

        # use this to initialize the self.bench_timesteps and the self.bench_damage if
        # the bench_sim has not ended before the main_sim yet
        # TODO make this more efficient or just have the benchsim run once before the
        # agent makes any actions
        elif self.bench_estimated is False:
            if self.benchmark_sim_analytics.num_sim_steps > self.bench_timesteps:
                self.bench_timesteps = self.benchmark_sim_analytics.num_sim_steps + 1

            if sim_area - benchsim_undamaged > self.bench_damage:
                self.bench_damage = sim_area - benchsim_undamaged + 1

    def update_after_one_harness_step(
        self, sim_run: bool, terminated: bool, reward: float, *, timestep: int
    ) -> None:
        """Update the analytics after one step in the harness.

        Args:
            sim_run (bool): [description]
            terminated (bool): [description]
            reward (float): [description]
            timestep (int): [description]
        """
        # Reset any attributes that monitor agent behavior between each simulation step.
        if sim_run and self.sim_analytics.agent_analytics:
            self.sim_analytics.agent_analytics.reset_after_one_simulation_step()

        # Once episode has terminated, check if episode performance is the best so far.
        if terminated:
            self.episodes_total += 1
            # log.info(f"Episode {self.episodes_total} has terminated.")

            sim_df = self.sim_analytics.df
            current_unburned = (
                0 if len(sim_df) == 0 else sim_df.iloc[-1]["unburned_total"]
            )
            update_best_episode_performance = True
            if self.best_episode_performance:
                max_unburned = self.best_episode_performance.max_unburned
                if current_unburned <= max_unburned:
                    update_best_episode_performance = False

            if update_best_episode_performance:
                self.best_episode_performance = BestEpisodePerformance(
                    max_unburned=current_unburned,
                    sim_area=self.sim_analytics.sim.fire_map.size,
                    num_sim_steps=self.sim_analytics.num_sim_steps,
                    episode=self.episodes_total,
                    reward=reward,
                )

    def reset(self):
        """Resets attributes that track data within each episode.

        This method is intended to be called within after the call to
        `ReactiveHarness._do_one_agent_step()` (within `ReactiveHarness.step()`).

        """
        self.sim_analytics.reset()

        if self.benchmark_sim_analytics:
            self.benchmark_sim_analytics.reset()


@dataclass
class BestEpisodePerformance:
    """Stores best performance (wrt reactive fire scenario) across all episodes in trial.

    Attributes:
        max_unburned: An int storing the maximum number of tiles in the main
            `FireSimulation.fire_map` that are `BurnStatus.UNBURNED` across all episodes
            in a trial. At the end of each episode, this value is updated if the number
            of `BurnStatus.UNBURNED` tiles is greater than the current value.
        max_unburned_rescaled: A float storing a rescaled value of `max_unburned`, where
            min-max normalization is used to perform the rescaling. The rescaled value
            will fall between 0 and 1 (inclusive). Intuitively, this value represents the
            maximum proportion of "land" that was "saved" by the agent's actions.
        num_sim_steps: An int storing the number of simulation steps that occurred in the
            episode with the maximum number of `BurnStatus.UNBURNED` tiles.
        episode: An int storing the episode number that corresponds to the best episode
            performance.
        reward: A float storing the cumulative reward achieved after the "best" episode.
    """

    max_unburned: int
    sim_area: int
    num_sim_steps: int
    episode: int
    reward: float

    def __post_init__(self):
        """TODO add description."""
        self.max_unburned_rescaled = self.max_unburned / self.sim_area
