"""Base AnalyticsTracker for SimHarness and BaseReward."""
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import partial
from typing import Any, Optional, Dict

from simfire.sim.simulation import FireSimulation

from simharness2.analytics.simulation_analytics import FireSimulationAnalytics
from simharness2.agents import ReactiveAgent

logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
handler.setFormatter(
    logging.Formatter("%(asctime)s\t%(levelname)s %(filename)s:%(lineno)s -- %(message)s")
)
logger.addHandler(handler)
logger.propagate = False

logger = logging.getLogger("ray.rllib")


class RLHarnessAnalytics(ABC):
    """Base class with several built in methods."""

    def __init__(
        self,
        *,
        sim: FireSimulation,
        sim_analytics_partial: partial,
        # use_benchmark_sim: bool = False
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
                self.benchmark_sim_analytics: FireSimulationAnalytics = None
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
        agent_ids: set,
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
            agent_ids: TODO
            benchmark_sim: A separate `FireSimulation` object, identical to
                `sim` (after initialization). No mitigation lines will be placed in this
                simulation, as it does not contain any agent (s).

        Raises:
            TypeError: If `sim_analytics_partial.keywords` does not contain a
            `agent_analytics_partial` key with value of type `functools.partial`.

        """
        # NOTE: Below is a hacky way to specify agent ids; Fix later
        # Inject `agent_ids` into keywords of `agent_analytics_partial`
        agent_partial: partial = sim_analytics_partial.keywords["agent_analytics_partial"]
        agent_partial.keywords.update({"agent_ids": agent_ids})
        sim_analytics_partial.keywords["agent_analytics_partial"] = agent_partial
        # Initialize sim_analytics object (s) and best_episode_performance attribute.
        super().__init__(
            sim=sim,
            sim_analytics_partial=sim_analytics_partial,
            benchmark_sim=benchmark_sim,
        )

       
        # Define attributes that are needed/accessed within `ComprehensiveReward` class.
        # TODO: Address where these attributes should be stored, see
        # https://gitlab.mitre.org/fireline/reinforcementlearning/simharness2/-/merge_requests/6#note_1504742

        if self.benchmark_sim_analytics:
            #track the existence of the benchmark sim to generate the comparative (ex. area saved or burn rate reduction) metrics
            self.sim_analytics.benchmark_exists = True

        # Track the latest episode reward
        # TODO is this the reward for the latest timestep or the latest episode?
        # FIXME: Decide how and where this attribute is/should be used.
        self.latest_reward = 0.0

        self.episodes_total = 0

    def update_after_one_agent_step(
        self,
        *,
        timestep: int,
        agents: Dict[Any, ReactiveAgent],
    ) -> None:
        """Updates `self.sim_analytics.agent_analytics`, if agents are in the sim.

        This method is intended to be called directly after the call to
        `ReactiveHarness._do_one_agent_step()` (within `ReactiveHarness.step()`).

        Arguments:
            sim: The underlying `FireSimulation` object that contains the agent (s) that
                are being trained. The agent (s) will place mitigation lines, and the
                simulation will spread the fire. An episode terminates when the fire is
                finished spreading. (FIXME later)
            timestep: An integer indicating the current timestep of the episode.
            agents: TODO
        """
        if self.sim_analytics.agent_analytics:
            self.sim_analytics.agent_analytics.update(timestep, agents)

    def update_after_one_simulation_step(self, *, timestep: int) -> None:
        """Updates `self.sim_analytics` (and `self.benchmark_sim_analytics`, if exists).

        This method is intended to be called directly after the call to
        `ReactiveHarness._do_one_simulation_step()` (within `ReactiveHarness.step()`).

        Arguments:
            timestep: An integer indicating the current timestep of the episode.
        """
        sim_area = self.sim_analytics.sim.fire_map.size
        
        if self.benchmark_exists:
            #update the sim metrics with comparison metrics that use the benchmark sim in sim_analytics
            self.sim_analytics.update_sim_bench_comparison_metrics(timestep, benchmark_data = self.benchmark_sim_analytics.data.damaged)
        else:
            self.sim_analytics.update(timestep)
        
        return

    def update_bench_after_one_simulation_step(self, *, timestep: int) -> None:
        """Updates `self.benchmark_sim_analytics`, if exists.

        This method is intended to be called at the beginning of each episode in
        ReactiveHarness.

        Arguments:
            timestep: An integer indicating the current timestep of the episode.
        """
        
        if self.benchmark_sim_analytics:
            self.benchmark_sim_analytics.update(timestep)

        return

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

            current_unburned = self.sim_analytics.data.unburned
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
            perf = self.best_episode_performance
            logger.info(f"Episode {self.episodes_total}: {perf}")

    def reset(self, env_is_rendering: bool = False):
        """Resets attributes that track data within each episode.

        This method is intended to be called within after the call to
        `ReactiveHarness._do_one_agent_step()` (within `ReactiveHarness.step()`).

        """

        self.sim_analytics.reset(env_is_rendering)
        if self.benchmark_exists: bool = True
            self.sim_analytics.benchmark_exists = True

        if self.benchmark_sim_analytics:
            self.benchmark_sim_analytics.reset(env_is_rendering)

    def save_sim_history(self, logdir: str, total_iters: int) -> None:
        """TODO Add docstring."""
        self.sim_analytics.data.save_episode_history(logdir, total_iters)

        if self.benchmark_sim_analytics:
            self.benchmark_sim_analytics.data.save_episode_history(logdir, total_iters)

    # def log_dfs(self):
    #     """Log the dataframes that are being tracked by the analytics."""
    #     logger.info("sim_analytics.df")
    #     logger.info(self.sim_analytics.df.to_markdown())
    #     if self.benchmark_sim_analytics:
    #         logger.info("benchmark_sim_analytics.df")
    #         logger.info(self.benchmark_sim_analytics.df.to_markdown())

    #     if self.sim_analytics.agent_analytics:
    #         logger.info("sim_analytics.agent_analytics.df")
    #         logger.info(self.sim_analytics.agent_analytics.df.to_markdown())


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
