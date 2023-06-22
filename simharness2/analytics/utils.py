from dataclasses import dataclass


@dataclass
class BestEpisodePerformance:
    """Stores best performance (wrt reactive fire scenario) across all episodes in trial.

    Attributes:
        max_unburned: An int storing the maximum number of tiles in the main
            `FireSimulation.fire_map` that are `BurnStatus.UNBURNED` across all episodes
            in a trial. At the end of each episode, this value is updated if the number
            of `BurnStatus.UNBURNED` tiles is greater than the current value.
        max_unburned_rescaled: A float storing a rescaled value of `max_unburned`, where min-max
            normalization is used to perform the rescaling. The rescaled value will fall
            between 0 and 1 (inclusive). Intuitively, this value represents the
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
        self.max_unburned_rescaled = self.max_unburned / self.sim_area
