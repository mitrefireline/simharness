"""FIXME: A one line summary of the module or program.

Leave one blank line.  The rest of this docstring should contain an
overall description of the module or program.  Optionally, it may also
contain a brief description of exported classes and functions and/or usage
examples.

Typical usage example:

  foo = ClassFoo()
  bar = foo.FunctionBar()
"""
import copy
from abc import ABC, abstractmethod
from collections import OrderedDict as ordered_dict
from enum import IntEnum
from typing import Any, Dict, List, Optional, OrderedDict, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from simfire.sim.simulation import Simulation


class RLHarness(gym.Env, ABC):
    """`Simulation` wrapper enabling RL agent's to interact with different simulators.

    The most important API methods a RLHarness exposes are `step()`, `reset()`,
    `render()`, and `close()`.

    Longer class information... FIXME.
    Longer class information... FIXME.

    Attributes:
        simulation: A subclass of `Simulation` that defines a given simulator.
        movements: A list containing the movements available to a given agent. For
          example, possible movements could be: ["up", "down", "left", "right"].
        interactions: A list containing the interactions available to a given agent.
          For example, if the simulation IS-A `FireSimulation`, possible interactions
          could be: ["fireline", "scratchline", "wetline"]. To learn more, see
          https://gitlab.mitre.org/fireline/simulators/simfire/-/blob/main/simfire/sim/simulation.py#L269-280
        attributes: (FIXME) A list containing the input features into the observations.
          Each feature is a channel in the input observation.
        normalized_attributes: A list containing attributes that need to be normalized.
          Any and all values within `normalized_attributes` must exist in `attributes`!
        deterministic: A boolean indicating whether the initial state of the environment
          is deterministic.
        sim_agent_id: FIXME.
        harness_to_sim: FIXME.
        sim_to_harness: FIXME.
        min_maxes: FIXME.
        low: FIXME.
        high: FIXME.
        observation_space: FIXME.
        action_space: FIXME.
        sim_attributes: FIXME.
        nonsim_attributes: FIXME.
    """

    def __init__(
        self,
        simulation: Simulation,
        movements: List[str],
        interactions: List[str],
        attributes: List[str],
        normalized_attributes: List[str],
        deterministic: bool = False,
    ) -> None:
        """Inits RLHarness with blah FIXME.

        Longer method information...
        Longer method information...

        Raises:
            AssertionError: FIXME.

        FIXME: Ideally, the docstr should encapsulate what is being initialized and any
        intuition behind design choices. This is relatively important since RLHarness
        serves as a base class that each environment will inherit from.
        """
        self.simulation = simulation
        self.movements = copy.deepcopy(movements)
        self.interactions = copy.deepcopy(interactions)
        self.attributes = attributes
        self.normalized_attributes = normalized_attributes
        self.deterministic = deterministic

        if not set(self.normalized_attributes).issubset(self.attributes):
            raise AssertionError(
                f"All normalized attributes ({str(self.normalized_attributes)}) must be "
                f"in attributes ({str(self.attributes)})!"
            )

        # Retrieve the observation space and action space for the simulation.
        sim_attributes = self.simulation.get_attribute_data()
        sim_actions = self.simulation.get_actions()

        # FIXME(afennelly) provide a better explanation (below) for sim_agent_id
        # Make ID of agent +1 of the max value returned by the simulation for a location
        # NOTE: Assume that every simulator will support 3 base scenarios:
        #  1. Untouched (Ex: simfire.enums.BurnStatus.UNBURNED)
        #  2. Currently Being Affected (Ex: simfire.enums.BurnStatus.BURNING)
        #  3. Affected (Ex: simfire.enums.BurnStatus.BURNED)
        self.sim_agent_id = 3 + len(self.interactions) + 1

        if not set(self.interactions).issubset(list(sim_actions.keys())):
            raise AssertionError(
                f"All interactions ({str(self.interactions)}) must be "
                f"in the simulator's actions ({str(list(sim_actions.keys()))})!"
            )

        # NOTE: In the RLHARNESS section in the config file (s) it says "NONE movement
        # and interaction is added by default at position 0 for both", which is referring
        # to the insertion below (for `self.movements` and `self.interactions`).
        # TODO(afennelly) add note in docs wrt the below insertion of "none".
        # NOTE: The insertion of "none" MUST happen AFTER the above usage check!
        self.movements.insert(0, "none")  # "don't move", "stay put", etc.
        self.interactions.insert(0, "none")  # "don't interact", "do nothing", etc.

        # FIXME review purpose of sim_nonsim conversions + add brief comment
        self._separate_sim_nonsim(sim_attributes)
        self.harness_to_sim, self.sim_to_harness = self._sim_harness_conv(sim_actions)
        self.min_maxes = self._get_min_maxes()

        channel_lows = np.array(
            [[[self.min_maxes[channel]["min"]]] for channel in self.attributes]
        )
        channel_highs = np.array(
            [[[self.min_maxes[channel]["max"]]] for channel in self.attributes]
        )

        self.low = np.repeat(
            np.repeat(channel_lows, self.simulation.config.area.screen_size, axis=1),
            self.simulation.config.area.screen_size,
            axis=2,
        )

        self.high = np.repeat(
            np.repeat(channel_highs, self.simulation.config.area.screen_size, axis=1),
            self.simulation.config.area.screen_size,
            axis=2,
        )

        self.observation_space = spaces.Box(
            np.float32(self.low),
            np.float32(self.high),
            shape=(
                len(self.attributes),
                self.simulation.config.area.screen_size,
                self.simulation.config.area.screen_size,
            ),
            dtype=np.float64,
        )

        action_shape = [len(self.movements), len(self.interactions)]
        self.action_space = spaces.MultiDiscrete(action_shape)

    @abstractmethod
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[Any, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[Any, Any]]:
        """Resets the environment to an initial state.

        This method generates a new starting state often with some randomness to ensure
        that the agent explores the state space and learns a generalized policy about the
        environment. This randomness can be controlled with the `seed` parameter.

        Subclasses, such as the ReactiveHarness, typically do the following within
        the overriden reset() method:
            1. set `self.num_burned = 0`.
            2. handle `self.deterministic`
            3. set `output = super().reset()`, which executes the below code and sets
               `output` (in child class reset()) to the return value, `self.state`.

        Arguments:
            seed: The (optional) int seed that is used to initialize the environment's
                PRNG (np_random). If the environment does not already have a PRNG and
                `seed=None` (the default option) is passed,

        Returns:
            An ndarray containing the initial state of the environment.
        """
        raise NotImplementedError

    @abstractmethod
    def step(
        self, action: Tuple[int, int]
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[Any, Any]]:
        """Run one timestep of the environment's dynamics.

        When end of episode is reached (`terminated or truncated` is True), you are
        responsible for calling `reset()` to reset the environment's state for the next
        episode.

        Arguments:
            action: An action provided by the agent to update the environment state.

        Returns:
            observation: A ndarray containing the observation of the environment.
            reward: A float representing the reward obtained as a result of taking the
                action.
            terminated: A boolean indicating whether the agent reaches the terminal state
            (as defined under the MDP of the task) which can be positive or negative.
            An example is reaching the goal state, or moving into the lava from the
            Sutton and Barton, Gridworld. If true, the user needs to call `reset()`.
            truncated: A boolean indicating whether the truncation condition outside
                the scope of the MDP is satisfied. Typically, this is a timelimit, but
                could also be used to indicate an agent physically going out of bounds.
                Can be used to end the episode prematurely before a terminal state is
                reached. If true, the user needs to call `reset()`.
            info: A dictionary containing auxiliary diagnostic information (helpful for
                debugging, learning, and logging). This might, for instance, contain:
                    - metrics that describe the agent's performance state
                    - variables that are hidden from observations, or
                    - individual reward terms that are combined to produce the total
                      reward.
        """
        raise NotImplementedError

    @abstractmethod
    def render(self) -> None:
        """Render a visualization of the environment."""
        raise NotImplementedError

    @abstractmethod
    def get_nonsim_attribute_data(self) -> OrderedDict[str, np.ndarray]:
        """Get data that does not come from the simulation."""
        raise NotImplementedError

    @abstractmethod
    def get_nonsim_attribute_bounds(self) -> OrderedDict[str, Dict[str, int]]:
        """Get bounds for data that does not come from the simulation."""
        raise NotImplementedError

    def _get_status_categories(self, disaster_categories: List[str]) -> List[str]:
        """Get disaster categories that aren't interactions.

        Arguments:
            disaster_categories (List[str]): List of potential Simulation space categories

        Returns:
            A list containing disaster categories (str), with interactions removed.
        """
        categories = []
        for cat in disaster_categories:
            if cat not in self.interactions:
                categories.append(cat)
        return categories

    def _separate_sim_nonsim(self, sim_attributes: OrderedDict[str, np.ndarray]) -> None:
        """Separate attributes based on if they are supported by the Simulation or not.

        Arguments:
            sim_attributes: An ordered dictionary linking all attributes of
                the Simulation to their respective data within the Sim.
        """
        self.sim_attributes = []
        self.nonsim_attributes = []
        for attribute in self.attributes:
            if attribute not in sim_attributes.keys():
                self.nonsim_attributes.append(attribute)
            else:
                self.sim_attributes.append(attribute)

    def _sim_harness_conv(
        self, sim_actions: Dict[str, IntEnum]
    ) -> Tuple[OrderedDict[int, int], OrderedDict[int, int]]:
        """Create conversion dictionaries for action (Sim) <-> interaction (Harness).

        Arguments:
            sim_actions: A dictionary mapping the action/mitigation strategies available
            to the corresponding `Enum` value within the simulation. FIXME update wording

        Returns:
            A tuple containing two ordered dictionaries for attribute conversion. The
            first will map interaction to action. and the second will map action to
            interaction.
        """
        # NOTE: hts == "harness_to_sim" and sth == "sim_to_harness"
        hts_action_conv = ordered_dict()
        sth_action_conv = ordered_dict()

        # NOTE: We define self.interactions[0] as "none" ("don't interact", "do nothing")
        hts_action_conv[0] = 0
        sth_action_conv[0] = 0

        if len(self.interactions) == 0:
            pass
        else:
            # NOTE: omit first interaction since it is handled above
            for idx, action in enumerate(self.interactions[1:], start=1):
                hts_action_conv[idx] = sim_actions[action].value
                sth_action_conv[sim_actions[action].value] = idx

        return hts_action_conv, sth_action_conv

    def _select_from_dict(
        self, dictionary: OrderedDict[str, Any], selections: List[str]
    ) -> OrderedDict[str, Any]:
        """Create an ordered subset with only specific keys from the input `dictionary`.

        Arguments:
            dictionary: A dictionary used to extract values from.
            selections: A list containing the desired keys to keep from `dictionary`.

        Returns:
            An ordered dictionary containing a subset of the input `dictionary`.
        """
        return_dict = OrderedDict()

        for selection in selections:
            return_dict[selection] = dictionary[selection]

        return return_dict

    def _get_min_maxes(self) -> OrderedDict[str, Dict[str, Tuple[int, int]]]:
        """Retrieves the minimum and maximum for all relevant attributes."""
        # TODO update docstring to be more specific
        # TODO add comments and refactor as needed
        sim_min_maxes = ordered_dict()
        # fetch the observation space bounds for the simulation.
        sim_bounds = self.simulation.get_attribute_bounds()
        for attribute in self.sim_attributes:
            sim_min_maxes[attribute] = sim_bounds[attribute]

        nonsim_min_maxes = self._select_from_dict(
            self.get_nonsim_attribute_bounds(), self.nonsim_attributes
        )

        if len(nonsim_min_maxes) != len(self.nonsim_attributes):
            raise AssertionError(
                f"Min-Maxes for {len(nonsim_min_maxes)} nonsim attributes were given but "
                f"there are {len(self.nonsim_attributes)} nonsim attributes."
            )

        min_maxes = ordered_dict({**sim_min_maxes, **nonsim_min_maxes})

        return min_maxes

    def _normalize_obs(
        self, observations: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """Convert an observation to the [0,1] range based on known min and max."""

        def normalize(data, min_max):
            # FIXME: Explain purpose/intention behind using a nested class here.
            return (data - min_max["min"]) / (min_max["max"] - min_max["min"])

        for attribute in self.normalized_attributes:
            observations[attribute] = normalize(
                observations[attribute], self.min_maxes[attribute]
            )

        return observations
