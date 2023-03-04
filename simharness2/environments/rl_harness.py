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
from typing import Any, Dict, List, OrderedDict, Tuple, Union

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

        # Set the value of the agent within the observations
        sim_attributes = self.simulation.get_attribute_data()
        sim_actions = self.simulation.get_actions()

        # FIXME(afennelly) provide a better explanation (below) for sim_agent_id
        # Make ID of agent +1 of the max value returned by the simulation for a location
        # NOTE: Assume that every simulator will support 3 base scenarios:
        #  1. Untouched
        #  2. Currently Being Affected
        #  3. Affected
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

    def _get_status_categories(self, disaster_categories: List[str]) -> List[str]:
        """Get disaster categories that aren't interactions.

        Args:
            disaster_categories (List[str]): List of potential Simulation space categories

        Returns:
            A list containing disaster categories (str), with interactions removed.
        """
        categories = []
        for cat in disaster_categories:
            if cat not in self.interactions:
                categories.append(cat)
        return categories

    # -----------------------------------------------------------------------------------
    def _separate_sim_nonsim(self, sim_attributes: OrderedDict[str, np.ndarray]) -> None:
        """Separate attributes based on if they are supported by the Simulation or not.

        Args:
            sim_attributes(OrderedDict[str, np.ndarray]): Dict linking all attributes of
                the Simulation to their respective data within the Sim.

        Returns:
            None
        """
        self.sim_attributes = []
        self.nonsim_attributes = []
        for attribute in self.attributes:
            if attribute not in sim_attributes.keys():
                self.nonsim_attributes.append(attribute)
            else:
                self.sim_attributes.append(attribute)

    # -----------------------------------------------------------------------------------

    def _sim_harness_conv(
        self, sim_actions
    ) -> Tuple[OrderedDict[int, int], OrderedDict[int, int]]:
        """Create conversion dictionaries for action (Sim) <-> interaction (Harness).

        Args:
            sim_actions (List[str]): List of supported actions within the Sim.

        Returns:
            Tuple[OrderedDict[int, int], OrderedDict[int, int]]: Conversion dictionary
                for Sim->Harness and Harness->Sim
        """
        hts_action_conv = ordered_dict()
        sth_action_conv = ordered_dict()

        hts_action_conv[0] = 0
        sth_action_conv[0] = 0

        if len(self.interactions) == 0:
            pass
        else:
            for e, action in enumerate(self.interactions[1:]):
                hts_action_conv[e + 1] = sim_actions[action].value
                sth_action_conv[sim_actions[action].value] = e + 1

        return hts_action_conv, sth_action_conv

    # -----------------------------------------------------------------------------------

    def _select_from_dict(
        self, dictionary: OrderedDict[str, Any], selections: List[str]
    ) -> OrderedDict[str, Any]:
        """Create an ordered subset with only specific keys from the input `dictionary`.

        Args:
            dictionary (OrderedDict[str, Any]): Dictionary to pull from.
            selections (List[str]): Keys to keep from dictionary.

        Returns:
            OrderedDict[str, Any]: Ordered subset from given dictionary
        """
        return_dict = OrderedDict()

        for selection in selections:
            return_dict[selection] = dictionary[selection]

        return return_dict

    # -----------------------------------------------------------------------------------

    def _get_min_maxes(self) -> OrderedDict[str, Dict[str, Tuple[int, int]]]:
        """Retrieves the minimum and maximum for all relevant attributes.

        Returns:
            OrderedDict[str, Dict[str, Tuple[int, int]]]: Min and max values for relevant
                attributes
        """
        sim_min_maxes = ordered_dict()
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

    # -----------------------------------------------------------------------------------

    def _normalize_obs(
        self, observations: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """Convert an observation to the [0,1] range based on known min and max."""

        def normalize(data, min_max):
            return (data - min_max["min"]) / (min_max["max"] - min_max["min"])

        for attribute in self.normalized_attributes:
            observations[attribute] = normalize(
                observations[attribute], self.min_maxes[attribute]
            )

        return observations

    # -----------------------------------------------------------------------------------

    def reset(self) -> np.ndarray:
        """Reset environment to initial state.

        If the environment is not deterministic, the fire initial position and elevation
        are randomized.

        Returns:
            np.ndarray: Initial observation
        """
        if not self.deterministic:
            # Set seeds for randomization
            fire_init_seed = self.simulation.get_seeds()["fire_initial_position"]
            elevation_seed = self.simulation.get_seeds()["elevation"]
            seed_dict = {
                "fire_initial_position": fire_init_seed + 1,
                "elevation": elevation_seed + 1,
            }
            self.simulation.set_seeds(seed_dict)

        self.simulation.reset()
        sim_observations = self._select_from_dict(
            self.simulation.get_attribute_data(), self.sim_attributes
        )
        nonsim_observations = self._select_from_dict(
            self.get_nonsim_attribute_data(), self.nonsim_attributes
        )

        if len(nonsim_observations) != len(self.nonsim_attributes):
            raise AssertionError(
                f"Data for {len(nonsim_observations)} nonsim attributes were given but "
                f"there are {len(self.nonsim_attributes)} nonsim attributes."
            )

        observations = self._normalize_obs({**sim_observations, **nonsim_observations})

        obs = []
        for attribute in self.attributes:
            obs.append(observations[attribute])

        self.state = np.stack(obs, axis=0).astype(np.float64)

        return self.state

    @abstractmethod
    def step(
        self, action: Union[int, Tuple[int]]
    ) -> Tuple[np.ndarray, float, bool, Dict[Any, Any]]:
        """Run one timestep of the environment's dynamics.

        When end of episode is reached, you are responsible for calling :meth:`reset` to
        reset this environment's state. Accepts an action and returns either a tuple
        `(observation, reward, terminated, truncated, info)`.

        Args:
            action (Union[int, Tuple[int]]): an action provided by the agent
        Returns:
            observation (np.ndarray): this will be an element of the environment's
                :attr:`observation_space`.
            reward (float): The amount of reward returned as a result of taking the
                action.
            terminated (bool): whether a `terminal state` (as defined under the MDP of
                the task) is reached. In this case further step() calls could return
                undefined results.
            info (Dict[Any, Any]): `info` contains auxiliary diagnostic information
                (helpful for debugging, learning, and logging). This might, for instance,
                contain: metrics that describe the agent's performance state, variables
                that are hidden from observations, or individual reward terms that are
                combined to produce the total reward. It also can contain information that
                distinguishes truncation and termination, however this is deprecated in
                favour of returning two booleans, and will be removed in a future version.
        """
        pass

    @abstractmethod
    def render(self) -> None:
        """Render a visualization of the environment."""
        pass

    def get_nonsim_attribute_data(self) -> OrderedDict[str, np.ndarray]:
        """Get data that does not come from the simulation."""
        return ordered_dict()

    def get_nonsim_attribute_bounds(self) -> OrderedDict[str, Dict[str, int]]:
        """Get bounds for data that does not come from the simulation."""
        return ordered_dict()
