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
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    OrderedDict,
    Tuple,
    Union,
    no_type_check,
)

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from simfire.sim.simulation import FireSimulation


class RLHarness(gym.Env, ABC):
    """`Simulation` wrapper enabling RL agent's to interact with different simulators.

    The most important API methods a RLHarness exposes are `step()`, `reset()`,
    `render()`, and `close()`.

    Longer class information... FIXME.
    Longer class information... FIXME.

    Attributes:
        sim: A subclass of `Simulation` that defines a given simulator.
        movements: A list containing the movements available to a given agent. For
          example, possible movements could be: ["up", "down", "left", "right"].
        interactions: A list containing the interactions available to a given agent.
          For example, if the sim IS-A `FireSimulation`, possible interactions
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
        sim: FireSimulation,
        movements: List[str],
        interactions: List[str],
        attributes: List[str],
        normalized_attributes: List[str],
        action_space_cls: Callable,
        deterministic: bool = False,
        benchmark_sim: FireSimulation = None,
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
        # NOTE: The caller is responsible for creating the `FireSimulation` object (s),
        # and if a `benchmark_sim` is provided, it should be a separate object, identical
        # to `sim` (after initialization), but will not receive any mitigations.
        self.sim = sim
        self.benchmark_sim = benchmark_sim
        # Indicates (internally) whether a benchmark simulation should be used
        # FIXME: I'm not sure if we need `_use_benchmark_sim`, since we can just check
        # if `benchmark_sim` is None or not.
        # self._use_benchmark_sim = True if benchmark_sim else False
        # TODO Create self._time_arg_passed_to_sim_run and set default value to 1. This
        # would allow the simulation to be run for an arbitrary number of timesteps.

        # TODO: use more apt name, ex: `available_movements`, `possible_movements`.
        self.movements = copy.deepcopy(movements)
        # TODO: use more apt name, ex: `available_interactions`, `possible_interactions`.
        self.interactions = copy.deepcopy(interactions)
        self.attributes = attributes
        # TODO: Maybe use `attributes_to_normalize` over `normalized_attributes`?
        self.normalized_attributes = normalized_attributes
        # FIXME: remove `deterministic` from the constructor; externally randomize env.
        self.deterministic = deterministic

        if not set(self.normalized_attributes).issubset(self.attributes):
            raise AssertionError(
                f"All normalized attributes ({str(self.normalized_attributes)}) must be "
                f"in attributes ({str(self.attributes)})!"
            )

        # Retrieve the observation space and action space for the simulation.
        sim_attributes = self.sim.get_attribute_data()
        sim_actions = self.sim.get_actions()

        # FIXME(afennelly) provide a better explanation (below) for sim_agent_id
        # Make ID of agent +1 of the max value returned by the simulation for a location
        # NOTE: Assume that every simulator will support 3 base scenarios:
        #  1. Untouched (Ex: simfire.enums.BurnStatus.UNBURNED)
        #  2. Currently Being Affected (Ex: simfire.enums.BurnStatus.BURNING)
        #  3. Affected (Ex: simfire.enums.BurnStatus.BURNED)
        self.sim_agent_id = 3 + len(self.interactions) + 1

        # Before verifying that all interactions are supported by the simulator, we need
        # to remove the "none" interaction (if it exists).
        if "none" in self.interactions:
            none_idx = self.interactions.index("none")
            interaction_types = (
                self.interactions[:none_idx] + self.interactions[none_idx + 1 :]
            )
        else:
            interaction_types = self.interactions

        if not set(interaction_types).issubset(list(sim_actions.keys())):
            raise AssertionError(
                f"All interactions ({str(interaction_types)}) must be "
                f"in the simulator's actions ({str(list(sim_actions.keys()))})!"
            )

        # FIXME review purpose of sim_nonsim conversions + add brief comment
        self._separate_sim_nonsim(sim_attributes)
        # NOTE: `self.harness_to_sim` used in `ReactiveHarness._update_mitigation()`.
        # FIXME `self.sim_to_harness` is NOT used anywhere else.
        self.harness_to_sim, self.sim_to_harness = self._sim_harness_conv(sim_actions)
        self.min_maxes = self._get_min_maxes()

        # NOTE: calling `reshape()` to switch to channel-minor format.
        channel_lows = np.array(
            [[[self.min_maxes[channel]["min"]]] for channel in self.attributes]
        ).reshape(1, 1, len(self.attributes))
        channel_highs = np.array(
            [[[self.min_maxes[channel]["max"]]] for channel in self.attributes]
        ).reshape(1, 1, len(self.attributes))

        self.low = np.repeat(
            np.repeat(channel_lows, self.sim.config.area.screen_size, axis=1),
            self.sim.config.area.screen_size,
            axis=0,
        )
        self.high = np.repeat(
            np.repeat(channel_highs, self.sim.config.area.screen_size, axis=1),
            self.sim.config.area.screen_size,
            axis=0,
        )

        # NOTE: Should we pass `seed` to seed the RNG used to sample from the space?
        self.observation_space = spaces.Box(
            self.low,
            self.high,
            dtype=np.float32,
        )

        action_shape = self._get_action_space_shape(space_type=action_space_cls)
        self.action_space = action_space_cls(action_shape)

    @no_type_check
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
        super().reset(seed=seed)

    @abstractmethod
    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
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

        actions = self.interactions
        if len(actions) > 0:
            # Using the "valid" interaction_types, populate the conversion dicts.
            valid_idxs = [actions.index(act) for act in actions if act != "none"]
            
            for idx in valid_idxs:
                interaction = self.interactions[idx]
                hts_action_conv[idx] = sim_actions[interaction].value
                sth_action_conv[sim_actions[interaction].value] = idx

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
        # FIXME (afennelly) I think the return type should be:
        #   - OrderedDict[str, Dict[str, object]]
        # TODO update docstring to be more specific
        # TODO add comments and refactor as needed
        sim_min_maxes = ordered_dict()
        # fetch the observation space bounds for the simulation.
        sim_bounds = self.sim.get_attribute_bounds()
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

    def _get_action_space_shape(
        self, space_type: spaces.Space
    ) -> Union[int, np.ndarray, List]:
        """Get the shape of the action space, dependent on the action space type.

        Args:
            space_type (spaces.Space): [description]

        Raises:
            NotImplementedError: [description]

        Returns:
            Union[int, np.ndarray, List]: [description]
        """
        if space_type is spaces.Discrete:
            return len(self.movements) * len(self.interactions)
        elif space_type is spaces.MultiDiscrete:
            return [len(self.movements), len(self.interactions)]
        else:
            # TODO provide a descriptive error message.
            raise NotImplementedError
