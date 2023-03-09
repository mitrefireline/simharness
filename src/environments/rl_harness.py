from abc import ABC, abstractmethod
from collections import OrderedDict as ordered_dict
from typing import Any, Dict, List, OrderedDict, Tuple

import gym
import numpy as np
from simfire.sim.simulation import Simulation


class RLHarness(gym.Env, ABC):
    def __init__(
        self,
        simulation: Simulation,
        actions: List[str],
        attributes: List[str],
        normalized_attributes: List[str],
    ) -> None:
        self.simulation = simulation
        self.actions = actions
        self.attributes = attributes
        self.normalized_attributes = normalized_attributes

        # ------------------

        # Removed assert statement because of bandit report:
        # Issue: [B101:assert_used] Use of assert detected. The enclosed code will be
        # removed when compiling to optimised byte code.
        #    Severity: Low   Confidence: High
        #    CWE: CWE-703 (https://cwe.mitre.org/data/definitions/703.html)

        if not set(self.normalized_attributes).issubset(self.attributes):
            raise AssertionError(
                f"An attribute in  {str(self.normalized_attributes)} is not in "
                f"{str(self.attributes)}!"
            )

        # ------------------

        sim_attributes = self.simulation.get_attribute_data()
        sim_actions = self.simulation.get_actions()

        self._separate_sim_nonsim(sim_attributes, sim_actions)
        self.hts_action_conv, self.sth_action_conv = self._convert_actions(
            self.actions, sim_actions
        )
        self.min_maxes = self._get_min_maxes()

        # ------------------

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

        self.observation_space = gym.spaces.Box(
            np.float32(self.low),
            np.float32(self.high),
            shape=(
                len(self.attributes),
                self.simulation.config.area.screen_size,
                self.simulation.config.area.screen_size,
            ),
            dtype=np.float64,
        )

        self.action_space = gym.spaces.Discrete(len(self.actions))

    # -----------------------------------------------------------------------------------
    def _separate_sim_nonsim(self, sim_attributes, sim_actions) -> None:
        self.sim_attributes = []
        self.nonsim_attributes = []
        for attribute in self.attributes:
            if attribute not in sim_attributes.keys():
                self.nonsim_attributes.append(attribute)
            else:
                self.sim_attributes.append(attribute)

        self.sim_actions = []
        self.nonsim_actions = []
        for action in self.actions:
            if action not in sim_actions:
                self.nonsim_actions.append(action)
            else:
                self.sim_actions.append(action)

    # -----------------------------------------------------------------------------------

    def _convert_actions(
        self, harness_actions, sim_actions
    ) -> Tuple[OrderedDict[int, int], OrderedDict[int, int]]:
        hts_action_conv = ordered_dict()
        sth_action_conv = ordered_dict()
        for e, action in enumerate(harness_actions):
            if action in self.sim_actions:
                hts_action_conv[e] = sim_actions[action].value
                sth_action_conv[sim_actions[action].value] = hts_action_conv[e]

        return hts_action_conv, sth_action_conv

    # -----------------------------------------------------------------------------------

    def _select_from_dict(self, dictionary: OrderedDict[str, Any], selections: List[str]):
        return_dict = OrderedDict()

        for selection in selections:
            return_dict[selection] = dictionary[selection]

        return return_dict

    # -----------------------------------------------------------------------------------

    def _get_min_maxes(self) -> OrderedDict[str, Dict[str, Tuple[int, int]]]:
        sim_min_maxes = ordered_dict()
        sim_bounds = self.simulation.get_attribute_bounds()
        for attribute in self.sim_attributes:
            sim_min_maxes[attribute] = sim_bounds[attribute]

        nonsim_min_maxes = self._select_from_dict(
            self.get_nonsim_attribute_bounds(), self.nonsim_attributes
        )

        # Removed assert statement because of bandit report:
        # Issue: [B101:assert_used] Use of assert detected. The enclosed code will be
        # removed when compiling to optimised byte code.
        #    Severity: Low   Confidence: High
        #    CWE: CWE-703 (https://cwe.mitre.org/data/definitions/703.html)
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
        def normalize(data, min_max):
            return (data - min_max["min"]) / (min_max["max"] - min_max["min"])

        for attribute in self.normalized_attributes:
            observations[attribute] = normalize(
                observations[attribute], self.min_maxes[attribute]
            )

        return observations

    # -----------------------------------------------------------------------------------

    def reset(self) -> np.ndarray:
        self.simulation.reset()
        sim_observations = self._select_from_dict(
            self.simulation.get_attribute_data(), self.sim_attributes
        )
        nonsim_observations = self._select_from_dict(
            self.get_nonsim_attribute_data(), self.nonsim_attributes
        )

        # Removed assert statement because of bandit report:
        # Issue: [B101:assert_used] Use of assert detected. The enclosed code will be
        # removed when compiling to optimised byte code.
        #    Severity: Low   Confidence: High
        #    CWE: CWE-703 (https://cwe.mitre.org/data/definitions/703.html)

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
    def step(self, action) -> Tuple[np.ndarray, float, bool, Dict]:
        pass

    @abstractmethod
    def render(self) -> None:
        pass

    def hts_mitigation(self, mitigation_map: np.ndarray) -> np.ndarray:
        none_index = self.actions.index("none")
        sim_mitigation_map = []
        for mitigation_i in mitigation_map:
            for action in mitigation_i:
                if self.actions[action] in self.sim_actions:
                    action = self.sth_action_conv[action]
                if not action == none_index:
                    raise ValueError(
                        f"Nonsim action {self.actions[action]} cannot be converted to "
                        "simulator."
                    )
                sim_mitigation_map.append(action)

        return np.asarray(sim_mitigation_map).reshape(
            len(mitigation_map[0]), len(mitigation_map[1])
        )

    def sth_mitigation(self, mitigation_map: np.ndarray) -> np.ndarray:
        none_index = self.actions.index("none")
        harness_mitigation_map = []
        for mitigation_i in mitigation_map:
            for action in mitigation_i:
                if not action == none_index:
                    action = self.sth_action_conv[action]
                harness_mitigation_map.append(action)

        return np.asarray(harness_mitigation_map).reshape(
            len(mitigation_map[0]), len(mitigation_map[1])
        )

    def get_nonsim_attribute_data(self) -> OrderedDict[str, np.ndarray]:
        return ordered_dict()

    def get_nonsim_attribute_bounds(self) -> OrderedDict[str, Dict[str, int]]:
        return ordered_dict()
