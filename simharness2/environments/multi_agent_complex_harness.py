import logging
from collections import OrderedDict
from typing import List, TypeVar

import numpy as np
from gymnasium import spaces
from simfire.sim.simulation import FireSimulation

from simharness2.environments.harness import get_unsupported_attributes
from simharness2.environments.multi_agent_fire_harness import MultiAgentFireHarness
from simharness2.models.custom_multimodal_torch_model import (
    AGENT_POSITION_KEY,
    FIRE_MAP_KEY,
)


logger = logging.getLogger(__name__)

AnyFireSimulation = TypeVar("AnyFireSimulation", bound=FireSimulation)


class MultiAgentComplexObsReactiveHarness(MultiAgentFireHarness[AnyFireSimulation]):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # TODO: Make this check more general, ie. not included in every subclass?
        # Validate that the attributes provided are supported by this harness.
        curr_cls = self.__class__
        attrs_to_check = self.attributes + self.sim_attributes
        bad_attributes = get_unsupported_attributes(attrs_to_check, curr_cls)
        if bad_attributes:
            msg = (
                f"The {curr_cls.__name__} class does not support the "
                f"following attributes: {bad_attributes}."
            )
            raise AssertionError(msg)

    @staticmethod
    def supported_attributes() -> List[str]:
        """Return the full list of attributes supported by the harness."""
        # TODO: Expand to include SimFire data layers, when ready.
        return [FIRE_MAP_KEY, AGENT_POSITION_KEY] + FireSimulation.supported_attributes()

    def get_initial_state(self) -> np.ndarray:
        """TODO."""
        fire_map = self.prepare_fire_map(place_agents=False)

        if self.sim_attributes:
            sim_data = self.sim.get_attribute_data()
            sim_data_to_use = [
                arr for attr_name, arr in sim_data.items() if attr_name in self.attributes
            ]
            fire_map = np.stack([fire_map] + sim_data_to_use, axis=-1).astype(np.float32)
        else:
            fire_map = np.expand_dims(fire_map, axis=-1).astype(np.float32)

        # Build MARL obs - position array will be different for each agent.
        marl_obs = {}
        for ag_id in self._agent_ids:
            curr_agent = self.agents[ag_id]
            pos_state = curr_agent.initial_position
            marl_obs[ag_id] = OrderedDict(
                {
                    FIRE_MAP_KEY: fire_map,
                    AGENT_POSITION_KEY: np.asarray(pos_state, dtype=np.float32),
                }
            )
            # marl_obs[ag_id] = {
            #     FIRE_MAP_KEY: fire_map,
            #     AGENT_POSITION_KEY: np.asarray(pos_state),
            # }

        # Note: Returning ordered dict bc spaces.Dict.sample() returns ordered dict.
        # return OrderedDict(marl_obs)
        return marl_obs

    def get_observation_space(self) -> spaces.Space:
        """TODO."""
        # NOTE: We are assuming each agent has the same observation space.
        agent_obs_space = spaces.Dict(
            OrderedDict(
                {
                    FIRE_MAP_KEY: self._get_fire_map_observation_space(),
                    AGENT_POSITION_KEY: self._get_position_observation_space(),
                }
            )
        )

        self._obs_space_in_preferred_format = True
        return spaces.Dict({agent_id: agent_obs_space for agent_id in self._agent_ids})

    def _get_fire_map_observation_space(self) -> spaces.Box:
        """TODO."""
        from simfire.enums import BurnStatus

        # TODO: Refactor to enable easier reuse of this method logic.
        # Prepare user-provided interactions
        interacts = [i.lower() for i in self.interactions]
        if "none" in interacts:
            interacts.pop(interacts.index("none"))
        # Prepare non-interaction disaster categories
        non_interacts = list(self._get_non_interaction_disaster_categories().keys())
        non_interacts = [i.lower() for i in non_interacts]
        cats = interacts + non_interacts
        cat_vals = []
        for status in BurnStatus:
            if status.name.lower() in cats:
                cat_vals.append(status.value)

        low = min(cat_vals)
        high = max(cat_vals)

        num_channels = 1 + len(self.sim_attributes)
        obs_shape = self.sim.fire_map.shape + (num_channels,)

        # FIXME: Manually setting low and high to -inf, +inf for now.
        low, high = float("-inf"), float("inf")
        return spaces.Box(low=low, high=high, shape=obs_shape, dtype=np.float32)

    def _get_position_observation_space(self) -> spaces.Box:
        """TODO."""
        row_max, col_max = self.sim.fire_map.shape
        return spaces.Box(low=np.array([0, 0]), high=np.array([row_max - 1, col_max - 1]))

    def _update_state(self):
        """Modify environment's state to contain updates from the current timestep."""
        # Copy the fire map from the simulation so we don't overwrite it.
        fire_map = np.copy(self.sim.fire_map)

        if self.sim_attributes:
            sim_data = self.sim.get_attribute_data()
            sim_data_to_use = [
                arr for attr_name, arr in sim_data.items() if attr_name in self.attributes
            ]
            fire_map = np.stack([fire_map] + sim_data_to_use, axis=-1).astype(np.float32)
        else:
            fire_map = np.expand_dims(fire_map, axis=-1).astype(np.float32)

        # Build MARL obs - position array will be different for each agent.
        marl_obs = {}
        for ag_id in self._agent_ids:
            curr_agent = self.agents[ag_id]
            pos_state = curr_agent.current_position
            marl_obs[ag_id] = OrderedDict(
                {
                    FIRE_MAP_KEY: fire_map,
                    AGENT_POSITION_KEY: np.asarray(pos_state, dtype=np.float32),
                }
            )
        # Note: Setting to ordered dict bc spaces.Dict.sample() returns ordered dict.
        # self.state = OrderedDict(marl_obs)
        self.state = marl_obs
