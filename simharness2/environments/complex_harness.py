import logging
from typing import Dict, OrderedDict, TypeVar

import numpy as np
from gymnasium import spaces
from simfire.sim.simulation import FireSimulation

from simharness2.environments.fire_harness import ReactiveHarness
from simharness2.models.custom_multimodal_torch_model import FIRE_MAP_KEY, POSITION_KEY


logger = logging.getLogger(__name__)

AnyFireSimulation = TypeVar("AnyFireSimulation", bound=FireSimulation)


# TODO: Where to put this? What to name the subclass?
class ComplexObsReactiveHarness(ReactiveHarness[AnyFireSimulation]):
    def __init__(self, **kwargs):
        # TODO: Verify this call to super init when using **kwargs
        super().__init__(**kwargs)

        # FIXME: Expand to include SimFire data layers (ie. `self.sim_attributes`).
        # if self.attributes != [FIRE_MAP_KEY, POSITION_KEY]:
        #     raise AssertionError(
        #         f"The `ComplexObsReactiveHarness` requires `self.attributes` to be "
        #         f"[{FIRE_MAP_KEY}, {POSITION_KEY}]."
        #     )

        if self.num_agents > 1:
            raise NotImplementedError(
                "The `ComplexObsReactiveHarness` does not support multiple agents."
            )

    def get_nonsim_attribute_data(self) -> OrderedDict[str, np.ndarray]:
        # nonsim_data = {
        #     FIRE_MAP_KEY: self.prepare_fire_map(place_agents=False),

        # }
        # return nonsim_data
        # TODO: Verify if this method is called when using ComplexObsReactiveHarness.
        max_y, max_x = self.sim.fire_map.shape
        default_agent = self.agents[self.default_agent_id]
        pos_state = default_agent.get_normalized_position(max_x=max_x, max_y=max_y)
        nonsim_data = {
            FIRE_MAP_KEY: np.copy(self.sim.fire_map),
            POSITION_KEY: pos_state,
        }
        return nonsim_data

    def get_nonsim_attribute_bounds(self) -> OrderedDict[str, Dict[str, int]]:
        # TODO: Probably better to return a dictionary of min/max for each space, similar
        # to the approach used in the overriden `get_observation_space()` method.
        fire_map_values = self._get_non_interaction_disaster_categories().values()
        fire_map_bounds = {"min": min(fire_map_values), "max": max(fire_map_values)}
        # FIXME: Make this min/max for the 1D vector of length 2.
        pos_bounds = {"min": 0, "max": max(self.sim.fire_map.shape) - 1}
        nonsim_min_maxes = {
            FIRE_MAP_KEY: fire_map_bounds,
            POSITION_KEY: pos_bounds,
        }
        return nonsim_min_maxes

    def _get_state(self):
        sim_observations = super()._select_from_dict(
            self.sim.get_attribute_data(), self.sim_attributes
        )
        nonsim_observations = super()._select_from_dict(
            self.get_nonsim_attribute_data(), self.nonsim_attributes
        )

        firemap_attributes = ["fire_map"]

        observations = super()._normalize_obs({**sim_observations, **nonsim_observations})

        fire_map_obs = [observations[attribute] for attribute in firemap_attributes]

        max_y, max_x = self.sim.fire_map.shape
        default_agent = self.agents[self.default_agent_id]
        pos_state = default_agent.get_normalized_position(max_x=max_x, max_y=max_y)

        return {
            FIRE_MAP_KEY: np.stack(fire_map_obs, axis=-1).astype(np.float32),
            POSITION_KEY: pos_state,
        }

    # FIXME: Add new logic. Current code is just a placeholder.
    def get_initial_state(self) -> np.ndarray:
        """TODO."""
        #fire_map_state = super().get_initial_state()
        sim_observations = super()._select_from_dict(
            self.sim.get_attribute_data(), self.sim_attributes
        )
        nonsim_observations = super()._select_from_dict(
            self.get_nonsim_attribute_data(), self.nonsim_attributes
        )

        firemap_attributes = ["fire_map"]

        observations = super()._normalize_obs({**sim_observations, **nonsim_observations})

        fire_map_obs = [observations[attribute] for attribute in firemap_attributes]

        max_y, max_x = self.sim.fire_map.shape
        default_agent = self.agents[self.default_agent_id]
        pos_state = default_agent.get_normalized_position(max_x=max_x, max_y=max_y)

        return {
            FIRE_MAP_KEY: np.stack(fire_map_obs, axis=-1).astype(np.float32),
            POSITION_KEY: pos_state,
        }

    def get_observation_space(self) -> spaces.Space:
        """TODO."""
        return spaces.Dict(
            {
                FIRE_MAP_KEY: self._get_fire_map_observation_space(),
                POSITION_KEY: self._get_position_observation_space(),
            }
        )

    def _get_fire_map_observation_space(self) -> spaces.Box:
        """TODO."""
        # Ensure POSITION_KEY is not in `self.sim_attributes`.
        """
        if POSITION_KEY in self.nonsim_attributes:
            self.nonsim_attributes.pop(self.nonsim_attributes.index(POSITION_KEY))
        """

        fire_map_attributes = self.attributes.copy()
        fire_map_attributes.pop(fire_map_attributes.index(POSITION_KEY))
        # FIXME: Does custom model expect channel-minor or channel-major format?
        # NOTE: calling `reshape()` to switch to channel-minor format.
        # TODO: Note that the following attributes DO NOT reflect POSITION_KEY.
        self._channel_lows = np.array(
            [[[self.min_maxes[channel]["min"]]] for channel in fire_map_attributes]
        ).reshape(1, 1, len(fire_map_attributes))
        self._channel_highs = np.array(
            [[[self.min_maxes[channel]["max"]]] for channel in fire_map_attributes]
        ).reshape(1, 1, len(fire_map_attributes))

        obs_shape = (
            self.sim.fire_map.shape[0],
            self.sim.fire_map.shape[1],
            len(fire_map_attributes),
        )
        low = np.broadcast_to(self._channel_lows, obs_shape)
        high = np.broadcast_to(self._channel_highs, obs_shape)

        return spaces.Box(low=low, high=high, dtype=np.float32)

    def _get_position_observation_space(self) -> spaces.Box:
        """TODO."""
        row_max, col_max = self.sim.fire_map.shape
        return spaces.Box(low=np.array([0, 0]), high=np.array([row_max - 1, col_max - 1]))

    # FIXME: Add new logic. Current code is just a placeholder.
    def _update_state(self):
        """Modify environment's state to contain updates from the current timestep."""
        # Copy the fire map from the simulation so we don't overwrite it.
        """
        fire_map = np.copy(self.sim.fire_map)
        # Update the fire map with the numeric identifier for the agent.
        for agent in self.agents.values():
            fire_map[agent.row, agent.col] = agent.sim_id
        # Modify the state to contain the updated fire map
        self.state[..., self.attributes.index("fire_map")] = fire_map
        """
        self.state = self._get_state()
