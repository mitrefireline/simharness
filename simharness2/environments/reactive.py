"""FIXME: A one line summary of the module or program.

Leave one blank line.  The rest of this docstring should contain an
overall description of the module or program.  Optionally, it may also
contain a brief description of exported classes and functions and/or usage
examples.

Typical usage example:

  foo = ClassFoo()
  bar = foo.FunctionBar()
"""
import logging
from collections import OrderedDict as ordered_dict
from typing import Any, Dict, List, Optional, OrderedDict, Tuple

import numpy as np
from gymnasium import spaces
from gymnasium.envs.registration import EnvSpec
from ray.rllib.env.env_context import EnvContext
from simfire.utils.log import create_logger

# TODO(afennelly) fix import path (relative to root)
from .rl_harness import RLHarness

log = create_logger(__name__)


class ReactiveHarness(RLHarness):  # noqa: D205,D212,D415
    """
    ### Description
    Model's the `reactive` case, where an agent is interacting with the environment as
    a disaster scenario is currently happening and resources are being deployed.

    ### Action Space
    The action space type is `MultiDiscrete`, and `sample()` returns an `np.ndarray` of
    shape `(M+1,I+1)`, where `M == movements` and `I == interactions`.
    - Movements refer to actions where the agent **traverses** the environment.
        - For example, possible movements could be: ["up", "down", "left", "right"].
    - Interactions refer to actions where the agent **interacts** with the environment.
        - For example, if the simulation IS-A `FireSimulation`, possible interactions
            could be: ["fireline", "scratchline", "wetline"]. To learn more, see
            [simulation.py](https://gitlab.mitre.org/fireline/simulators/simfire/-/blob/main/simfire/sim/simulation.py#L269-280).
    - Actions are determined based on the provided (harness) config file.
    - When `super()._init__()` is called, the option "none" is inserted to element 0 of
        both `movements` and `interactions`, representing "don't move" and
        "don't interact", respectively (this is the intuition for the +1 in the shape).

    ### Observation Space
    The observation space type is `Box`, and `sample()` returns an `np.ndarray` of shape
    `(A,X,X)`, where `A == len(ReactiveHarness.attributes)` and
    `X == ReactiveHarness.simulation.config.area.screen_size`.
    - The value of `ReactiveHarness.simulation.config.area.screen_size` is determined
      based on the value of the `screen_size` attribute (within the `area` section) of
      the (simulation) config file. See `simharness2/sim_registry.py` to find more info
      about the `register_simulation()` method, which is used to register the simulation
      class and set the config file path associated with a given simulation.
    - The number of `attributes` is determined by the `attributes` attribute (within the
      `RLHARNESS` section) of the (harness) config file. Each attribute must be contained
      in the observation space returned for the respective `Simulation` class. The
      locations within the observation are based ontheir corresponding location within
      the array.

    ### Rewards
    The agent is rewarded for saving the most land and reducing the amount of affected
    area.
    - TODO(afennelly) add more details about the reward function.
    - TODO(afennelly) implement modular reward function configuration.

    ### Starting State
    The initial map is set by data given from the Simulation.
    - TODO(afennelly) add more details about the starting state.

    ### Episode Termination
    The episode ends once the disaster is finished and it cannot spread any more.
    - TODO(afennelly) add more details about the episode termination.
    """

    def __init__(self, config: EnvContext) -> None:
        """See RLHarness (parent/base class)."""
        # NOTE: We don't set a default value in `config.get` for required arguments.

        # Create a log that will be used to log information about the environment
        self.log = logging.getLogger(__name__)
        # Indicates that environment information should be logged at various points.
        self._debug_mode = config.get("debug_mode", False)
        self._debug_duration = config.get("debug_duration", 1)
        self._episodes_debugged = 0
        self.log.debug(f"Initializing environment {hex(id(self))}")

        # When there are multiple workers created, this uniquely identifies the worker
        # the env is created in. 0 for local worker, >0 for remote workers.
        self.worker_idx = config.worker_index
        # When there are multiple envs per worker, this uniquely identifies the env index
        # within the worker. Starts from 0.
        self.vector_idx = config.vector_index
        # Whether individual sub-envs (in a vectorized env) are @ray.remote actors.
        self.is_remote = config.remote
        # Total number of (remote) workers in the set. 0 if only a local worker exists.
        self.num_workers = config.num_workers
        # Set the max number of steps that the environment can take before truncation
        # self.spec.max_episode_steps = 1000
        self.spec = EnvSpec(
            id="ReactiveHarness-v0",
            entry_point="simharness2.environments.reactive:ReactiveHarness",
            max_episode_steps=2000,
        )
        # Set the number of steps an agent has taken in the current simulation.
        self.num_agent_steps = 0
        self.agent_speed = config.get("agent_speed")

        # Store agent position parameters for use in `step()`, `reset()`, etc.
        self.agent_pos: List[int]
        # FIXME: Default value (ie. [15, 15]) should be set in the config file.
        self.initial_agent_pos = config.get("initial_agent_pos", [15, 15])
        self.randomize_initial_agent_pos = config.get(
            "randomize_initial_agent_pos", False
        )
        # Set the agent's initial position on the map
        self._set_agent_pos_for_episode_start()

        super().__init__(
            config.get("simulation"),
            config.get("movements"),
            config.get("interactions"),
            config.get("attributes"),
            config.get("normalized_attributes"),
            config.get("deterministic"),
        )
        self._log_env_init()

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:  # noqa
        # TODO(afennelly) Add docstring?
        movement = action[0]
        interaction = action[1]

        movement_str = self.movements[movement]
        interaction_str = self.interactions[interaction]
        reward = 0.0

        pos_placeholder = self.agent_pos.copy()
        screen_size = self.simulation.config.area.screen_size

        # Update agent location on map
        if movement_str == "none":
            pass
        elif movement_str == "up" and not self.agent_pos[0] == 0:
            pos_placeholder[0] -= 1
        elif movement_str == "down" and not self.agent_pos[0] == screen_size - 1:
            pos_placeholder[0] += 1
        elif movement_str == "left" and not self.agent_pos[1] == 0:
            pos_placeholder[1] -= 1
        elif movement_str == "right" and not self.agent_pos[1] == screen_size - 1:
            pos_placeholder[1] += 1
        else:
            pass

        self.agent_pos = pos_placeholder

        # Check if there was an interaction already done on this space
        fire_map_idx = self.attributes.index("fire_map")
        is_empty = self.state[self.agent_pos[0]][self.agent_pos[1]][fire_map_idx] == 0

        if is_empty and not interaction_str == "none":
            # Perform interaction on new space
            sim_interaction = self.harness_to_sim[interaction]
            mitigation_update = (self.agent_pos[1], self.agent_pos[0], sim_interaction)
            self.simulation.update_mitigation([mitigation_update])

        # Update the Simulation with new agent position (s).
        # NOTE: We assume the single-agent case here, so agent ID == 0.
        point = [self.agent_pos[1], self.agent_pos[0], 0]
        self.simulation.update_agent_positions([point])

        # Don't run the Simulation every step depending on speed
        if self.num_agent_steps % self.agent_speed == 0:
            sim_fire_map, sim_active = self.simulation.run(1)
            fire_map = np.copy(sim_fire_map)
            fire_map[self.agent_pos[0]][self.agent_pos[1]] = self.sim_agent_id
            reward += self._calculate_reward(fire_map)
        else:
            sim_active = True
            sim_fire_map = self.simulation.fire_map
            fire_map = np.copy(sim_fire_map)
            fire_map[self.agent_pos[0]][self.agent_pos[1]] = self.sim_agent_id

        # Update the state with the new fire map
        self.state[..., fire_map_idx] = fire_map

        if not sim_active:
            reward += 10

        # if self._nearby_fire():
        #     reward -= 2.0

        self.num_agent_steps += 1
        # TODO(afennelly): Need to handle truncation properly. For now, we assume that
        # the episode will never be truncated, but this isn't necessarily true.
        truncated = False
        return self.state, reward, not sim_active, truncated, {}

    def _nearby_fire(self) -> bool:
        """Check if the agent is adjacent to a space that is currently burning.

        Returns:
            nearby_fire: A boolean indicating if there is a burning space adjacent to the
              agent.
        """
        nearby_locs = []
        screen_size = self.simulation.config.area.screen_size
        # Get all spaces surrounding agent
        for i in range(self.agent_pos[0] - 1, self.agent_pos[0] + 2):
            for j in range(self.agent_pos[1] - 1, self.agent_pos[1] + 2):
                if (
                    i < 0
                    or i >= screen_size
                    or j < 0
                    or j >= screen_size
                    or [i, j] == self.agent_pos
                ):
                    pass
                else:
                    nearby_locs.append((i, j))

        for i, j in nearby_locs:
            if self.state[self.attributes.index("fire_map")][i][j] == 1:
                return True

        return False

    def _calculate_reward(self, fire_map: np.ndarray) -> float:
        """Calculate the reward given the current fire_map.

        Arguments:
            fire_map: An ndarray containing the current state of the `Simulation`.

        Returns:
            reward: A float representing the reward for given state.
        """
        burning = np.count_nonzero(fire_map == 1)
        # burnt = np.count_nonzero(fire_map == 2)

        # diff = burnt - self.num_burned
        # self.num_burned = burnt

        # firelines = np.count_nonzero(fire_map == 3)

        total = self.simulation.config.area.screen_size**2
        reward = -(burning / total) * 10

        return reward

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[Any, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[Any, Any]]:  # noqa
        # log.info("Resetting environment")
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        # If the environment is stochastic, set the seeds for randomization parameters.
        # An evaluation environment will generally be set as deterministic.
        # NOTE: Other randomization parameters include "fuel", "wind_speed", and
        # "wind_direction". For reference with `FireSimulation`, see
        # https://gitlab.mitre.org/fireline/simulators/simfire/-/blob/d70358ec960af5cfbf1855ef78218475cc569247/simfire/sim/simulation.py#L672-718
        # TODO(afennelly) Enable selecting attributes to randomize from config file.
        # FIXME this needs to not be hard-coded and moved outside of method logic.
        # if not self.deterministic:
        #     # Set seeds for randomization
        #     fire_init_seed = self.simulation.get_seeds()["fire_initial_position"]
        #     elevation_seed = self.simulation.get_seeds()["elevation"]
        #     seed_dict = {
        #         "fire_initial_position": fire_init_seed + 1,
        #         "elevation": elevation_seed + 1,
        #     }
        #     self.simulation.set_seeds(seed_dict)

        # Reset the `Simulation` to initial conditions. In particular, this resets the
        # `fire_map`, `terrain`, `fire_manager`, and all mitigations.
        self.simulation.reset()

        # Reset the agent's initial position on the map
        self._set_agent_pos_for_episode_start()

        # Get the starting state of the `Simulation` after it has been reset (above).
        sim_observations = super()._select_from_dict(
            self.simulation.get_attribute_data(), self.sim_attributes
        )
        nonsim_observations = super()._select_from_dict(
            self.get_nonsim_attribute_data(), self.nonsim_attributes
        )

        if len(nonsim_observations) != len(self.nonsim_attributes):
            raise AssertionError(
                f"Data for {len(nonsim_observations)} nonsim attributes were given but "
                f"there are {len(self.nonsim_attributes)} nonsim attributes."
            )

        observations = super()._normalize_obs({**sim_observations, **nonsim_observations})

        obs = []
        for attribute in self.attributes:
            obs.append(observations[attribute])

        # NOTE: We may be able to use lower precision here, such as np.float32.
        self.state = np.stack(obs, axis=-1).astype(np.float32)

        # Update the Simulation with new agent position (s).
        # NOTE: We assume the single-agent case here, so agent ID == 0.
        point = [self.agent_pos[1], self.agent_pos[0], 0]
        self.simulation.update_agent_positions([point])

        # NOTE: `self.num_burned` is not currently used in the reward calculation.
        # self.num_burned = 0 FIXME include once we modularize the reward function
        self.num_agent_steps = 0

        self._log_env_reset()

        return self.state, {}

    def get_nonsim_attribute_bounds(self) -> OrderedDict[str, Dict[str, int]]:  # noqa
        nonsim_min_maxes = ordered_dict()
        # The values in "fire_map" are:
        #   - 0: BurnStatus.UNBURNED
        #   - 1: BurnStatus.BURNING
        #   - 2: BurnStatus.BURNED
        #   - 3: BurnStatus.FIRELINE (if "fireline" in self.interactions)
        #   - 4: BurnStatus.SCRATCHLINE (if "scratchline" in self.interactions)
        #   - 5: BurnStatus.WETLINE (if "wetline" in self.interactions)
        #   - X: self.sim_agent_id (value is set in RLHarness.__init__)
        nonsim_min_maxes["fire_map"] = {"min": 0, "max": self.sim_agent_id}
        return nonsim_min_maxes

    def get_nonsim_attribute_data(self) -> OrderedDict[str, np.ndarray]:  # noqa
        # TODO(afennelly) Make note that agent is "placed" on the `fire_map`, etc. here.
        # This method is more of a `reset_and_update_nonsim_attribute_data` method.
        nonsim_data = ordered_dict()

        nonsim_data["fire_map"] = np.zeros(
            (
                self.simulation.config.area.screen_size,
                self.simulation.config.area.screen_size,
            )
        )

        # Place the agent on the fire map using the agent ID.
        nonsim_data["fire_map"][self.agent_pos[0]][self.agent_pos[1]] = self.sim_agent_id
        # FIXME the below line has no dependence on `nonsim_data`; needs to be moved.
        # FIXME Why are we placing a fireline at the agents position here?
        self.simulation.update_mitigation([(self.agent_pos[1], self.agent_pos[0], 3)])

        return nonsim_data

    def render(self):  # noqa
        self.simulation.rendering = True

    # TODO(atapley): Move this to RLHarness
    def _set_agent_pos_for_episode_start(self):
        """Set the agent's initial position in the map for the start of the episode."""
        if self.randomize_initial_agent_pos:
            self.agent_pos = self.np_random.integers(
                0, self.simulation.config.area.screen_size, size=2, dtype=int
            )
        else:
            # TODO(afennelly): Verify initial_agent_pos is within the bounds of the map
            self.agent_pos = self.initial_agent_pos

    def _log_env_init(self):
        """Log information about the environment that is being initialized."""

        if self._is_eval_env:
            i, j = self.worker_idx, self.vector_idx
            self.log.warning(
                f"Object {hex(id(self))}: index (i+1)*(j+1) == {(i+1)*(j+1)}"
            )

        if not self._debug_mode:
            return

        # TODO: What log level should we use here?
        self.log.info(f"Object {hex(id(self))}: worker_index: {self.worker_idx}")
        self.log.info(f"Object {hex(id(self))}: vector_index: {self.vector_idx}")
        self.log.info(f"Object {hex(id(self))}: num_workers: {self.num_workers}")
        self.log.info(f"Object {hex(id(self))}: is_remote: {self.is_remote}")

    def _log_env_reset(self):
        """Log information about the environment that is being reset."""
        if not self._debug_mode or self._episodes_debugged > self._debug_duration:
            return

        # TODO: What log level should we use here?
        for idx, feat in enumerate(self.attributes):
            low, high = self.low[..., idx].min(), self.high[..., idx].max()
            obs_min = round(self.state[..., idx].min(), 2)
            obs_max = round(self.state[..., idx].max(), 2)
            # Log lower bound of the (obs space) and max returned obs for each attribute.
            self.log.info(f"{feat} LB: {low}, obs min: {obs_min}")
            # Log upper (lower) bounds of the returned observations for each attribute.
            self.log.info(f"{feat} UB: {high}, obs max: {obs_max}")

        # Increment the number of episodes that have been debugged.
        self._episodes_debugged += 1


class PGReactiveHarness(ReactiveHarness):  # noqa: D205,D212,D415
    """
    ### Description
    Progressive Growth Reactive Harness for use with slowly growing the "static" fire
    start location. The fire start location will expand every `sims_per_growth`
    simulations. The fire start location will expand up to `growth_per_step` pixels for
    every start location growth. `growth_seed` will control the randomization of start
    locations for reproducability.
    """

    def __init__(self, config: EnvContext) -> None:
        """Initialize the harness.

        This checks that the fire initial position is static
        and creates a counter for the total number of simulations run.

        Progressive Growth Reactive Harness for use with slowly growing the "static" fire
        start location. The fire start location will expand every `sims_per_growth`
        simulations. The fire start location will expand up to `growth_per_step` pixels
        for every start location growth.

        TODO (afennelly) Update docstring.

        """
        super().__init__(config)
        # Verify that a static initial position is used
        fire_init_pos_type = self.simulation.config.yaml_data["fire"][
            "fire_initial_position"
        ]["type"]
        if fire_init_pos_type != "static":
            raise ValueError(
                "The PG Harness requires a static `fire_initial_position`, "
                f"but the supplied value is {fire_init_pos_type}"
            )
        # Number of sims to run before each growth step
        self.sims_per_growth = config.get("sims_per_growth", 1)
        # Number of pixels to allow starting region to grow per growths step
        self.growth_per_step = config.get("growth_per_step", 1)
        # Number of times the simulation has run (or been reset)
        self.num_sims = 0

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[Any, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[Any, Any]]:  # noqa
        output, _ = super().reset()
        # FIXME Currently, `ReactiveHarness.reset()` will update the seed used for
        # "fire_initial_position", which is not what we want in the PG case.

        # Update the number of times the simulation has run (or been reset).
        self.num_sims += 1

        # Retreive the "static" fire start position, as specified in the config.
        fire_pos = self.simulation.config.yaml_data["fire"]["fire_initial_position"][
            "static"
        ]["position"]
        fire_pos = fire_pos[1:-1].split(",")
        x, y = (int(fire_pos[0]), int(fire_pos[1]))

        # Randomly select a new fire start position within the growth range.
        additional_pts = (self.num_sims // self.sims_per_growth) * self.growth_per_step
        new_x = x + self.np_random.integers(
            -additional_pts, additional_pts + 1, dtype=int
        )
        new_y = y + self.np_random.integers(
            -additional_pts, additional_pts + 1, dtype=int
        )

        # Clamp the new values to the screen size.
        new_x = max(
            0, min(new_x, self.simulation.config.yaml_data["area"]["screen_size"] - 1)
        )
        new_y = max(
            0, min(new_y, self.simulation.config.yaml_data["area"]["screen_size"] - 1)
        )
        self.simulation.set_fire_initial_position((new_x, new_y))

        return output, {}


class ReactiveDiscreteHarness(ReactiveHarness):  # noqa: D205,D212,D415
    """
    ### Description
    This environment is the same as the above ReactiveHarness, except it uses a discrete
    action space instead of a multi-discrete (used for algorithms that cannot support
    multi-discrete)

    ### Action Space
    The action space is discrete with the shape `((M*I)+3)`. Movements refer to actions
    where the agent traverses the environment (ie up, down). Interactions refer to actions
    where the agent interacts with the environment (ie place fireline, cut trees). The
    environment combines both movements and actions into a discrete space and adds the
    option for no movement and no interaction.
    """

    def __init__(self, config: EnvContext) -> None:
        """See ReactiveHarness (parent/base class)."""
        super().__init__(config)
        self.spec = EnvSpec(
            id="ReactiveHarness-v1",
            entry_point="simharness2.environments.reactive:ReactiveDiscreteHarness",
            max_episode_steps=2000,
        )
        action_shape = len(self.movements) * len(self.interactions)
        # Overwrite the action space to be Discrete.
        self.action_space = spaces.Discrete(action_shape)

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:  # noqa
        # TODO(afennelly) Add docstring?
        # Separate out the movement and the interaction from the action
        movement = action % len(self.movements)
        interaction = int(action / len(self.movements))

        # FIXME (afennelly) Everything below is identical to parent class. We should
        # refactor this to avoid code duplication, but it's okay to keep for now.
        movement_str = self.movements[movement]
        interaction_str = self.interactions[interaction]
        reward = 0.0

        pos_placeholder = self.agent_pos.copy()
        screen_size = self.simulation.config.area.screen_size

        # Update agent location on map
        if movement_str == "none":
            pass
        elif movement_str == "up" and not self.agent_pos[0] == 0:
            pos_placeholder[0] -= 1
        elif movement_str == "down" and not self.agent_pos[0] == screen_size - 1:
            pos_placeholder[0] += 1
        elif movement_str == "left" and not self.agent_pos[1] == 0:
            pos_placeholder[1] -= 1
        elif movement_str == "right" and not self.agent_pos[1] == screen_size - 1:
            pos_placeholder[1] += 1
        else:
            pass

        self.agent_pos = pos_placeholder

        # Check if there was an interaction already done on this space
        fire_map_idx = self.attributes.index("fire_map")
        is_empty = self.state[self.agent_pos[0]][self.agent_pos[1]][fire_map_idx] == 0

        if is_empty and not interaction_str == "none":
            # Perform interaction on new space
            sim_interaction = self.harness_to_sim[interaction]
            mitigation_update = (self.agent_pos[1], self.agent_pos[0], sim_interaction)
            self.simulation.update_mitigation([mitigation_update])

        # Update the Simulation with new agent position (s).
        # NOTE: We assume the single-agent case here, so agent ID == 0.
        point = [self.agent_pos[1], self.agent_pos[0], 0]
        self.simulation.update_agent_positions([point])

        # Don't run the Simulation every step depending on speed
        if self.num_agent_steps % self.agent_speed == 0:
            sim_fire_map, sim_active = self.simulation.run(1)
            fire_map = np.copy(sim_fire_map)
            fire_map[self.agent_pos[0]][self.agent_pos[1]] = self.sim_agent_id
            reward += self._calculate_reward(fire_map)
        else:
            sim_active = True
            sim_fire_map = self.simulation.fire_map
            fire_map = np.copy(sim_fire_map)
            fire_map[self.agent_pos[0]][self.agent_pos[1]] = self.sim_agent_id

        # Update the state with the new fire map
        self.state[..., fire_map_idx] = fire_map

        if not sim_active:
            reward += 10

        # if self._nearby_fire():
        #     reward -= 2.0

        self.num_agent_steps += 1
        # TODO(afennelly): Need to handle truncation properly. For now, we assume that
        # the episode will never be truncated, but this isn't necessarily true.
        truncated = False
        return self.state, reward, not sim_active, truncated, {}
