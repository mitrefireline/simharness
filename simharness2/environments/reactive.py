"""FIXME: A one line summary of the module or program.

"Integrates Reward Class that inherits FEAR data extraction and dual (Benchmark + Agent)
simulation capabilities" - dgandikota

Leave one blank line.  The rest of this docstring should contain an
overall description of the module or program.  Optionally, it may also
contain a brief description of exported classes and functions and/or usage
examples.

Typical usage example:

  foo = ClassFoo()
  bar = foo.FunctionBar()
"""
from collections import OrderedDict as ordered_dict
from typing import Any, Dict, List, Optional, OrderedDict, Tuple

import numpy as np
from gymnasium import spaces
from simfire.sim.simulation import FireSimulation

from simharness2.rewards.base_reward import BaseReward

# TODO(afennelly) fix import path (relative to root)
from .rl_harness import RLHarness


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
        - For example, if the sim IS-A `FireSimulation`, possible interactions
            could be: ["fireline", "scratchline", "wetline"]. To learn more, see
            [simulation.py](https://gitlab.mitre.org/fireline/simulators/simfire/-/blob/main/simfire/sim/simulation.py#L269-280).
    - Actions are determined based on the provided (harness) config file.
    - When `super()._init__()` is called, the option "none" is inserted to element 0 of
        both `movements` and `interactions`, representing "don't move" and
        "don't interact", respectively (this is the intuition for the +1 in the shape).

    ### Observation Space
    The observation space type is `Box`, and `sample()` returns an `np.ndarray` of shape
    `(A,X,X)`, where `A == len(ReactiveHarness.attributes)` and
    `X == ReactiveHarness.sim.config.area.screen_size`.
    - The value of `ReactiveHarness.sim.config.area.screen_size` is determined
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

    def __init__(
        self,
        sim: FireSimulation,
        movements: List[str],
        interactions: List[str],
        attributes: List[str],
        normalized_attributes: List[str],
        agent_speed: int,
        reward_cls: BaseReward = None,
        deterministic: bool = False,
        initial_agent_pos: List[int] = [15, 15],
        randomize_initial_agent_pos: bool = False,
        benchmark_sim: FireSimulation = None,
    ) -> None:
        """See RLHarness (parent/base class)."""
        # Track the number of timesteps that have occurred within an episode.
        self.timesteps = 0

        # If provided, the object is used to perform reward calculation.
        self.reward_cls = reward_cls

        # Store parameters relevant to the agent; for use in `step()`, `reset()`, etc.
        self.agent_speed = agent_speed
        self.agent_pos: List[int]
        self.initial_agent_pos = initial_agent_pos
        self.randomize_initial_agent_pos = randomize_initial_agent_pos

        # Set the agent's initial position on the map
        self._set_agent_pos_for_episode_start()

        super().__init__(
            sim,
            movements,
            interactions,
            attributes,
            normalized_attributes,
            deterministic,
            benchmark_sim=benchmark_sim,
        )

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:  # noqa
        # NOTE: We can also return (agent_moved, agent_interacted) as (bool, bool),
        # and then call the `tracker.update_after_one_agent_step()` method (for clarity?)
        self._do_one_agent_step(action)  # alternatively, self._step_agent(action)
        if self.reward_cls:
            # Update reward tracker after agent has taken one step
            self.reward_cls.tracker.update_after_one_agent_step(
                self.agent_pos,
                self.sim.fire_map,
                # FIXME what other args are needed as input??
                # self.interactions[interaction] != "none",
            )
        # NOTE: `sim_run` indicates if `FireSimulation.run()` was called. This helps
        # indicate how to calculate the reward for the current timestep.
        sim_run = self._do_one_simulation_step()  # alternatively, self._step_simulation()
        if sim_run and self.reward_cls:
            # Update reward tracker after simulation has taken one step
            self.reward_cls.tracker.update_after_one_simulation_step(
                self.sim.fire_map,
                self.sim.active,
                self.benchmark_sim.fire_map if self.benchmark_sim else None,
                self.benchmark_sim.active if self.benchmark_sim else None,
                # FIXME what args are needed as input??
            )

        # Calculate the reward for the current timestep
        if self.reward_cls:
            reward = self.reward_cls.get_reward(self.timesteps, sim_run)
        else:
            fire_map_idx = self.attributes.index("fire_map")
            reward = self._calculate_reward(self.state[..., fire_map_idx], sim_run)

        # TODO account for below updates in the reward_cls.calculate_reward() method
        # "End of episode" reward
        if not self.sim.active:
            reward += 10
        # if self._nearby_fire():
        #     reward -= 2.0

        # Convention: increment the timestep AFTER all method logic is performed.
        self.timesteps += 1
        # TODO(afennelly): Need to handle truncation properly. For now, we assume that
        # the episode will never be truncated, but this isn't necessarily true.
        truncated = False
        return self.state, reward, not self.sim.active, truncated, {}

    def _do_one_agent_step(self, action: np.ndarray) -> None:
        """Move the agent and interact with the environment."""
        # Parse the movement and interaction from the action
        movement, interaction = self._parse_action(action)

        # Update agent location on map
        movement_str = self.movements[movement]
        if movement_str != "none":
            self._update_agent_position(movement_str)

        # Check if there was an interaction already done on this space
        is_empty = self._is_empty_space()

        # TODO Penalize agent when `is_empty == False` (chose "invalid" interaction)?
        # Interact with the environment
        if is_empty and self.interactions[interaction] != "none":
            self._update_mitigation(interaction)

        # Update reward tracker after agent has taken one step (callback inside method)
        self.reward_cls.tracker.update_after_one_agent_step(
            self.agent_pos,
            self.sim.fire_map,
            self.interactions[interaction] != "none",
        )

    def _parse_action(self, action: np.ndarray) -> Tuple[int, int]:
        """Parse the action into movement and interaction."""
        # Handle the MultiDiscrete case (currently used in `ReactiveHarness`)
        if isinstance(self.action_space, spaces.MultiDiscrete):
            return action[0], action[1]
        # Handle the Discrete case (currently used in `ReactiveDiscreteHarness`)
        elif isinstance(self.action_space, spaces.Discrete):
            return action % len(self.movements), int(action / len(self.movements))
        else:
            # TODO provide a descriptive error message.
            raise NotImplementedError

    def _update_agent_position(self, movement_str: str) -> None:
        """Update the agent's position on the map by performing the provided movement."""
        # Store agent's current position in a temporary variable to avoid overwriting it.
        temp_agent_pos = self.agent_pos.copy()
        map_boundary = self.sim.config.area.screen_size - 1

        # Update the agent's position based on the provided movement.
        if movement_str == "up" and not self.agent_pos[0] == 0:
            temp_agent_pos[0] -= 1
        elif movement_str == "down" and not self.agent_pos[0] == map_boundary:
            temp_agent_pos[0] += 1
        elif movement_str == "left" and not self.agent_pos[1] == 0:
            temp_agent_pos[1] -= 1
        elif movement_str == "right" and not self.agent_pos[1] == map_boundary:
            temp_agent_pos[1] += 1
        else:
            # TODO should we provide a more descriptive error message here?
            raise ValueError(f"Invalid movement string provided: {movement_str}.")

        # Store the updated agent position.
        self.agent_pos = temp_agent_pos

        # Update the Simulation with new agent position (s).
        # NOTE: We assume the single-agent case here, so agent ID == 0.
        point = [self.agent_pos[1], self.agent_pos[0], 0]
        self.sim.update_agent_positions([point])

    def _is_empty_space(self) -> bool:
        """Check if the space is empty."""
        # FIXME Store the value that indicates whether space is empty or not
        #   - Ex. NOT hardcoding `== 0` (which is `== int(BurnStatus.UNBURNED)`)
        fire_map_idx = self.attributes.index("fire_map")
        return self.state[self.agent_pos[0]][self.agent_pos[1]][fire_map_idx] == 0

    def _update_mitigation(self, interaction: int) -> None:
        """Interact with the environment by performing the provided interaction."""
        # Perform interaction on new space
        sim_interaction = self.harness_to_sim[interaction]
        mitigation_update = (self.agent_pos[1], self.agent_pos[0], sim_interaction)
        self.sim.update_mitigation([mitigation_update])

    def _do_one_simulation_step(self) -> bool:
        """Step the simulation forward one timestep, depending on the "agent's speed"."""
        run_sim = self.timesteps % self.agent_speed == 0
        # The simulation WILL NOT be run every step, unless `self.agent_speed` == 1.
        if run_sim:
            self._run_simulation()
        # Prepare the observation that is returned in the `self.step()` method.
        self._update_state()
        return run_sim

    def _run_simulation(self):
        """Run the simulation (s) for one timestep."""
        if self._use_benchmark_sim:
            # benchmark_sim_fire_map, benchmark_sim_active = self.benchmark_sim.run(1)
            self.benchmark_sim.run(1)

        # sim_fire_map, sim_active = self.sim.run(1)
        self.sim.run(1)

    def _update_state(self):
        """Modify environment's state to contain updates from the current timestep."""
        # Copy the fire map from the simulation so we don't overwrite it.
        fire_map = np.copy(self.sim.fire_map)
        # Update the fire map with the numeric identifier for the agent.
        fire_map[self.agent_pos[0]][self.agent_pos[1]] = self.sim_agent_id
        # Modify the state to contain the updated fire map
        fire_map_idx = self.attributes.index("fire_map")
        self.state[..., fire_map_idx] = fire_map

    def _nearby_fire(self) -> bool:
        """Check if the agent is adjacent to a space that is currently burning.

        Returns:
            nearby_fire: A boolean indicating if there is a burning space adjacent to the
              agent.
        """
        nearby_locs = []
        screen_size = self.sim.config.area.screen_size
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

        for (i, j) in nearby_locs:
            if self.state[self.attributes.index("fire_map")][i][j] == 1:
                return True

        return False

    def _calculate_reward(self, fire_map: np.ndarray, sim_run: bool) -> float:
        """Calculate the reward given the current fire_map.

        Arguments:
            fire_map: An ndarray containing the current state of the `Simulation`.
            sim_run: A boolean indicating whether the simulation was run this timestep.

        Returns:
            reward: A float representing the reward for given state.
        """
        if sim_run:
            return 0.0

        burning = np.count_nonzero(fire_map == 1)
        # burnt = np.count_nonzero(fire_map == 2)

        # diff = burnt - self.num_burned
        # self.num_burned = burnt

        # firelines = np.count_nonzero(fire_map == 3)

        total = self.sim.config.area.screen_size**2
        reward = -(burning / total) * 10

        return reward

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[Any, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[Any, Any]]:  # noqa
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        # If the environment is stochastic, set the seeds for randomization parameters.
        # An evaluation environment will generally be set as deterministic.
        # NOTE: Other randomization parameters include "fuel", "wind_speed", and
        # "wind_direction". For reference with `FireSimulation`, see
        # https://gitlab.mitre.org/fireline/simulators/simfire/-/blob/d70358ec960af5cfbf1855ef78218475cc569247/simfire/sim/simulation.py#L672-718
        # TODO(afennelly) Enable selecting attributes to randomize from config file.
        if not self.deterministic:
            # Set seeds for randomization
            fire_init_seed = self.sim.get_seeds()["fire_initial_position"]
            elevation_seed = self.sim.get_seeds()["elevation"]
            seed_dict = {
                "fire_initial_position": fire_init_seed + 1,
                "elevation": elevation_seed + 1,
            }
            self.sim.set_seeds(seed_dict)
            # set seeds of benchmark simulation
            self.benchmark_sim.set_seeds(seed_dict)

        # Reset the `Simulation` to initial conditions. In particular, this resets the
        # `fire_map`, `terrain`, `fire_manager`, and all mitigations.
        self.sim.reset()
        # reset benchmark simulation
        self.benchmark_sim.reset()

        # Reset the agent's initial position on the map
        self._set_agent_pos_for_episode_start()

        # Get the starting state of the `Simulation` after it has been reset (above).
        sim_observations = super()._select_from_dict(
            self.sim.get_attribute_data(), self.sim_attributes
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
        self.sim.update_agent_positions([point])
        # update the benchmark simulation - Not sure if actually needed but can't hurt
        self.benchmark_sim.update_agent_positions([point])

        # NOTE: `self.num_burned` is not currently used in the reward calculation.
        # self.num_burned = 0 FIXME include once we modularize the reward function
        self.num_agent_steps = 0

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
                self.sim.config.area.screen_size,
                self.sim.config.area.screen_size,
            )
        )

        # Place the agent on the fire map using the agent ID.
        nonsim_data["fire_map"][self.agent_pos[0]][self.agent_pos[1]] = self.sim_agent_id
        # FIXME the below line has no dependence on `nonsim_data`; needs to be moved.
        # FIXME Why are we placing a fireline at the agents position here?
        self.sim.update_mitigation([(self.agent_pos[1], self.agent_pos[0], 3)])

        return nonsim_data

    def render(self):  # noqa
        self.sim.rendering = True

    def _set_agent_pos_for_episode_start(self):
        """Set the agent's initial position in the map for the start of the episode."""
        if self.randomize_initial_agent_pos:
            self.agent_pos = self.np_random.integers(
                0, self.sim.config.area.screen_size, size=2, dtype=int
            )
        else:
            # TODO(afennelly): Verify initial_agent_pos is within the bounds of the map
            self.agent_pos = self.initial_agent_pos


class PGReactiveHarness(ReactiveHarness):  # noqa: D205,D212,D415
    """
    ### Description
    Progressive Growth Reactive Harness for use with slowly growing the "static" fire
    start location. The fire start location will expand every `sims_per_growth`
    simulations. The fire start location will expand up to `growth_per_step` pixels for
    every start location growth. `growth_seed` will control the randomization of start
    locations for reproducability.
    """

    def __init__(
        self,
        sim: FireSimulation,
        movements: List[str],
        interactions: List[str],
        attributes: List[str],
        normalized_attributes: List[str],
        agent_speed: int,
        deterministic: bool = False,
        initial_agent_pos: List[int] = [15, 15],
        randomize_initial_agent_pos: bool = False,
        sims_per_growth: int = 1,
        growth_per_step: int = 1,
    ) -> None:
        """Initialize the harness.

        This checks that the fire initial position is static
        and creates a counter for the total number of simulations run.

        Progressive Growth Reactive Harness for use with slowly growing the "static" fire
        start location. The fire start location will expand every `sims_per_growth`
        simulations. The fire start location will expand up to `growth_per_step` pixels
        for every start location growth.

        TODO (afennelly) Update docstring.

        """
        super().__init__(
            sim,
            movements,
            interactions,
            attributes,
            normalized_attributes,
            agent_speed,
            deterministic,
            initial_agent_pos,
            randomize_initial_agent_pos,
        )
        # Verify that a static initial position is used
        fire_init_pos_type = self.sim.config.yaml_data["fire"]["fire_initial_position"][
            "type"
        ]
        if fire_init_pos_type != "static":
            raise ValueError(
                "The PG Harness requires a static `fire_initial_position`, "
                f"but the supplied value is {fire_init_pos_type}"
            )
        # Number of sims to run before each growth step
        self.sims_per_growth = sims_per_growth
        # Number of pixels to allow starting region to grow per growths step
        self.growth_per_step = growth_per_step
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
        fire_pos = self.sim.config.yaml_data["fire"]["fire_initial_position"]["static"][
            "position"
        ]
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
        new_x = max(0, min(new_x, self.sim.config.yaml_data["area"]["screen_size"] - 1))
        new_y = max(0, min(new_y, self.sim.config.yaml_data["area"]["screen_size"] - 1))
        self.sim.set_fire_initial_position((new_x, new_y))

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

    def __init__(
        self,
        sim: FireSimulation,
        movements: List[str],
        interactions: List[str],
        attributes: List[str],
        normalized_attributes: List[str],
        agent_speed: int,
        deterministic: bool = False,
        agent_pos: List[int] = [15, 15],
        randomize_agent_pos: bool = False,
    ) -> None:
        """See ReactiveHarness (parent/base class)."""
        super().__init__(
            sim,
            movements,
            interactions,
            attributes,
            normalized_attributes,
            agent_speed,
            deterministic,
            agent_pos,
            randomize_agent_pos,
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
        screen_size = self.sim.config.area.screen_size

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
            self.sim.update_mitigation([mitigation_update])

        # Update the Simulation with new agent position (s).
        # NOTE: We assume the single-agent case here, so agent ID == 0.
        point = [self.agent_pos[1], self.agent_pos[0], 0]
        self.sim.update_agent_positions([point])

        # Don't run the Simulation every step depending on speed
        if self.num_agent_steps % self.agent_speed == 0:
            sim_fire_map, sim_active = self.sim.run(1)
            fire_map = np.copy(sim_fire_map)
            fire_map[self.agent_pos[0]][self.agent_pos[1]] = self.sim_agent_id
            reward += self._calculate_reward(fire_map)
        else:
            sim_active = True
            sim_fire_map = self.sim.fire_map
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
