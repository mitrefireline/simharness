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
import os
from collections import OrderedDict as ordered_dict
from functools import partial
from typing import Any, Dict, List, Optional, OrderedDict, Tuple

import numpy as np
from gymnasium import spaces
from gymnasium.envs.registration import EnvSpec
from ray.rllib.env.env_context import EnvContext
from simfire.enums import BurnStatus
from simfire.utils.config import Config

from simharness2.analytics.harness_analytics import ReactiveHarnessAnalytics
from simharness2.environments.rl_harness import RLHarness
from simharness2.rewards.base_reward import BaseReward

logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
handler.setFormatter(
    logging.Formatter("%(asctime)s\t%(levelname)s %(filename)s:%(lineno)s -- %(message)s")
)
logger.addHandler(handler)
logger.propagate = False


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
    `X == ReactiveHarness.sim.config.area.screen_size[0]`.
    - The value of `ReactiveHarness.sim.config.area.screen_size[0]` is determined
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
        # Indicates that environment information should be logged at various points.
        self._debug_mode = config.get("debug_mode", False)
        self._debug_duration = config.get("debug_duration", 1)  # unit == episodes
        self._episodes_debugged = 0
        logger.debug(f"Initializing environment {hex(id(self))}")

        # Indicator variable to determine if environment has ever been reset.
        self._has_reset = False
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

        # FIXME: Perform env setup depending on if the env is used for eval/train.
        # Indicates whether the environment was created for evaluation purposes.
        self._is_eval_env = config.get("is_evaluation_env", False)
        eval_duration = config.get("evaluation_duration")
        if self.num_workers != 0:
            if eval_duration and not (eval_duration / self.num_workers).is_integer():
                raise ValueError(
                    f"The `evaluation_duration` ({eval_duration}) must be evenly "
                    f"divisible by the `num_workers` ({self.num_workers}.)"
                )
            # Indicates how many rounds of evaluation will be run using this environment.
            self._total_eval_rounds = (
                eval_duration / self.num_workers if eval_duration else 0
            )
        else:
            # Eval will be run in the algorithm process, so no need to divide.
            self._total_eval_rounds = eval_duration if eval_duration else 0

        self._current_eval_round = 1
        # Incremented on each call to `RenderEnv.on_evaluate_start()` callback, via the
        # `_increment_evaluation_iterations()` helper method.
        self._num_eval_iters = 0

        self.fire_scenarios = config.get("scenarios", None)
        # Set the max number of steps that the environment can take before truncation
        # self.spec.max_episode_steps = 1000
        self.spec = EnvSpec(
            id="ReactiveHarness-v0",
            entry_point="simharness2.environments.reactive:ReactiveHarness",
            max_episode_steps=2000,
        )
        # Track the number of timesteps that have occurred within an episode.
        self.timesteps = 0

        # Store parameters relevant to the agent; for use in `step()`, `reset()`, etc.
        self.agent_speed: int = config.get("agent_speed")
        # NOTE: Assume convention of agent_pos[0] == y (row), agent_pos[1] == x (col).
        self.agent_pos: List[int]
        # FIXME: Default value (ie. [15, 15]) should be set in the config file.
        self.initial_agent_pos: List[int] = config.get("initial_agent_pos", [15, 15])
        self.randomize_initial_agent_pos: bool = config.get(
            "randomize_initial_agent_pos", False
        )

        action_space_partial: partial = config.get("action_space_partial")
        # Ensure the provided `action_space_partial` has a `func` attribute.
        if not isinstance(action_space_partial, partial):
            raise TypeError(
                f"Expected `action_space_partial` to be an instance of "
                f"`functools.partial`, but got {type(action_space_partial)}."
            )

        super().__init__(
            sim=config.get("sim"),
            movements=config.get("movements"),
            interactions=config.get("interactions"),
            attributes=config.get("attributes"),
            normalized_attributes=config.get("normalized_attributes"),
            action_space_cls=action_space_partial.func,
            deterministic=config.get("deterministic"),
            benchmark_sim=config.get("benchmark_sim"),
        )

        self._log_env_init()

        # Set the agent's initial position on the map
        self._set_agent_pos_for_episode_start()

        # If provided, construct the class used to monitor this `ReactiveHarness` object.
        self._setup_harness_analytics(
            harness_analytics_partial=config.get("harness_analytics_partial")
        )

        # If provided, construct the class used to perform reward calculation.
        self._setup_reward_cls(reward_cls_partial=config.get("reward_cls_partial"))

        # After every agent action, store the movement and interaction that were taken.
        self._latest_movement: int = None
        self._latest_interaction: int = None
        # If the square the agent is on is "empty", this is set to True.
        # If the agent places a mitigation, this is set to True.
        self.mitigation_placed: bool = False
        # If the agent attempts to move out of bounds, this is set to True.
        self._moved_off_map = False

    def set_trial_results_path(self, path: str) -> None:
        """Set the path to the directory where (tune) trial results will be stored."""
        self._trial_results_path = path

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:  # noqa
        # TODO: Refactor to better utilize `RLHarness` ABC, or update the API.
        self._do_one_agent_step(action)  # alternatively, self._step_agent(action)

        if self.harness_analytics:
            self.harness_analytics.update_after_one_agent_step(
                timestep=self.timesteps,
                movement=self._latest_movement,
                interaction=self._latest_interaction,
                agent_pos=self.agent_pos,
                moved_off_map=self._moved_off_map,
            )

        # NOTE: `sim_run` indicates if `FireSimulation.run()` was called. This helps
        # indicate how to calculate the reward for the current timestep.
        sim_run = self._do_one_simulation_step()  # alternatively, self._step_simulation()

        if sim_run and self.harness_analytics:
            self.harness_analytics.update_after_one_simulation_step(
                timestep=self.timesteps
            )

        # TODO(afennelly): Need to handle truncation properly. For now, we assume that
        # the episode will never be truncated, but this isn't necessarily true.
        truncated = False
        # FIXME `fire_status` is set in `FireSimulation.__init__()`, while `active` is
        # set in `FireSimulation.run()`, so attribute DNE prior to first call to `run()`.
        # terminated = self.sim.fire_status == GameStatus.QUIT
        # The simulation has not yet been run via `run()`
        if self.sim.elapsed_steps == 0:
            terminated = False
        else:
            terminated = not self.sim.active

        # Calculate the reward for the current timestep
        # TODO pass `terminated` into `get_reward` method
        reward = self.reward_cls.get_reward(self.timesteps, sim_run)

        # TODO account for below updates in the reward_cls.calculate_reward() method
        # "End of episode" reward
        if terminated:
            reward += 10

        if self.harness_analytics:
            self.harness_analytics.update_after_one_harness_step(
                sim_run, terminated, reward, timestep=self.timesteps
            )

        self.timesteps += 1  # increment AFTER method logic is performed (convention).

        # FIXME: When in debug mode, always write sim.fire_map to file.
        if terminated and self._debug_mode:
            outdir = self._trial_results_path
            subdir = "eval" if self._is_eval_env else "train"
            savedir = os.path.join(outdir, "fire_map", subdir)
            os.makedirs(savedir, exist_ok=True)
            # Make file name used for saving the fire map
            episodes_total = self.harness_analytics.episodes_total
            fname = f"{os.getpid()}-episode-{episodes_total}-fire_map"
            save_path = os.path.join(savedir, fname)
            logger.info(f"Saving fire map to {save_path}...")
            np.save(save_path, self.sim.fire_map)

        return self.state, reward, terminated, truncated, {}

    def _do_one_agent_step(self, action: np.ndarray) -> None:
        """Move the agent and interact with the environment.

        Within this method, the movement and interaction that the agent takes are stored
        in `self._latest_movement` and `self._latest_interaction`, respectively. If this
        movement is not "none", then the agent's position stored in `self.agent_pos` is
        updated, as well as the corresponding agent stored in the `self.sim.agents` dict.

        If the (new) space occupied by the agent is `UNBURNED` and the interaction is not
        "none", then the agent will place a mitigation on the map, and
        `self.mitigation_placed` is set to True. Otherwise, `self.mitigation_placed` is
        set to False.

        Arguments:
            action: An ndarray provided by the agent to update the environment state.
        """
        # Parse the movement and interaction from the action, and store them.
        self._latest_movement, self._latest_interaction = self._parse_action(action)

        # Update agent location on map
        if self.movements[self._latest_movement] != "none":
            # NOTE: `self.agent_pos` is updated in `_update_agent_position()`.
            self._update_agent_position()

        interact = self.interactions[self._latest_interaction] != "none"
        # Ensure that mitigations are only placed on squares with `UNBURNED` status
        if self._agent_pos_is_unburned() and interact:
            # NOTE: `self.mitigation_placed` is updated in `_update_mitigation()`.
            self._update_mitigation()

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

    def _update_agent_position(self) -> None:
        """Update the agent's position on the map by performing the provided movement."""
        # Store agent's current position in a temporary variable to avoid overwriting it.
        temp_agent_pos = self.agent_pos.copy()
        map_boundary = self.sim.config.area.screen_size[0] - 1

        # Update the agent's position based on the provided movement.
        movement_str = self.movements[self._latest_movement]
        # First, check that the movement string is valid.
        if movement_str not in ["up", "down", "left", "right"]:
            raise ValueError(f"Invalid movement string provided: {movement_str}.")
        # Then, ensure that the agent will not move off the map.
        elif movement_str == "up" and not self.agent_pos[0] == 0:
            temp_agent_pos[0] -= 1
        elif movement_str == "down" and not self.agent_pos[0] == map_boundary:
            temp_agent_pos[0] += 1
        elif movement_str == "left" and not self.agent_pos[1] == 0:
            temp_agent_pos[1] -= 1
        elif movement_str == "right" and not self.agent_pos[1] == map_boundary:
            temp_agent_pos[1] += 1
        # Movement invalid from current pos, so the agent movement will be ignored.
        # Depending on `self.reward_cls`, the agent may receive a small penalty.
        else:
            # Inform caller that the agent cannot move in the provided direction.
            logger.debug(f"Agent cannot move {movement_str} from {self.agent_pos}.")
            logger.debug("Setting `self._moved_off_map = True`...")
            self._moved_off_map = True

        # Store the updated agent position.
        self.agent_pos = temp_agent_pos

        # Update the Simulation with new agent position (s).
        # NOTE: We assume the single-agent case here, so agent ID == 0.
        # NOTE: Elements of `point` should follow (column, row, agent_id) convention.
        point = [self.agent_pos[1], self.agent_pos[0], self.sim_agent_id]
        self.sim.update_agent_positions([point])

    def _agent_pos_is_unburned(self) -> bool:
        """Returns true if the space occupied by the agent has `BurnStatus.UNBURNED`."""
        pos_0, pos_1 = self.agent_pos[0], self.agent_pos[1]
        return self.sim.fire_map[pos_0, pos_1] == BurnStatus.UNBURNED

    def _update_mitigation(self) -> None:
        """Interact with the environment by performing the provided interaction."""
        # Perform interaction on new space
        sim_interaction = self.harness_to_sim[self._latest_interaction]
        # NOTE: Elements of `mitigation_update` should follow (column, row, agent_id)
        # convention.
        mitigation_update = (self.agent_pos[1], self.agent_pos[0], sim_interaction)
        self.sim.update_mitigation([mitigation_update])
        # Store indicator that a mitigation was placed
        self.mitigation_placed = True

    def _do_one_simulation_step(self) -> bool:
        """Step the simulation forward one timestep, depending on `self.agent_speed`."""
        run_sim = self.timesteps % self.agent_speed == 0
        # The simulation WILL NOT be run every step, unless `self.agent_speed` == 1.
        if run_sim:
            self._run_simulation()
        # Prepare the observation that is returned in the `self.step()` method.
        self._update_state()
        return run_sim

    def _run_simulation(self):
        """Run the simulation (s) for one timestep."""
        if self.benchmark_sim:
            self.benchmark_sim.run(1)

        self.sim.run(1)

    def _update_state(self):
        """Modify environment's state to contain updates from the current timestep."""
        # Copy the fire map from the simulation so we don't overwrite it.
        fire_map = np.copy(self.sim.fire_map)
        # Update the fire map with the numeric identifier for the agent.
        fire_map[self.agent_pos[0], self.agent_pos[1]] = self.sim_agent_id
        # Modify the state to contain the updated fire map
        fire_map_idx = self.attributes.index("fire_map")
        self.state[..., fire_map_idx] = fire_map

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
        logger.info(f"Resetting environment {hex(id(self))}")
        self.sim.reset()
        # FIXME quick fix to avoid errors if benchmark_sim is not used (ie. None)
        if self.benchmark_sim:
            # reset benchmark simulation
            self.benchmark_sim.reset()

        # Reset the `ReactiveHarnessData` to initial conditions, if it exists.
        if self.harness_analytics:
            render = self._should_render if hasattr(self, "_should_render") else False
            self.harness_analytics.reset(env_is_rendering=render)

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
        point = [self.agent_pos[1], self.agent_pos[0], self.sim_agent_id]
        self.sim.update_agent_positions([point])

        # NOTE: `self.num_burned` is not currently used in the reward calculation.
        # self.num_burned = 0 FIXME include once we modularize the reward function
        self.timesteps = 0

        self._log_env_reset()
        self._has_reset = True

        # Reset attributes that help track the agent's actions.
        self._latest_movement: int = None
        self._latest_interaction: int = None
        # If the square the agent is on is "empty", this is set to True.
        # If the agent places a mitigation, this is set to True.
        self.mitigation_placed: bool = False
        # If the agent attempts to move out of bounds, this is set to True.
        self._moved_off_map = False

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
                self.sim.config.area.screen_size[0],
                self.sim.config.area.screen_size[0],
            )
        )

        # Place the agent on the fire map using the agent ID.
        nonsim_data["fire_map"][self.agent_pos[0], self.agent_pos[1]] = self.sim_agent_id
        # FIXME the below line has no dependence on `nonsim_data`; needs to be moved.
        # FIXME Why are we placing a fireline at the agents position here?
        # self.sim.update_mitigation([(self.agent_pos[1], self.agent_pos[0], 3)])

        return nonsim_data

    def render(self):  # noqa
        self.sim.rendering = True

    def _configure_env_rendering(self, should_render: bool) -> None:
        """Configure the environment's `FireSimulation` to be rendered (or not).

        If the simulation should be rendered, then the `headless` parameter in the
        simulation's config (file) should be set to `False`, enabling the usage of pygame.

        Additionally, the environment's `_should_render` attribute is set to ensure
        that rendering is active when desired. This is especially important when the
        number of eval episodes, specified via `evaluation.evaluation_duration`, is >1.
        """
        sim_data = self.sim.config.yaml_data
        sim_data["simulation"]["headless"] = not should_render

        # Update simulation's config attribute.
        logger.info("Updating the `self.sim.config` with new `Config` object...")
        self.sim.config = Config(config_dict=sim_data)

        # Reset the simulation to ensure that the new config is used.
        logger.info(f"Resetting `self.sim` to configure rendering == {should_render}.")
        self.sim.reset()

        # Update the simulation's rendering attribute to match the provided value.
        if should_render:
            logger.info("Setting SDL_VIDEODRIVER environment variable to 'dummy'...")
            os.environ["SDL_VIDEODRIVER"] = "dummy"

        self.sim.rendering = should_render

        # Indicate whether the environment's `FireSimulation` should be rendered.
        self._should_render = should_render

    def _increment_evaluation_iterations(self) -> None:
        """Increment the number of evaluation iterations that have been run."""
        self._num_eval_iters += 1

    def _set_agent_pos_for_episode_start(self):
        """Set the agent's initial position in the map for the start of the episode."""
        if self.randomize_initial_agent_pos:
            self.agent_pos = self.np_random.integers(
                0, self.sim.config.area.screen_size[0], size=2, dtype=int
            )
        else:
            # TODO(afennelly): Verify initial_agent_pos is within the bounds of the map
            self.agent_pos = self.initial_agent_pos

    def _log_env_init(self):
        """Log information about the environment that is being initialized."""
        if self._is_eval_env:
            i, j = self.worker_idx, self.vector_idx
            logger.warning(f"Object {hex(id(self))}: index (i+1)*(j+1) == {(i+1)*(j+1)}")

        if not self._debug_mode:
            return

        # TODO: What log level should we use here?
        logger.info(f"Object {hex(id(self))}: worker_index: {self.worker_idx}")
        logger.info(f"Object {hex(id(self))}: vector_index: {self.vector_idx}")
        logger.info(f"Object {hex(id(self))}: num_workers: {self.num_workers}")
        logger.info(f"Object {hex(id(self))}: is_remote: {self.is_remote}")

    def _log_env_reset(self):
        """Log information about the environment that is being reset."""
        if not self._debug_mode or self._episodes_debugged > self._debug_duration:
            return

        # TODO: What log level should we use here?
        for idx, feat in enumerate(self.attributes):
            low, high = self._low[..., idx].min(), self._high[..., idx].max()
            obs_min = round(self.state[..., idx].min(), 2)
            obs_max = round(self.state[..., idx].max(), 2)
            # Log lower bound of the (obs space) and max returned obs for each attribute.
            logger.info(f"{feat} LB: {low}, obs min: {obs_min}")
            # Log upper (lower) bounds of the returned observations for each attribute.
            logger.info(f"{feat} UB: {high}, obs max: {obs_max}")

        # Increment the number of episodes that have been debugged.
        self._episodes_debugged += 1

    def _setup_harness_analytics(self, harness_analytics_partial: partial) -> None:
        """Instantiates the harness_analytics used to monitor this object.

        Arguments:
            harness_analytics_partial: A `functools.partial` object that indicates the
            top-level class that will be used to monitor the `ReactiveHarness` object.
            The user is expected to provide the `sim_data_partial` keyword argument,
            along with a valid value.

        Raises:
            TypeError: If `harness_analytics_partial.keywords` does not contain a
            `sim_data_partial` key with value of type `functools.partial`.

        """
        self.harness_analytics: ReactiveHarnessAnalytics
        if harness_analytics_partial:
            try:
                self.harness_analytics = harness_analytics_partial(
                    sim=self.sim, benchmark_sim=self.benchmark_sim
                )
            except TypeError as e:
                raise e
        else:
            self.harness_analytics = None

    def _setup_reward_cls(self, reward_cls_partial: partial) -> None:
        """Instantiates the reward class used to perform reward calculation each episode.

        This method must be called AFTER `self._setup_harness_analytics()`, as the reward
        class requires `self.harness_analytics` to be passed as an argument to its
        constructor.

        Arguments:
            reward_cls_partial: A `functools.partial` object that indicates the reward
                class that will be used to perform reward calculation after each timestep
                in an episode.

        Raises:
            TypeError: If `harness_analytics_partial.keywords` does not contain a
                `sim_data_partial` key with value of type `functools.partial`.
            AttributeError: If `self` does not have a `harness_analytics` attribute. See
            the above message for more details.

        """
        self.reward_cls: BaseReward
        if reward_cls_partial:
            try:
                self.reward_cls = reward_cls_partial(
                    harness_analytics=self.harness_analytics
                )
            except TypeError as e:
                raise e
            except AttributeError as e:
                raise e
        else:
            self.reward_cls = None
