"""ReactiveHarness with support for mutiple agents operating simulanteously.

This file contains the environment file for `MARLReactiveHarness` which is an environment
with multiple agents operating at the same time within the same environment. The code
is very similar to the single agent case, just multiplied for each agents action. Agents
can be monogomous or heterogenous depending on the training run - meaning agents can
have the same speed/abilities or different.

The reward function used is configurable depending on the fire manager intent displayed
within the training config and corresponding reward class.
"""
import logging
import os
from collections import OrderedDict as ordered_dict
from functools import partial
from typing import Any, Dict, List, Optional, OrderedDict, Tuple
from dataclasses import dataclass, replace

import numpy as np
from gymnasium import spaces
from gymnasium.envs.registration import EnvSpec
from ray.rllib.env.env_context import EnvContext
from simfire.enums import BurnStatus
from simfire.utils.config import Config

from simharness2.analytics.harness_analytics import ReactiveHarnessAnalytics
from simharness2.environments.rl_harness import RLHarness
from simharness2.rewards.base_reward import BaseReward

# FIXME: Update logger configuration.
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
handler.setFormatter(
    logging.Formatter("%(asctime)s\t%(levelname)s %(filename)s:%(lineno)s -- %(message)s")
)
logger.addHandler(handler)
logger.propagate = False


@dataclass
class ReactiveAgent:
    # NOTE: `agent_speed` ommitted, only used within `_do_one_simulation_step`
    # Attrs that should be specified on initialization
    agent_id: str  # ex: "agent_0", "dozer_0", "handcrew_0", "ff_0", etc.
    sim_id: int  # should be contained within sim.agents.keys()
    initial_position: Tuple[int, int]

    # Attributes with default values
    latest_movement: Optional[int] = None
    latest_interaction: Optional[int] = None
    mitigation_placed: bool = False
    moved_off_map: bool = False

    def __post_init__(self):
        self.current_position = self.initial_position
        # x,y pos, where (0,0) is top-left corner and (max_x, max_y) is bottom-right
        self.x, self.y = self.current_position
        self.row, self.col = self.y, self.x

        # Store the movement and interaction for the current timestep
        self.latest_movement: int = None
        self.latest_interaction: int = None
        # If the agent places a mitigation, this is set to True.
        self.mitigation_placed: bool = False
        # If the agent attempts to move out of bounds, this is set to True.
        self.moved_off_map: bool = False

        # actions: np.ndarray
        # reward: float = 0

    def reset(self):
        self.current_position = self.initial_position
        self.reward = 0

    # def move(self, env: np.ndarray, direction: int) -> bool:
    #     """Moves the agent in the given direction if possible."""
    #     current_x, current_y = self.current_position
    #     dx, dy = self.actions[direction]
    #     next_x, next_y = current_x + dx, current_y + dy

    #     if env[next_y][next_x] == "_":
    #         self.current_position = (next_x, next_y)
    #         return True
    #     else:
    #         return False


class MARLReactiveHarness(RLHarness):  # noqa: D205,D212,D415
    """
    ### Description
    Environment which potrays the case where a fire has already started and we are
    deploying our resources to best mitigate the damage. Multiple agents are interacting
    at once with the environment in a collaborative manner.

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

    def __init__(self, config: EnvContext) -> None:
        """See RLHarness (parent/base class)."""
        # NOTE: We don't set a default value in `config.get` for required arguments.

        # FIXME Most, if not all, of these can be moved into the RLHarness.
        # TODO Should we make an RLlibHarness class to handle all these extras?

        # Indicates that environment information should be logged at various points.
        self._set_debug_options(config)

        self._store_env_context(config)

        # FIXME: Perform env setup depending on if the env is used for eval/train.
        # Indicates whether the environment was created for evaluation purposes.
        self._is_eval_env = config.get("is_evaluation_env", False)
        if self._is_eval_env:
            self._prepare_eval_env(config)
        else:
            self._prepare_train_env(config)

        # Set the max number of steps that the environment can take before truncation
        # self.spec.max_episode_steps = 1000
        self.spec = EnvSpec(
            id="MARLReactiveHarness-v0",
            entry_point="simharness2.environments.reactive_marl:MARLReactiveHarness",
            max_episode_steps=2000,
        )
        # Track the number of timesteps that have occurred within an episode.
        self.timesteps: int = 0

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
            num_agents=config.get("num_agents", 1),
        )

        self._log_env_init()

        # NOTE: only used in `_do_one_simulation_step`, so keep as harness attr
        self.agent_speed: int = config.get("agent_speed")
        # Spawn the agent (s) that will interact with the simulation
        logger.debug("Spawning agents...")
        agent_init_method = config.get("agent_initialization_method", "automatic")
        if agent_init_method == "manual":
            agent_init_positions = config.get("initial_agent_positions", None)
            if agent_init_positions is None:
                raise ValueError(
                    "Must provide 'initial_agent_positions' when using 'manual' agent initialization method."
                )
            self._spawn_agents(method="manual", pos_list=agent_init_positions)
        elif agent_init_method == "automatic":
            self._spawn_agents(method="random")
        else:
            raise ValueError(
                "Invalid agent initialization method. Must be either 'automatic' or 'manual'."
            )

        breakpoint()
        # If provided, construct the class used to monitor this `ReactiveHarness` object.
        # FIXME Move into RLHarness

        self._setup_harness_analytics(
            harness_analytics_partial=config.get("harness_analytics_partial")
        )

        # If provided, construct the class used to perform reward calculation.
        self._setup_reward_cls(reward_cls_partial=config.get("reward_cls_partial"))

    def _set_debug_options(self, config: EnvContext):
        """Set the debug options for the environment."""
        self._debug_mode = config.get("debug_mode", False)
        self._debug_duration = config.get("debug_duration", 1)  # unit == episodes
        self._episodes_debugged = 0
        logger.debug(f"Initializing environment {hex(id(self))}")

    def _store_env_context(self, config: EnvContext):
        """Store the environment context for later use."""
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

    def _prepare_eval_env(self, config: EnvContext):
        """Prepare the environment for evaluation purposes."""
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

    def _prepare_train_env(self, config: EnvContext):
        """Prepare the environment for training purposes."""
        # TODO Add any training-specific logic here
        pass

    def set_trial_results_path(self, path: str) -> None:
        """Set the path to the directory where (tune) trial results will be stored."""
        self._trial_results_path = path

    def step(
        self, actions: Dict[str, np.ndarray]
    ) -> Tuple[
        Dict[str, np.ndarray],
        Dict[str, float],
        Dict[str, bool],
        Dict[str, bool],
        Dict[str, Dict[str, Any]],
    ]:  # noqa FIXME
        # TODO: Refactor to better utilize `RLHarness` ABC, or update the API.

        # TODO: Create and use `self._agent_ids` to store agent IDs.
        # Additionally, when we update the position of an agent on the simulation
        # map, we will specify the int for agent ID. So, we can store `_agent_ids`
        # as strs and map to unique integer agent IDs for the sim to use.
        # This should be done when the environment is initialized?

        # TODO: Can we parallelize this method? If so, how? I'm not sure if that
        # will make sense wrt updating the sim, etc.?
        movements, interactions = {}, {}
        for agent_id, agent in self.agents.items():
            agent.latest_movement, agent.latest_interaction = self._do_one_agent_step(
                agent_id, actions[agent_id]
            )  # alternatively, self._step_agent(action)

        if self.harness_analytics:
            # Naive approach: Iterate over each agent and do (roughly) same as SARL case.
            for agent_id_num in range(self.num_agents):
                self.harness_analytics.update_after_one_agent_step(
                    timestep=self.timesteps,
                    movement=movements[agent_id],
                    interaction=interactions[agent_id],
                    agent_pos=self.agent_pos[agent_id_num],
                    moved_off_map=self._moved_off_map[agent_id_num],
                    agent_id=f"agent_{agent_id_num}",
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

        new_obs, rewards, truncateds, terminateds, infos = {}, {}, {}, {}, {}
        truncs = set()
        terms = set()
        for id_num in range(self.num_agents):
            agent_id = f"agent_{id_num}"
            new_obs[agent_id] = self.state
            rewards[agent_id] = reward
            truncateds[agent_id] = truncated
            terminateds[agent_id] = terminated
            infos[agent_id] = {}

            if truncated:
                truncs.add(id_num)
            if terminated:
                terms.add(id_num)

        terminateds["__all__"] = len(truncs) == self.num_agents
        truncateds["__all__"] = len(terms) == self.num_agents

        self.timesteps += 1  # increment AFTER method logic is performed (convention).

        return new_obs, rewards, terminateds, truncateds, infos

    def _do_one_agent_step(self, agent_id_num: int, action: np.ndarray) -> None:
        """Move the agent and interact with the environment.

        Within this method, the movement and interaction that the agent takes are stored
        in `self.latest_movements` and `self.latest_interactions`, respectively. If this
        movement is not "none", then the agent's position on the map is updated and
        stored in `self.agent_pos`.

        Given some arbitrary method that defines whether a space in the simulation is
        empty or not (see `_agent_pos_is_empty_space()`), the value of
        `self.agent_pos_is_empty_space` is updated accordingly. If the space occupied by
        the agent (`self.agent_pos`) is *empty* and the interaction is not "none", then
        the agent will place a mitigation on the map and `self.mitigation_placed` is set
        to True. Otherwise, `self.mitigation_placed` is set to False.


        Data that we want to store after each AGENT step:
        - interaction (via `_parse_action`)
            - connected_mitigation (via `_update_mitigation`)
        - movement (via `_parse_action`)
            - moved_off_map (via `_update_agent_position`)
        - near_fire (calculated within `AgentAnalytics.update`)
        - burn_status (calculated within `AgentAnalytics.update`)

        Additional data needed ONLY when storing all episode data:
        - agent_pos (via `_update_agent_position`)
        - timestep (via `self.timesteps`)

        It seems like an efficient way to store the timestep data would be with a
        namedtuple. I'm looking into more details now.

        Args:
            agent_id_num (int): _description_
            action (np.ndarray): _description_

        Returns:
            _type_: _description_
        """
        # Parse the movement and interaction from the action, and store them.
        movement_id, interaction_id = self._parse_action(action)

        # Update agent location on map
        if self.movements[movement_id] != "none":
            # NOTE: `self.agent_pos` is updated in `_update_agent_position()`.
            self._update_agent_position(agent_id_num, movement_id)

        # Check if there was an interaction already done on this space
        # NOTE: `self.agent_pos_is_empty_space` will be updated in below method.
        is_empty_space = self._agent_pos_is_empty_space(
            agent_id_num
        )  # FIXME do we still need this??

        # Interact with the environment
        interact = self.interactions[interaction_id] != "none"
        if is_empty_space and interact:
            # NOTE: `self.mitigation_placed` is updated in `_update_mitigation()`.
            self._update_mitigation(agent_id_num, interaction_id)

        return movement_id, interaction_id

    def _parse_action(self, action: np.ndarray) -> Tuple[int, int]:
        """Parse the action into movement and interaction."""
        # Handle the MultiDiscrete case (currently used in `ReactiveHarness`)
        if isinstance(self.action_space, spaces.MultiDiscrete):
            return action[0], action[1]
        # Handle the Discrete case (currently used in `ReactiveDiscreteHarness`)
        elif isinstance(self.action_space, spaces.Discrete):
            return action % len(self.movements), int(action / len(self.movements))
        else:
            raise NotImplementedError(f"{self.action_space} is not supported.")

    def _update_agent_position(self, agent_id: str) -> None:
        """Update the agent's position on the map by performing the provided movement."""
        agent = self.agents[agent_id]
        # Store agent's current position in a temporary variable to avoid overwriting it.
        agent_pos = agent.current_position
        temp_agent_pos = agent_pos.copy()
        map_boundary = self.sim.config.area.screen_size[0] - 1

        # Update the agent's position based on the provided movement.
        latest_movement = agent.latest_movement
        movement_str = self.movements[latest_movement]
        # First, check that the movement string is valid.
        if movement_str not in ["up", "down", "left", "right"]:
            raise ValueError(f"Invalid movement string provided: {movement_str}.")
        # Then, ensure that the agent will not move off the map.
        elif movement_str == "up" and not agent_pos[0] == 0:
            temp_agent_pos[0] -= 1
        elif movement_str == "down" and not agent_pos[0] == map_boundary:
            temp_agent_pos[0] += 1
        elif movement_str == "left" and not agent_pos[1] == 0:
            temp_agent_pos[1] -= 1
        elif movement_str == "right" and not agent_pos[1] == map_boundary:
            temp_agent_pos[1] += 1
        # Movement invalid from current pos, so the agent movement will be ignored.
        # Depending on `self.reward_cls`, the agent may receive a small penalty.
        else:
            # Inform caller that the agent cannot move in the provided direction.
            logger.debug(
                f"Agent cannot move {movement_str} from {agent.current_position}."
            )
            logger.debug("Setting `self._moved_off_map = True`...")
            agent.moved_off_map = True

        # Store the updated agent position.
        # TODO: Probably want a setter method for this?
        self.agents[agent_id].current_position = temp_agent_pos

        # Update the Simulation with new agent position (s).
        # NOTE: We assume the single-agent case here, so agent ID == 0.
        # NOTE: Elements of `point` should follow (column, row, agent_id) convention.
        point = [
            self.agents[agent_id].col,
            self.agents[agent_id].row,
            self.agents[agent_id].sim_id,
        ]
        self.sim.update_agent_positions([point])

    def _agent_pos_is_unburned(self, agent_id: str) -> bool:
        """Returns true if the space occupied by the agent has `BurnStatus.UNBURNED`."""
        pos_0, pos_1 = self.agents[agent_id].current_position
        return self.sim.fire_map[pos_0, pos_1] == BurnStatus.UNBURNED

    def _update_mitigation(self, agent_id: str, interaction_id: int) -> None:
        """Interact with the environment by performing the provided interaction."""
        # Perform interaction on new space
        sim_interaction = self.harness_to_sim[interaction_id]
        # NOTE: Elements of `mitigation_update` should be (col, row, id) convention.
        agent = self.agents[agent_id]
        mitigation_update = (agent.col, agent.row, sim_interaction)
        self.sim.update_mitigation([mitigation_update])

    def _do_one_simulation_step(self) -> bool:
        # The simulation WILL NOT be run every step, unless `self.agent_speed` == 1.
        self._run_simulation()
        # Prepare the observation that is returned in the `self.step()` method.
        self._update_state()
        return True

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
        for agent_id in self._agent_ids:
            agent = self.agents[agent_id]
            fire_map[agent.x, agent.y] = agent.sim_id
        # Modify the state to contain the updated fire map
        fire_map_idx = self.attributes.index("fire_map")
        self.state[..., fire_map_idx] = fire_map

    def _nearby_fire(self) -> bool:
        """Check if the agent is adjacent to a space that is currently burning.

        Returns:
            nearby_fire: A boolean indicating if there is a burning space adjacent to the
              agent.
        """
        # TODO: This method MUST be tested to ensure it returns the correct boolean!!

        nearby_fire = [False] * self.num_agents
        for agent_id in range(self.num_agents):
            agent_pos = self.agent_pos[agent_id]
            # Get all squares nearby this agent
            nearby_locs = []
            screen_size = self.sim.config.area.screen_size
            # Get all spaces surrounding agent
            for i in range(agent_pos[0] - 1, agent_pos[0] + 2):
                for j in range(agent_pos[1] - 1, agent_pos[1] + 2):
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

            # Mark if a nearby location is on fire
            for i, j in nearby_locs:
                if self.state[i][j][self.attributes.index("fire_map")] == 1:
                    nearby_fire[agent_id] = True
                    break

        return nearby_fire

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[Any, Any]] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict[Any, Any]]]:  # noqa
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
        self.sim.reset()
        # FIXME quick fix to avoid errors if benchmark_sim is not used (ie. None)
        if self.benchmark_sim:
            # reset benchmark simulation
            self.benchmark_sim.reset()

        # Reset the `ReactiveHarnessData` to initial conditions, if it exists.
        if self.harness_analytics:
            self.harness_analytics.reset()

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

        # Update the Simulation with new agent positions.
        points = []
        for agent_id in range(self.num_agents):
            agent_pos = self.agent_pos[agent_id]
            points.append([agent_pos[1], agent_pos[0], 0])

        self.sim.update_agent_positions(points)

        # NOTE: `self.num_burned` is not currently used in the reward calculation.
        # self.num_burned = 0 FIXME include once we modularize the reward function
        self.timesteps = 0

        self._log_env_reset()

        marl_obs, infos = {}, {}
        for id_num in range(self.num_agents):
            agent_id = f"agent_{id_num}"
            marl_obs[agent_id] = self.state
            infos[agent_id] = {}

        return marl_obs, infos

    def get_nonsim_attribute_bounds(self) -> OrderedDict[str, Dict[str, int]]:  # noqa
        nonsim_min_maxes = ordered_dict()
        # The values in "fire_map" are:
        #   - 0: BurnStatus.UNBURNED
        #   - 1: BurnStatus.BURNING
        #   - 2: BurnStatus.BURNED
        #   - 3: BurnStatus.FIRELINE (if "fireline" in self.interactions)
        #   - 4: BurnStatus.SCRATCHLINE (if "scratchline" in self.interactions)
        #   - 5: BurnStatus.WETLINE (if "wetline" in self.interactions)
        #   - X: self._min_sim_agent_id + self.num_agents (value is set in RLHarness.__init__)
        nonsim_min_maxes["fire_map"] = {
            "min": 0,
            "max": max(self._sim_agent_ids),
        }
        return nonsim_min_maxes

    def get_nonsim_attribute_data(self) -> OrderedDict[str, np.ndarray]:  # noqa
        # TODO(afennelly) Make note that agent is "placed" on the `fire_map`, etc. here.
        # This method is more of a `reset_and_update_nonsim_attribute_data` method.
        nonsim_data = ordered_dict()

        nonsim_data["fire_map"] = np.zeros(
            # FIXME: can just do area.screen_size without indexing?
            (self.sim.config.area.screen_size[0], self.sim.config.area.screen_size[0])
        )

        for agent_id in range(self.num_agents):
            agent_pos = self.agent_pos[agent_id]
            sim_agent_id = self._min_sim_agent_id + agent_id
            # Place the agent on the fire map using the agent ID.
            nonsim_data["fire_map"][agent_pos[1]][agent_pos[0]] = sim_agent_id

        return nonsim_data

    def render(self):  # noqa
        self.sim.rendering = True

    # TODO: Finish code to allow manually specifying agent positions.
    # def _check_start_pos(self, start_pos: Tuple[int, int]) -> bool:
    #     # Check that value is in the correct range
    #     if (
    #         start_pos[0] < 0
    #         or start_pos[0] >= self.sim.config.area.screen_size[0]
    #         or start_pos[1] < 0
    #         or start_pos[1] >= self.sim.config.area.screen_size[0]
    #     ):
    #         return False

    #     for pos in self.agent_pos:
    #         if np.array_equal(pos, start_pos):
    #             return False

    #     return True

    # def _validate_position(self, x, y):
    #     """Check whether (x,y) is within the bounds of the environment."""
    #     return all([x >= 0, x < self.width, y >= 0, y < self.height])

    # def _check_collision(self, pos1, pos2):
    #     """Check whether two positions overlap."""
    #     return pos1[0] == pos2[0] and pos1[1] == pos2[1]

    # def _spawn_agents(self, method='random', pos_list=None):
    #     """Spawn agents according to the given method and position list."""

    #     # Initialize empty lists for holding agent objects and positions
    #     self.agents = []
    #     self.agent_positions = {}

    #     if method == 'manual':
    #         # Validate and assign positions from the input list
    #         assert len(pos_list) == len(self.agent_ids), \
    #             f"Number of positions ({len(pos_list)}) does not match number of agents ({len(self.agent_ids)})."

    #         for i, pos in enumerate(pos_list):
    #             assert len(pos) == 3, f"Position {i} has invalid length ({len(pos)}, expected 3)"

    #             agent_id, x, y = pos
    #             assert agent_id in self.agent_ids, f"Agent ID '{agent_id}' is not recognized."

    #             assert self._validate_position(x, y), f"Position {pos} is out of bounds."

    #             for j in range(i+1, len(pos_list)):
    #                 assert not self._check_collision(pos, pos_list[j]), f"Position collision detected between {pos} and {pos_list[j]}."

    #             self.agents.append(ReactiveAgent(agent_id))
    #             self.agent_positions[agent_id] = (x, y)

    # if method == "manual":
    #     if len(pos_list) < self.num_agents:
    #         # Pad with default positions
    #         num_missing = self.num_agents - len(pos_list)
    #         logger.warning(
    #             "%d manual agent position(s) provided; padding with %d defaults.",
    #             len(pos_list),
    #             num_missing,
    #         )
    #         pos_list += [(f"default{i}", 0, 0) for i in range(num_missing)]
    #     elif len(pos_list) > self.num_agents:
    #         # Truncate the list
    #         num_extra = len(pos_list) - self.num_agents
    #         logger.warning(
    #             "%d manual agent position(s) provided; ignoring %d extra.",
    #             len(pos_list),
    #             num_extra,
    #         )
    #         pos_list = pos_list[: self.num_agents]

    def _spawn_agents(self, method: str = "random", pos_list: List = None):
        """Initialize agent positions."""
        self.agents: Dict[str, ReactiveAgent] = {}
        max_sim_agent_id = self._min_sim_agent_id + self.num_agents
        sim_agent_ids = np.arange(start=self._min_sim_agent_id, stop=max_sim_agent_id)
        logger.debug(f"sim_agent_ids: {sim_agent_ids}")
        # Use the user-provided agent positions to initialize the agents on the map.
        if method == "manual":
            # NOTE: The provided pos_list must be the same length as the number of agents
            # TODO: Allow option to randomly generate any "missing" agent positions.
            if len(pos_list) != self.num_agents:
                raise ValueError(
                    f"Expected {self.num_agents} agent positions; got {len(pos_list)}."
                )

            # FIXME: We assume provided pos are valid wrt map dims and agent collisions.
            # FIXME: Finish logic HERE to create `self.agents` dict
            raise NotImplementedError  # adding so I don't forget!
            # for agent_info, sim_id in zip(pos_list, sim_agent_ids):
            #     agent_str, x, y = agent_info
            #     agent = ReactiveAgent(agent_str, sim_id, (x, y))
            #     self.agents[agent_str] = agent

        # Generate random agent locations for the start of the episode.
        elif method == "random":
            agent_locs = self.np_random.choice(
                # FIXME: Not robust for rectangular maps
                np.arange(self.sim.fire_map.size),
                size=(self.num_agents, 2),
                replace=False,
            )  # .reshape(-1, 2)
            agent_locs = np.unravel_index(agent_locs, self.sim.fire_map.size)
            # Populate the `self.agents` dict with `ReactiveAgent` object (s).
            for agent_str, sim_id, loc in zip(self._agent_ids, sim_agent_ids, agent_locs):
                agent = ReactiveAgent(agent_str, sim_id, tuple(loc))
                self.agents[agent_str] = agent
        # This should be caught within the init. To be safe, also raise error here.
        else:
            raise NotImplementedError(f"Agent spawn method {method} not implemented.")

    # def _set_agent_pos_for_episode_start(self):
    #     """Set the agent's initial position in the map for the start of the episode."""
    #     for agent_id in self._agent_ids:
    #         valid_pos = False
    #         # Keep looping until we get a valid position
    #         while not valid_pos:
    #             random_pos = self.np_random.integers(
    #                 0, self.sim.config.area.screen_size, size=2, dtype=int
    #             )

    #             valid_pos = self._check_start_pos(random_pos)

    #         self.agent_pos[agent_id] = random_pos

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

    def _setup_harness_analytics(self, analytics_partial: partial) -> None:
        """Instantiates the `harness_analytics` used to monitor this `ReactiveHarness` obj.

        Arguments:
            analytics_partial:
                A `functools.partial` object that indicates the top-level
                class that will be used to monitor the `ReactiveHarness` object. The user
                is expected to provide the `sim_data_partial` keyword argument, along
                with a valid value.

        Raises:
            TypeError: If `harness_analytics_partial.keywords` does not contain a
            `sim_data_partial` key with value of type `functools.partial`.

        """
        self.harness_analytics: ReactiveHarnessAnalytics
        if analytics_partial:
            try:
                self.harness_analytics = analytics_partial(
                    sim=self.sim,
                    benchmark_sim=self.benchmark_sim,
                    agents=self.agents,
                )
            except Exception as e:
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
            AttributeError: If `self` does not have a `harness_analytics` attribute.
                See the above message for more details.

        """
        self.reward_cls: BaseReward
        if reward_cls_partial:
            try:
                self.reward_cls = reward_cls_partial(
                    harness_analytics=self.harness_analytics
                )
            except Exception as e:
                raise e
        else:
            self.reward_cls = None
