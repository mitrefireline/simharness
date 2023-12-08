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
import math
import copy

import numpy as np
from gymnasium import spaces
from gymnasium.envs.registration import EnvSpec
from ray.rllib.env.env_context import EnvContext
from simfire.enums import BurnStatus
from simfire.utils.config import Config

from simharness2.analytics.harness_analytics import ReactiveHarnessAnalytics
from simharness2.environments.rl_harness import RLHarness
from simharness2.rewards.base_reward import BaseReward
from simharness2.agents import ReactiveAgent

# FIXME: Update logger configuration.
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
handler.setFormatter(
    logging.Formatter("%(asctime)s\t%(levelname)s %(filename)s:%(lineno)s -- %(message)s")
)
logger.addHandler(handler)
logger.propagate = False


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

        # Spawn the agent (s) that will interact with the simulation
        logger.debug(f"Creating {self.num_agents} agent (s)...")
        agent_init_method = config.get("agent_initialization_method", "automatic")
        if agent_init_method == "manual":
            agent_init_positions = config.get("initial_agent_positions", None)
            if agent_init_positions is None:
                raise ValueError(
                    "Must provide 'initial_agent_positions' when using 'manual' agent initialization method."
                )
            self._create_agents(method="manual", pos_list=agent_init_positions)
        elif agent_init_method == "automatic":
            self._create_agents(method="random")
        else:
            raise ValueError(
                "Invalid agent initialization method. Must be either 'automatic' or 'manual'."
            )
            # NOTE: only used in `_do_one_simulation_step`, so keep as harness attr
        self.agent_speed: int = config.get("agent_speed")

        # If provided, construct the class used to monitor this `ReactiveHarness` object.
        # FIXME Move into RLHarness
        analytics_partial = config.get("harness_analytics_partial")
        self._setup_harness_analytics(analytics_partial)

        # If provided, construct the class used to perform reward calculation.
        self._setup_reward_cls(reward_cls_partial=config.get("reward_cls_partial"))

        # If the agent(s) places an effective mitigation (not placed in already damaged/mitigated square), this is set to True.
        # FIXME Have this tracked across all of the agents
        self.true_mitigation_placed: bool = False

        # Bool to toggle the ability to terminate the agent simulation early if at the current timestep of the agent simulation
        #   , the agents have caused more burn damage (burned + burning) than the final state of the benchmark fire map
        # FIXME Have this value set in the configs
        self._terminate_if_greater_damage = True

        if self.benchmark_sim:
            #Validate that benchmark and sim match seeds
            assert self.sim.get_seeds() == self.benchmark_sim.get_seeds()

            #create static list to store the episode benchsim firemaps
            self.max_bench_length = 600
            self.bench_firemaps = [0] * self.max_bench_lenght

            #run the first benchmark sim to generate the benchmark sim firemaps and metrics for this episode
            self._run_benchmark()


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
        self, action_dict: Dict[Any, np.ndarray]
    ) -> Tuple[
        Dict[Any, np.ndarray],
        Dict[Any, float],
        Dict[Any, bool],
        Dict[Any, bool],
        Dict[Any, Dict[Any, Any]],
    ]:  # noqa FIXME
        # TODO: Refactor to better utilize `RLHarness` ABC, or update the API.
        # TODO: Can we parallelize this method? If so, how? I'm not sure if that
        # will make sense wrt updating the sim, etc.?
        for agent_id, agent in self.agents.items():
            self._do_one_agent_step(agent, action_dict[agent_id])

        if self.harness_analytics:
            self.harness_analytics.update_after_one_agent_step(
                timestep=self.timesteps, agents=self.agents, true_mitigation_placed = self.true_mitigation_placed
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
        # The simulation has not yet been run via `run()`
        if self.sim.elapsed_steps == 0:
            terminated = False
        else:
            terminated = not self.sim.active

        # Calculate the reward for the current timestep
        # TODO pass `terminated` into `get_reward` method
        # FIXME: Update reward for MARL case!!
        # TODO: Give each agent the "same" simple reward for now.
        reward = self.reward_cls.get_reward(self.timesteps, sim_run)

        # Terminate episode early if burn damage in Agent Sim is larger than final bench fire map
        if self.benchmark_sim:
            if self._terminate_if_greater_damage:
                total_area = self.harness_analytics.sim_analytics.sim.config.area.screen_size[0] ** 2

                sim_damaged_total = self.harness_analytics.sim_analytics.data.burned + self.harness_analytics.sim_analytics.data.burning

                benchsim_damaged_total = total_area - self.harness_analytics.benchmark_sim_analytics.data.unburned

                if sim_damaged_total > benchsim_damaged_total:
                    terminated = True
                    # TODO potentially add a static negative penalty for making the fire worse

        # TODO account for below updates in the reward_cls.calculate_reward() method
        # "End of episode" reward
        #if terminated:
            #reward += 10

        if self.harness_analytics:
            self.harness_analytics.update_after_one_harness_step(
                sim_run, terminated, reward, timestep=self.timesteps
            )

        new_obs, rewards, truncateds, terminateds, infos = {}, {}, {}, {}, {}
        truncs = set()
        terms = set()
        for agent_id, agent in self.agents.items():
            new_obs[agent_id] = self.state
            rewards[agent_id] = reward  # FIXME !!
            truncateds[agent_id] = truncated
            terminateds[agent_id] = terminated
            infos[agent_id] = {}

            if truncated:
                truncs.add(agent_id)
            if terminated:
                terms.add(agent_id)

        terminateds["__all__"] = len(truncs) == self.num_agents
        truncateds["__all__"] = len(terms) == self.num_agents

        self.timesteps += 1  # increment AFTER method logic is performed (convention).

        return new_obs, rewards, terminateds, truncateds, infos

    # NOTE: if passing `agent` doesn't persist updates, pass `agent_id` instead.
    def _do_one_agent_step(self, agent: ReactiveAgent, action: np.ndarray) -> None:
        """Move the agent and interact with the environment.

        FIXME: Below details need to be changed to reflect API updates!!

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
        agent.latest_movement, agent.latest_interaction = self._parse_action(action)

        interact = self.interactions[agent.latest_interaction] != "none"
        # Ensure that mitigations are only placed on squares with `UNBURNED` status
        if self._agent_pos_is_unburned(agent) and interact:
            # NOTE: `self.mitigation_placed` is updated in `_update_mitigation()`.
            self._update_mitigation(agent)
        elif (not self._agent_pos_is_unburned()) and interact:
            #set true_mitigation_placed to False if agent has placed mitigation in damaged/mitigated square
            #FIXME: do for each agent
            self.true_mitigation_placed = False
        else:
            # Overwrite value from previous timestep.
            agent.mitigation_placed = False

        # Update agent location on map
        if self.movements[agent.latest_movement] != "none":
            # NOTE: `agent.current_position` is updated in `_update_agent_position()`.
            self._update_agent_position(agent)

    def _parse_action(self, action: np.ndarray) -> Tuple[int, int]:
        """Parse the action into movement and interaction."""
        # NOTE: Assuming that all agents are homogeneous
        if isinstance(self.action_space, spaces.Dict):
            unique_spaces = set([type(v) for v in self.action_space.values()])
            if len(unique_spaces) != 1:
                raise ValueError("Only homogeneous agents are currently supported.")
            act_space = unique_spaces.pop()
            # Handle the MultiDiscrete case
            if issubclass(act_space, spaces.MultiDiscrete):
                return action[0], action[1]
            # Handle the Discrete case
            elif issubclass(act_space, spaces.Discrete):
                return action % len(self.movements), int(action / len(self.movements))
            else:
                raise NotImplementedError(f"{self.action_space} is not supported.")

        # FIXME: Decide what to do with the SARL action parsing; keep for now.
        # Handle the MultiDiscrete case
        elif isinstance(self.action_space, spaces.MultiDiscrete):
            return action[0], action[1]
        # Handle the Discrete case
        elif isinstance(self.action_space, spaces.Discrete):
            return action % len(self.movements), int(action / len(self.movements))
        else:
            raise NotImplementedError(f"{self.action_space} is not supported.")

    def _update_agent_position(self, agent: ReactiveAgent) -> None:
        """Update the agent's position on the map by performing the provided movement."""
        # Store agent's current position in a temporary variable to avoid overwriting it.
        map_boundary = self.sim.fire_map.shape[0] - 1

        # Update the agent's position based on the provided movement.
        movement_str = self.movements[agent.latest_movement]
        # First, check that the movement string is valid.
        if movement_str not in ["up", "down", "left", "right"]:
            raise ValueError(f"Invalid movement string provided: {movement_str}.")
        # Then, ensure that the agent will not move off the map.
        elif movement_str == "up" and not agent.row == 0:
            agent.row -= 1
        elif movement_str == "down" and not agent.row == map_boundary:
            agent.row += 1
        elif movement_str == "left" and not agent.col == 0:
            agent.col -= 1
        elif movement_str == "right" and not agent.col == map_boundary:
            agent.col += 1
        # Movement invalid from current pos, so the agent movement will be ignored.
        # Depending on `self.reward_cls`, the agent may receive a small penalty.
        else:
            # Inform caller that the agent cannot move in the provided direction.
            logger.debug(f"Agent `sim_id`={agent.sim_id}")
            logger.debug(
                f"Agent can't move {movement_str} from row={agent.row}, col={agent.col}."
            )
            logger.debug("Setting `agent.moved_off_map = True` for agent...")
            agent.moved_off_map = True

        # Update the Simulation with new agent position (s).
        point = [agent.col, agent.row, agent.sim_id]
        self.sim.update_agent_positions([point])

    def _agent_pos_is_unburned(self, agent: ReactiveAgent) -> bool:
        """Returns true if the space occupied by the agent has `BurnStatus.UNBURNED`."""
        return self.sim.fire_map[agent.row, agent.col] == BurnStatus.UNBURNED

    def _update_mitigation(self, agent: ReactiveAgent) -> None:
        """Interact with the environment by performing the provided interaction."""
        sim_interaction = self.harness_to_sim[agent.latest_interaction]
        mitigation_update = (agent.col, agent.row, sim_interaction)
        self.sim.update_mitigation([mitigation_update])
        agent.mitigation_placed = True
        # Store indicator that a true mitigation was placed, which will be set back to False in self._do_one_agent_step if agent was in an already damaged/mitigated square
        self.true_mitigation_placed = False

    def _do_one_simulation_step(self) -> bool:
        """Check if the simulation should be run, and then run it if necessary."""
        run_sim = self.timesteps % self.agent_speed == 0
        # The simulation WILL NOT be run every step, unless `self.agent_speed` == 1.
        if run_sim:
            self._run_simulation()
        # Prepare the observation that is returned in the `self.step()` method.
        self._update_state()
        return run_sim

    def _run_simulation(self):
        """Run the simulation (s) for one timestep."""

        self.sim.run(1)

    def _run_benchmark(self):
        """Runs the entire benchmark sim and stores the data needed for the rewards and bench fire maps within each episode"""

        #use timesteps_copy to track the matching timestep that each benchsim fire map will match with the sim fire map
        timesteps_copy = 0

        #if the benchmark simulation has not been updated yet
        if self.benchmark_sim.elapsed_steps == 0:
            
            self.benchmark_sim.run(1)

            #update the benchsim metrics at this timesteps_copy in the harness analytics
            if self.harness_analytics:     
                self.harness_analytics.update_bench_after_one_simulation_step(
            timestep=timesteps_copy
            )

            #update timesteps_copy to next time the simulation with the agent will update
            timesteps_copy = timesteps_copy + self.agent_speed
            
            #store the bench fire map at the sim step
            self.bench_firemaps[(self.harness_analytics.benchmark_sim_analytics.num_sim_steps) - 1] = np.copy(self.benchmark_sim.fire_map)

        #continue to run the benchmark simulation and update the benchsim data/metrics after each sim step    
        while self.benchmark_sim.active == True:

            self.benchmark_sim.run(1)

            #update the benchsim metrics at this timesteps_copy in the harness analytics
            if self.harness_analytics:     
                self.harness_analytics.update_bench_after_one_simulation_step(
            timestep=timesteps_copy
            )

            #update timesteps_copy to next time the simulation with the agent will update
            timesteps_copy = timesteps_copy + self.agent_speed

           #update the size of self.bench_firemaps if this benchmark simulation has lasted longer than any previous benchmark simulations
           if ((self.harness_analytics.benchmark_sim_analytics.num_sim_steps) - 1) > (self.max_bench_length - 1):

                #append the bench fire map to the self.bench_firemaps
                self.bench_firemaps.append(np.copy(self.benchmark_sim.fire_map))

                #update the max length of the benchsim when defining future lists for self.bench_firemaps
                self.max_bench_length = self.max_bench_length + 1

           #else store the bench fire map at the sim step
           else:
                self.bench_firemaps[(self.harness_analytics.benchmark_sim_analytics.num_sim_steps) - 1] = np.copy(self.benchmark_sim.fire_map)

    def _update_state(self):
        """Modify environment's state to contain updates from the current timestep."""
        # Copy the fire map from the simulation so we don't overwrite it.
        fire_map = np.copy(self.sim.fire_map)
        # Update the fire map with the numeric identifier for the agent.
        for agent in self.agents.values():
            fire_map[agent.row, agent.col] = agent.sim_id
        # Modify the state to contain the updated fire map
        self.state[..., self.attributes.index("fire_map")] = fire_map

        #Modify the state to contain the bench fire map at that sim step
        if "bench_fire_map" in self.attributes:

            bench_fire_map_idx = self.attributes.index("bench_fire_map")

            #if the simulation has lasted longer that the benchmark sim, use the final state of the benchsim fire map
            if (self.harness_analytics.benchmark_sim_analytics.num_sim_steps < self.harness_analytics.sim_analytics.num_sim_steps):
                self.state[..., (bench_fire_map_idx)] = self.bench_firemaps[(self.harness_analytics.benchmark_sim_analytics.num_sim_steps) - 1]
            #else get the benchmark sim fire map from the same sim step as the simulation fire map
            else:
                self.state[..., (bench_fire_map_idx)] = self.bench_firemaps[(self.harness_analytics.sim_analytics.num_sim_steps) - 1]

        #Modify the state to contain the final state of bench fire map       
        if "bench_fire_map_final" in self.attributes:

            bench_fire_map_final_idx = self.attributes.index("bench_fire_map_final")
            self.state[..., (bench_fire_map_final_idx)] = self.bench_firemaps[(self.harness_analytics.benchmark_sim_analytics.num_sim_steps) - 1]

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[Any, Any]] = None,
    ) -> Tuple[Dict[Any, np.ndarray], Dict[Any, Dict[Any, Any]]]:
        # log.info("Resetting environment")
        # Use the following line to seed `self.np_random`
        super().reset(seed=seed)
        # Reset the `Simulation` to initial conditions. In particular, this resets the
        # `fire_map`, `terrain`, `fire_manager`, and all mitigations.
        logger.debug("Resetting `self.sim`...")
        self.sim.reset()
        
        bench_exists = False
        if self.benchmark_sim:
            logger.debug("Resetting `self.benchmark_sim`...")
            # set the benchmark seeds to match the sim seeds
            self.benchmark_sim.set_seeds(seed_dict)
            # reset benchmark simulation
            self.benchmark_sim.reset()
            bench_exists = True

        # Reset the agent's contained within the `FireSimulation`.
        logger.debug("Resetting `self.agents`...")
        for agent_id, agent in self.agents.items():
            self.agents[agent_id].reset()

        # Reset `ReactiveHarnessAnalytics` to initial conditions, if it exists.
        if self.harness_analytics:
            logger.debug("Resetting `self.harness_analytics`...")
            self.harness_analytics.reset(benchmark_exists=bench_exists)

        # Get the initial state of the `FireSimulation`, after it has been reset (above).
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

        logger.debug(f"Normalizing obs for attributes: {self.normalized_attributes}")
        observations = super()._normalize_obs({**sim_observations, **nonsim_observations})
        obs = [observations[attribute] for attribute in self.attributes]
        self.state = np.stack(obs, axis=-1).astype(np.float32)

        # Update the `FireSimulation` with the (new) initial agent positions.
        # NOTE: This is slightly redundant, since we can build the list of points within
        # `_create_fire_map()`. For now, it's okay to iterate over `self.agents` twice.
        points = []
        for agent in self.agents.values():
            points.append([agent.col, agent.row, agent.sim_id])

        logger.debug(f"Updating `self.sim` with (new) initial agent positions...")
        self.sim.update_agent_positions(points)

        self.timesteps = 0
        self._log_env_reset()

        # FIXME: Will need to update creation of `marl_obs`` to handle POMDP.
        marl_obs = {ag_id: self.state for ag_id in self._agent_ids}
        infos = {ag_id: {} for ag_id in self._agent_ids}

        # If the agent(s) places an effective mitigation (not placed in already damaged/mitigated square), this is set to True.
        # FIXME Have this tracked across all of the agents
        self.true_mitigation_placed: bool = False

        #Run the new benchsim to obtain the benchsim data used to generate the rewards and policy
        if self.benchmark_sim:
            #run benchmark sim to generate the benchmark sim firemaps and metrics for this episode
            self._run_benchmark()

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
        nonsim_min_maxes["bench_fire_map"] = {
            "min": 0,
            "max": max(self._sim_agent_ids),
        }
        nonsim_min_maxes["bench_fire_map_final"] = {
            "min": 0,
            "max": max(self._sim_agent_ids),
        }
        return nonsim_min_maxes

    def get_nonsim_attribute_data(self) -> OrderedDict[str, np.ndarray]:  # noqa
        nonsim_data = ordered_dict()
        nonsim_data["fire_map"] = self._create_fire_map()
        nonsim_data["bench_fire_map"] = np.zeros(
            (
                self.sim.config.area.screen_size[0],
                self.sim.config.area.screen_size[0],
            )
        )
        nonsim_data["bench_fire_map_final"] = np.zeros(
            (
                self.sim.config.area.screen_size[0],
                self.sim.config.area.screen_size[0],
            )
        )
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

    # def _create_agents(self, method='random', pos_list=None):
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

    def _create_fire_map(self) -> np.ndarray:
        """Prepare the inital copy of `self.sim.fire_map`.

        Creates an ndarray of entirely `BurnStatus.UNBURNED`, except for:
          - The initial fire postion, which is set to `BurnStatus.BURNING`.
          - Each respective agent position is set to the agent's `sim_id`.
        """
        fire_map = np.full(self.sim.fire_map.shape, BurnStatus.UNBURNED)

        # TODO: Potential place to update the initial fire pos to a new value?
        x, y = self.sim.config.fire.fire_initial_position
        logger.debug(f"Placing initial fire position at row={y}, col={x}.")
        fire_map[y, x] = BurnStatus.BURNING  # array should be indexed via (row, col)

        for agent in self.agents.values():
            # Enforce resetting `self.agents` before calling `_create_fire_map()`.
            if agent.initial_position != agent.current_position:
                msg = f"The init and curr pos for agent {agent.agent_id} are different!"
                raise RuntimeError(msg)
            logger.debug(f"Placing {agent.sim_id} at row={agent.row}, col={agent.col}.")
            fire_map[agent.row, agent.col] = agent.sim_id

        return fire_map

    def _create_agents(self, method: str = "random", pos_list: List = None):
        """Create the `ReactiveAgent` objects that will interact with the `FireSimulation`.

        This method will create and populate the `agents` attribute.

        Arguments:
            method: TODO
            pos_list: TODO

        """
        self.agents: Dict[str, ReactiveAgent] = {}
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
            # Create a boolean mask of valid positions (i.e., inside the boundaries).
            mask = np.ones(self.sim.fire_map.shape, dtype=bool)
            # Agent (s) can only be spawned on an unburning square
            # NOTE: Any other "prohibited" agent start locations can be specified here.
            mask[np.where(self.sim.fire_map != BurnStatus.UNBURNED)] = False

            # Randomly select unique positions from the valid ones.
            idx = np.random.choice(range(mask.sum()), size=self.num_agents, replace=False)
            flat_idx = np.argwhere(mask.flatten())[idx].flatten()
            agent_locs = np.vstack(np.unravel_index(flat_idx, mask.shape)).T

            # Populate the `self.agents` dict with `ReactiveAgent` object (s).
            agent_ids = sorted(self._agent_ids, key=lambda x: int(x.split("_")[-1]))
            sim_ids = self._sim_agent_ids
            for agent_str, sim_id, loc in zip(agent_ids, sim_ids, agent_locs):
                agent = ReactiveAgent(agent_str, sim_id, tuple(loc))
                self.agents[agent_str] = agent
        # This should be caught within the init. To be safe, also raise error here.
        else:
            raise NotImplementedError(f"Agent spawn method {method} not implemented.")

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
                    agent_ids=self._agent_ids,
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
