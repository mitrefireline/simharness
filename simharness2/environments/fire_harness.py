import copy
import logging
import os
from abc import abstractmethod
from collections import OrderedDict as ordered_dict
from functools import partial
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    OrderedDict,
    SupportsFloat,
    Tuple,
    TypeVar,
)

import numpy as np
from gymnasium import spaces
from simfire.enums import BurnStatus
from simfire.sim.simulation import FireSimulation
from simfire.utils.config import Config

from simharness2.agents import ReactiveAgent
from simharness2.environments import Harness

logger = logging.getLogger(__name__)

AnyFireSimulation = TypeVar("AnyFireSimulation", bound=FireSimulation)


class FireHarness(Harness[AnyFireSimulation]):
    def __init__(
        self,
        *,
        sim: AnyFireSimulation,
        attributes: List[str],
        normalized_attributes: List[str],
        movements: List[str],
        interactions: List[str],
        action_space_cls: Callable,
        in_evaluation: bool = False,
        benchmark_sim: Optional[AnyFireSimulation] = None,
        harness_analytics_partial: Optional[partial] = None,
        reward_cls_partial: Optional[partial] = None,
        num_agents: int = 1,
        agent_speed: int = 1,
        agent_initialization_method: str = "automatic",
        initial_agent_positions: Optional[List[Tuple[int, int]]] = None,
    ):
        super().__init__(
            sim=sim,
            attributes=attributes,
            normalized_attributes=normalized_attributes,
            in_evaluation=in_evaluation,
        )

        # Define attributes that are specific to the FireHarness.
        self.benchmark_sim = benchmark_sim
        # TODO: use more apt name, ex: `available_movements`, `possible_movements`.
        self.movements = copy.deepcopy(movements)  # FIXME: is deepcopy necessary?
        # TODO: use more apt name, ex: `available_interactions`, `possible_interactions`.
        self.interactions = copy.deepcopy(interactions)  # FIXME: is deepcopy necessary?
        self.harness_to_sim = self.get_harness_to_sim_action_map()

        # Verify that all interactions are supported by the simulator.
        sim_actions = self.sim.get_actions()
        interaction_types = [x for x in self.interactions if x != "none"]
        if not set(interaction_types).issubset(list(sim_actions.keys())):
            raise AssertionError(
                f"All interactions ({str(interaction_types)}) must be "
                f"in the simulator's actions ({str(list(sim_actions.keys()))})!"
            )

        self.agent_speed = agent_speed
        self.num_agents = num_agents
        # Each sim_agent_id is used to "encode" the agent position within the `fire_map`
        # dimension of the returned observation of the environment. The intention is to
        # help the model learn/use the location of the respective agent on the fire_map.
        # NOTE: Assume that every simulator will support 3 base scenarios:
        #  1. Untouched (Ex: simfire.enums.BurnStatus.UNBURNED)
        #  2. Currently Being Affected (Ex: simfire.enums.BurnStatus.BURNING)
        #  3. Affected (Ex: simfire.enums.BurnStatus.BURNED)
        # The max value is +1 of the max mitigation value available (wrt the sim).
        self._agent_id_start = max(self.harness_to_sim.values()) + 1
        self._agent_id_stop = self._agent_id_start + self.num_agents
        self._sim_agent_ids = np.arange(self._agent_id_start, self._agent_id_stop)
        # FIXME: Usage of "agent_{}" doesn't allow us to delineate agents groups.
        self._agent_ids = {f"agent_{i}" for i in self._sim_agent_ids}
        self.default_agent_id = f"agent_{self._agent_id_start}"

        # Spawn the agent (s) that will interact with the simulation
        logger.debug(f"Creating {self.num_agents} agent (s)...")
        input_kwargs = {}
        if agent_initialization_method == "manual":
            if initial_agent_positions is None:
                raise ValueError(
                    "Must provide 'initial_agent_positions' when using 'manual' agent "
                    "initialization method."
                )
            input_kwargs.update({"method": "manual", "pos_list": initial_agent_positions})
        elif agent_initialization_method == "automatic":
            input_kwargs.update({"method": "random"})
        else:
            raise ValueError(
                "Invalid agent initialization method. Must be either 'automatic' or "
                "'manual'."
            )
        self.agents = self.create_agents(**input_kwargs)

        self.min_maxes = self._get_min_maxes()
        self.observation_space = self.get_observation_space()
        self.action_space = self.get_action_space(action_space_cls)

        # FIXME: Update method naming and return value for below methods.
        # TODO: Update type anns on harness_analytics and reward_cls
        # If provided, construct the class used to monitor this `ReactiveHarness` object.
        self._setup_harness_analytics(harness_analytics_partial)
        # If provided, construct the class used to perform reward calculation.
        self._setup_reward_cls(reward_cls_partial)

    def get_observation_space(self) -> spaces.Space:
        """TODO."""
        # NOTE: calling `reshape()` to switch to channel-minor format.
        self._channel_lows = np.array(
            [[[self.min_maxes[channel]["min"]]] for channel in self.attributes]
        ).reshape(1, 1, len(self.attributes))
        self._channel_highs = np.array(
            [[[self.min_maxes[channel]["max"]]] for channel in self.attributes]
        ).reshape(1, 1, len(self.attributes))

        obs_shape = (
            self.sim.fire_map.shape[0],
            self.sim.fire_map.shape[1],
            len(self.attributes),
        )
        low = np.broadcast_to(self._channel_lows, obs_shape)
        high = np.broadcast_to(self._channel_highs, obs_shape)

        return spaces.Box(low=low, high=high, dtype=np.float32)

    @abstractmethod
    def get_action_space(self, action_space_cls: Callable) -> spaces.Space:
        """TODO."""
        raise NotImplementedError

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, SupportsFloat, bool, bool, dict[str, Any]]:
        """Run one timestep of the environment's dynamics."""
        self._do_one_agent_step(action)

        if self.harness_analytics:
            self.harness_analytics.update_after_one_agent_step(
                timestep=self.timesteps, agents=self.agents
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
        # TODO: Ask @mdoyle to fix the below issue.
        # FIXME `fire_status` is set in `FireSimulation.__init__()`, while `active` is
        # set in `FireSimulation.run()`, so attribute DNE prior to first call to `run()`.
        # terminated = self.sim.fire_status == GameStatus.QUIT
        # The simulation has not yet been run via `run()`
        if self.sim.elapsed_steps == 0:
            terminated = False
        else:
            terminated = not self.sim.active

        # Calculate the reward for the current timestep
        # FIXME: pass `terminated` into `get_reward` method
        reward = self.reward_cls.get_reward(self.timesteps, sim_run)

        # FIXME account for below updates in the reward_cls.calculate_reward() method
        # "End of episode" reward
        if terminated:
            reward += 10

        if self.harness_analytics:
            self.harness_analytics.update_after_one_harness_step(
                sim_run, terminated, reward, timestep=self.timesteps
            )

        self.timesteps += 1  # increment AFTER method logic is performed (convention).

        # FIXME: Add configurable option to save sim.fire_map to file.
        # if terminated and self._debug_mode:
        #     outdir = self._trial_results_path
        #     subdir = "eval" if self._is_eval_env else "train"
        #     savedir = os.path.join(outdir, "fire_map", subdir)
        #     os.makedirs(savedir, exist_ok=True)
        #     # Make file name used for saving the fire map
        #     episodes_total = self.harness_analytics.episodes_total
        #     fname = f"{os.getpid()}-episode-{episodes_total}-fire_map"
        #     save_path = os.path.join(savedir, fname)
        #     logger.info(f"Saving fire map to {save_path}...")
        #     np.save(save_path, self.sim.fire_map)

        return self.state, reward, terminated, truncated, {}

    def _do_one_agent_step(
        self,
        *,
        action: np.ndarray,
        agent: Optional[ReactiveAgent] = None,
    ) -> None:
        """Move the agent and interact with the environment."""
        if agent is None:
            # TODO: Handle key error in event of bad usage.
            agent = self.agents[self.default_agent_id]

        # Parse the movement and interaction from the action, and store them.
        agent.latest_movement, agent.latest_interaction = self._parse_action(action)

        interact = self.interactions[agent.latest_interaction] != "none"
        # Ensure that mitigations are only placed on squares with `UNBURNED` status
        if self._agent_pos_is_unburned(agent) and interact:
            # NOTE: `self.mitigation_placed` is updated in `_update_mitigation()`.
            self._update_mitigation(agent)
        else:
            # Overwrite value from previous timestep.
            agent.mitigation_placed = False

        # Update agent location on map
        if self.movements[agent.latest_movement] != "none":
            # NOTE: `agent.current_position` is updated in `_update_agent_position()`.
            self._update_agent_position(agent)

    def _parse_action(self, action: np.ndarray) -> Tuple[int, int]:
        """Parse the action into movement and interaction."""
        # Handle the MultiDiscrete case
        if isinstance(self.action_space, spaces.MultiDiscrete):
            # FIXME: Indexing assumes action only has 2 elements.
            return action[0], action[1]
        # Handle the Discrete case
        elif isinstance(self.action_space, spaces.Discrete):
            return action % len(self.movements), int(action / len(self.movements))
        else:
            raise NotImplementedError(f"{self.action_space} is not supported.")

    def _agent_pos_is_unburned(self, agent: ReactiveAgent) -> bool:
        """Returns true if the space occupied by the agent has `BurnStatus.UNBURNED`."""
        return self.sim.fire_map[agent.row, agent.col] == BurnStatus.UNBURNED

    def _update_mitigation(self, agent: ReactiveAgent) -> None:
        """Interact with the environment by performing the provided interaction."""
        sim_interaction = self.harness_to_sim[agent.latest_interaction]
        mitigation_update = (agent.col, agent.row, sim_interaction)
        self.sim.update_mitigation([mitigation_update])
        agent.mitigation_placed = True

    def _update_agent_position(self, agent: ReactiveAgent) -> None:
        """Update the agent's position on the map by performing the provided movement."""
        # Store agent's current position in a temporary variable to avoid overwriting it.
        row_boundary, col_boundary = [x - 1 for x in self.sim.fire_map.shape]

        # Update the agent's position based on the provided movement.
        movement_str = self.movements[agent.latest_movement]
        # First, check that the movement string is valid.
        if movement_str not in ["up", "down", "left", "right"]:
            raise ValueError(f"Invalid movement string provided: {movement_str}.")
        # Then, ensure that the agent will not move off the map.
        elif movement_str == "up" and not agent.row == 0:
            agent.row -= 1
        elif movement_str == "down" and not agent.row == row_boundary:
            agent.row += 1
        elif movement_str == "left" and not agent.col == 0:
            agent.col -= 1
        elif movement_str == "right" and not agent.col == col_boundary:
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
        if self.benchmark_sim:
            self.benchmark_sim.run(1)

        self.sim.run(1)

    def _update_state(self):
        """Modify environment's state to contain updates from the current timestep."""
        # Copy the fire map from the simulation so we don't overwrite it.
        fire_map = np.copy(self.sim.fire_map)
        # Update the fire map with the numeric identifier for the agent.
        for agent in self.agents.values():
            fire_map[agent.row, agent.col] = agent.sim_id
        # Modify the state to contain the updated fire map
        self.state[..., self.attributes.index("fire_map")] = fire_map

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, dict]:
        """Reset environment to an initial state, returning an initial obs and info."""
        # Use the following line to seed `self.np_random`
        super().reset(seed=seed, options=options)
        # Reset the `Simulation` to initial conditions. In particular, this resets the
        # `fire_map`, `terrain`, `fire_manager`, and all mitigations.
        logger.debug("Resetting `self.sim`...")
        self.sim.reset()
        if self.benchmark_sim:
            logger.debug("Resetting `self.benchmark_sim`...")
            self.benchmark_sim.reset()

        # Reset the agent's contained within the `FireSimulation`.
        logger.debug("Resetting `self.agents`...")
        for agent_id in self.agents.keys():
            self.agents[agent_id].reset()

        # Reset `ReactiveHarnessAnalytics` to initial conditions, if it exists.
        if self.harness_analytics:
            logger.debug("Resetting `self.harness_analytics`...")
            self.harness_analytics.reset()

        # Get the initial state of the `FireSimulation`, after it has been reset (above).
        self.state = self.get_initial_state()

        self.timesteps = 0
        self._num_eval_iters = 0

        self._log_env_reset()

        return self.state, {}

    def get_initial_state(self) -> np.ndarray:
        """TODO."""
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

        return np.stack(obs, axis=-1).astype(np.float32)

    def get_harness_to_sim_action_map(self) -> Dict[int, int]:
        """Create conversion dictionaries for action (Sim) <-> interaction (Harness)."""
        # NOTE: hts == "harness_to_sim" and sth == "sim_to_harness"
        action_map = {}

        sim_actions = self.sim.get_actions()
        if len(self.interactions) > 0:
            # Using the "valid" interaction_types, populate the conversion dicts.
            valid_idxs = [
                self.interactions.index(act) for act in self.interactions if act != "none"
            ]

            for idx in valid_idxs:
                interaction = self.interactions[idx]
                action_map[idx] = int(sim_actions[interaction])

        return action_map

    def create_agents(
        self, method: str = "random", pos_list: List = None
    ) -> Dict[str, ReactiveAgent]:
        """Create ReactiveAgent object (s) that will interact w/ the FireSimulation."""
        agents_dict = {}
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
            # raise NotImplementedError  # adding so I don't forget!
            agent_ids = sorted(self._agent_ids, key=lambda x: int(x.split("_")[-1]))
            for agent_str, agent_info, sim_id in zip(
                agent_ids, pos_list, self._sim_agent_ids
            ):
                x, y = agent_info
                agent = ReactiveAgent(agent_str, sim_id, (x, y))
                agents_dict[agent_str] = agent
            return agents_dict

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
            for agent_str, sim_id, loc in zip(agent_ids, self._sim_agent_ids, agent_locs):
                agent = ReactiveAgent(agent_str, sim_id, tuple(loc))
                agents_dict[agent_str] = agent
            return agents_dict

        # This should be caught within the init. To be safe, also raise error here.
        else:
            raise NotImplementedError(f"Agent spawn method {method} not implemented.")

    @property
    def default_agent_id(self) -> str:
        """Return the default agent id."""
        return self._default_agent_id

    @default_agent_id.setter
    def default_agent_id(self, agent_id: str) -> None:
        """Set default agent id. When num_agents > 1, the value is an empty string."""
        if agent_id not in self._agent_ids:
            raise ValueError(f"Invalid agent id provided: {agent_id}.")
        self._default_agent_id = agent_id if self.num_agents == 1 else ""

    def render(self):
        """Render a visualization of the environment."""
        self._configure_env_rendering(should_render=True)

    def get_nonsim_attribute_data(self) -> OrderedDict[str, np.ndarray]:  # noqa
        nonsim_data = ordered_dict()
        nonsim_data["fire_map"] = self.prepare_fire_map()
        return nonsim_data

    def prepare_fire_map(self) -> np.ndarray:
        """Prepare initial state of the `fire_map` attribute.

        Creates an ndarray of entirely `BurnStatus.UNBURNED`, except for:
          - The initial fire postion, which is set to `BurnStatus.BURNING`.
          - Each respective agent position is set to the agent's `sim_id`.
        """
        fire_map = np.full(self.sim.fire_map.shape, BurnStatus.UNBURNED)

        # TODO: Verify that the initial fire position is as expected.
        col, row = self.sim.config.fire.fire_initial_position
        logger.debug(f"Placing initial fire position at row={row}, col={col}.")
        fire_map[row, col] = BurnStatus.BURNING

        agent_points = []
        for agent in self.agents.values():
            # Enforce resetting `self.agents` before calling `_create_fire_map()`.
            if agent.initial_position != agent.current_position:
                msg = f"The init and curr pos for agent {agent.agent_id} are different!"
                raise RuntimeError(msg)
            logger.debug(f"Placing {agent.sim_id} at row={agent.row}, col={agent.col}.")
            fire_map[agent.row, agent.col] = agent.sim_id

            agent_points.append([agent.col, agent.row, agent.sim_id])

        # Update the `FireSimulation` with the (new) initial agent positions.
        logger.debug("Updating `self.sim` with (new) initial agent positions...")
        self.sim.update_agent_positions(agent_points)

        return fire_map

    def get_nonsim_attribute_bounds(self) -> OrderedDict[str, Dict[str, int]]:  # noqa
        nonsim_min_maxes = ordered_dict()
        # The values in "fire_map" are:
        #   - 0: BurnStatus.UNBURNED
        #   - 1: BurnStatus.BURNING
        #   - 2: BurnStatus.BURNED
        #   - 3: BurnStatus.FIRELINE (if "fireline" in self.interactions)
        #   - 4: BurnStatus.SCRATCHLINE (if "scratchline" in self.interactions)
        #   - 5: BurnStatus.WETLINE (if "wetline" in self.interactions)
        #   - X: self._agent_id_start <= X < self._agent_id_stop
        nonsim_min_maxes["fire_map"] = {
            "min": 0,
            "max": max(self._sim_agent_ids),
        }
        return nonsim_min_maxes

    def _get_min_maxes(self) -> OrderedDict[str, Dict[str, Tuple[int, int]]]:
        """Retrieves the minimum and maximum for all relevant attributes."""
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

        return ordered_dict({**sim_min_maxes, **nonsim_min_maxes})

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

    def _setup_harness_analytics(self, analytics_partial: partial) -> None:
        """Instantiates `harness_analytics` used to monitor this `ReactiveHarness` obj.

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
        # self.harness_analytics: ReactiveHarnessAnalytics
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
        # self.reward_cls: BaseReward
        if reward_cls_partial:
            try:
                self.reward_cls = reward_cls_partial(
                    harness_analytics=self.harness_analytics
                )
            except Exception as e:
                raise e
        else:
            self.reward_cls = None

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


class ReactiveHarness(FireHarness[AnyFireSimulation]):
    def get_action_space(self, action_space_cls: Callable) -> spaces.Space:
        """TODO."""
        if action_space_cls is spaces.Discrete:
            input_arg = len(self.movements) * len(self.interactions)
        elif action_space_cls is spaces.MultiDiscrete:
            input_arg = [len(self.movements), len(self.interactions)]
        else:
            raise NotImplementedError

        return action_space_cls(input_arg)


class MultiAgentAsTupleActionReactiveHarness(ReactiveHarness[AnyFireSimulation]):
    def get_action_space(self, action_space_cls: Callable) -> spaces.Space:
        """Get Tuple action space, where indices contain each agent's action space."""
        if action_space_cls is spaces.Discrete:
            act_spaces = [
                spaces.Discrete(len(self.movements) * len(self.interactions))
                for _ in range(self.num_agents)
            ]
        elif action_space_cls is spaces.MultiDiscrete:
            act_spaces = [
                spaces.MultiDiscrete([len(self.movements), len(self.interactions)])
                for _ in range(self.num_agents)
            ]
        else:
            # TODO provide a descriptive error message.
            raise NotImplementedError

        return spaces.Tuple(act_spaces)

    def _do_one_agent_step(
        self,
        *,
        action: np.ndarray,
        agent: Optional[ReactiveAgent] = None,
    ) -> None:
        """Move each agent and interact with the environment (when num_agents > 1)."""
        # This harness assumes MARL, so we need to iterate over each agent.
        for agent_id, agent in self.agents.items():
            agent_idx = np.where(self._sim_agent_ids == int(agent_id.split("_")[1]))[
                0
            ].item()
            agent_action = action[agent_idx]
            super()._do_one_agent_step(agent=agent, action=agent_action)
