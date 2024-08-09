import copy
import logging
import os
from abc import abstractmethod
from collections import OrderedDict as ordered_dict
from collections import namedtuple
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
    Type,
    TypeVar,
    TYPE_CHECKING,
)

import numpy as np
from gymnasium import spaces
from simfire.enums import BurnStatus
from simfire.sim.simulation import FireSimulation
from simfire.utils.config import Config

from simharness2.agents.agent import ReactiveAgent
from simharness2.agents.initialization import AgentInitializer
from simharness2.environments.harness import Harness, get_unsupported_attributes
from simharness2.environments import utils as env_utils

if TYPE_CHECKING:
    from simharness2.environments.utils import BurnMDOperationalLocation

logger = logging.getLogger(__name__)

AnyFireSimulation = TypeVar("AnyFireSimulation", bound=FireSimulation)
AnyAgentInitializer = TypeVar("AnyAgentInitializer", bound=AgentInitializer)

FIRE_MAP_ATTRIBUTES = ["fire_map", "fire_map_with_agents"]
BENCHMARK_ATTRIBUTES = ["bench_fire_map", "bench_fire_map_final"]
SIMFIRE_ATTRIBUTES = FireSimulation.supported_attributes()


class FireHarness(Harness[AnyFireSimulation]):
    def __init__(
        self,
        *,
        sim_init_cfg: Dict[str, Any],
        attributes: List[str],
        normalized_attributes: List[str],
        movements: List[str],
        interactions: List[str],
        action_space_cls: Callable,
        in_evaluation: bool = False,
        reward_init_cfg: Dict[str, Any],
        benchmark_sim_init_cfg: Dict[str, Any] = None,
        harness_analytics_partial: Optional[partial] = None,
        num_agents: int = 1,
        agent_speed: int = 1,
        agent_initialization_cls: Callable = None,
        agent_initialization_kwargs: Dict[str, Any] = {},
        **kwargs,
    ):
        super().__init__(
            sim_init_cfg=sim_init_cfg,
            attributes=attributes,
            normalized_attributes=normalized_attributes,
            in_evaluation=in_evaluation,
        )
        # TODO: Define `benchmark_sim` in `DamageAwareReactiveHarness`.
        # Define attributes that are specific to the FireHarness.
        # Use provided benchmark simulation info to create a simulation object.
        self.benchmark_sim = env_utils.create_fire_simulation_from_config(
            benchmark_sim_init_cfg
        )
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
        self.agents = self.create_agents(
            agent_initialization_cls, agent_initialization_kwargs
        )

        self.min_maxes = self._get_min_maxes()
        self.observation_space = self.get_observation_space()
        self.action_space = self.get_action_space(action_space_cls)

        # FIXME: Update method naming and return value for below methods.
        # TODO: Update type anns on harness_analytics and reward_cls
        # If provided, construct the class used to monitor this `ReactiveHarness` object.
        self._setup_harness_analytics(harness_analytics_partial)
        # If provided, construct the class used to perform reward calculation.
        self._setup_reward_cls(reward_init_cfg)

        # Indicator flag to determine if fire in the sim can spread diagonally.
        self._fire_diagonal_spread = self.sim.config.fire.diagonal_spread
        # TODO: Decide default val. Setting to False allows _set_fire_initial_position()
        # to be the only method that should change this to True?
        # NOTE: If default is False, then ray/rllib/utils/pre_checks/env.py:check_env()
        # method will fail, because we DO NOT run the benchmark and therefore do not have
        # the data needed! So, set to True for now, and we can figure out how to be less
        # redundant later on.
        self._new_fire_scenario = True
        # Fire context will be initialized from external callback, InitializeSimfire.
        self.fire_context: env_utils.EnvFireContext

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
        self._do_one_agent_step(action=action)

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

        truncated = self._should_truncate()
        terminated = self._should_terminate()

        # Calculate the timestep reward for each agent.
        reward = self.reward_cls.get_reward(
            timestep=self.timesteps,
            sim_run=sim_run,
            done_episode=terminated or truncated,
            agents=self.agents,
            agent_speed=self.agent_speed,
        )

        # FIXME account for below updates in the reward_cls.calculate_reward() method
        # "End of episode" reward
        # if terminated:
        # reward += 10

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

        row, col, shape = agent.row, agent.col, self.sim.fire_map.shape
        diag_spread = self._fire_diagonal_spread
        adj_rows, adj_cols = env_utils.get_adjacent_points(row, col, shape, diag_spread)
        agent.adj_to_mitigation[adj_rows, adj_cols] = 1

    def _update_agent_position(self, agent: ReactiveAgent) -> None:
        """Update the agent's position on the map by performing the provided movement."""
        # Store agent's current position in a temporary variable to avoid overwriting it.
        row_boundary, col_boundary = [x - 1 for x in self.sim.fire_map.shape]

        # Reset the `moved_off_map` attribute for the current timestep.
        moved_off_map = False
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
            moved_off_map = True

        # Update the `moved_off_map` attribute for the current timestep.
        logger.debug(f"Setting `agent.moved_off_map = {moved_off_map}` for agent...")
        agent.moved_off_map = moved_off_map

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

    def _should_truncate(self) -> bool:
        # TODO(afennelly): Need to handle truncation properly. For now, we assume that
        # the episode will never be truncated, but this isn't necessarily true.
        return False

    def _should_terminate(self) -> bool:
        # TODO: Ask @mdoyle to fix the below issue.
        # FIXME `fire_status` is set in `FireSimulation.__init__()`, while `active` is
        # set in `FireSimulation.run()`, so attribute DNE prior to first call to `run()`.
        # terminated = self.sim.fire_status == GameStatus.QUIT
        # The simulation has not yet been run via `run()`
        if self.sim.elapsed_steps == 0:
            terminated = False
        else:
            terminated = not self.sim.active

        return terminated

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

        # Reset the agent's contained within the `FireSimulation`.
        logger.debug("Resetting `self.agents`...")
        for agent_id in self.agents.keys():
            self.agents[agent_id].reset()

        # Reset `ReactiveHarnessAnalytics` to initial conditions, if it exists.
        if self.harness_analytics:
            logger.debug("Resetting `self.harness_analytics`...")
            render = self._should_render if hasattr(self, "_should_render") else False
            # Don't reset benchmark analytics if the data is still being used!
            self.harness_analytics.reset(
                env_is_rendering=render,
                reset_benchmark=self._new_fire_scenario,
            )

        # Get the initial state of the `FireSimulation`, after it has been reset (above).
        self.state = self.get_initial_state()
        self.timesteps = 0

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
        self,
        initializer_cls: Type[AnyAgentInitializer],
        initializer_kwargs: Dict[str, Any],
    ) -> Dict[str, ReactiveAgent]:
        """Create ReactiveAgent object (s) that will interact w/ the FireSimulation."""
        agent_ids = sorted(self._agent_ids, key=lambda x: int(x.split("_")[-1]))
        agent_initializer = initializer_cls(**initializer_kwargs)
        return agent_initializer.initialize_agents(
            agent_ids=agent_ids,
            sim_ids=self._sim_agent_ids,
            fire_map_shape=self.sim.fire_map.shape,
        )

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

    @staticmethod
    def supported_attributes() -> List[str]:
        """Return full list of attributes supported by the harness."""
        return FIRE_MAP_ATTRIBUTES + SIMFIRE_ATTRIBUTES

    def render(self):
        """Render a visualization of the environment."""
        self._configure_env_rendering(should_render=True)

    def get_nonsim_attribute_data(self) -> OrderedDict[str, np.ndarray]:  # noqa
        nonsim_data = ordered_dict()
        fire_map_attr = set(FIRE_MAP_ATTRIBUTES).intersection(set(self.attributes))
        # Ensure that only 1 attribute from FIRE_MAP_ATTRIBUTES is provided.
        if len(fire_map_attr) == 1:
            fire_map_attr = fire_map_attr.pop()
            place_agents = True if "agents" in fire_map_attr else False
            nonsim_data[fire_map_attr] = self.prepare_fire_map(place_agents=place_agents)
        else:
            raise AssertionError(
                f"Expected 1 attribute from {FIRE_MAP_ATTRIBUTES}; got {len(fire_map_attr)}."
            )
        return nonsim_data

    # FIXME: `prepare_initial_fire_map` is a better name for this method?
    def prepare_fire_map(
        self, place_agents: bool = True, set_agent_positions: bool = True
    ) -> np.ndarray:
        """Prepare initial state of the `fire_map` attribute.

        Creates an ndarray of entirely `BurnStatus.UNBURNED`, except for:
          - The initial fire postion, which is set to `BurnStatus.BURNING`.
          - Each respective agent position is set to the agent's `sim_id`.
        """
        # Prepare inital state of fire map using known starting conditions.
        fire_map = np.full(self.sim.fire_map.shape, BurnStatus.UNBURNED)
        col, row = self.sim.config.fire.fire_initial_position
        logger.debug(f"Placing initial fire position at row={row}, col={col}.")
        fire_map[row, col] = BurnStatus.BURNING

        agent_points = []
        for agent in self.agents.values():
            # Enforce resetting `self.agents` before calling `_create_fire_map()`.
            if agent.initial_position != agent.current_position:
                msg = f"The init and curr pos for agent {agent.agent_id} are different!"
                raise RuntimeError(msg)
            # Enable the user to choose whether agent ids should be placed on fire_map.
            if place_agents:
                logger.debug(
                    f"Placing {agent.sim_id} at row={agent.row}, col={agent.col}."
                )
                fire_map[agent.row, agent.col] = agent.sim_id

            agent_points.append([agent.col, agent.row, agent.sim_id])

        # Update the `FireSimulation` with the (new) initial agent positions.
        if set_agent_positions:
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

    def _set_operational_location(
        self,
        *,
        locations: Optional[List["BurnMDOperationalLocation"]] = None,
        num_envs_per_worker: int = None,
        location_to_use: Optional["BurnMDOperationalLocation"] = None,
    ) -> str:
        """Set the operational location for the current environment.

        This method sets the operational location for the current environment by either
        selecting from a provided list of operational locations based on an even
        distribution algorithm or using a directly provided operational location. The
        selected operational location is then used to update the simulation and, if
        present, the benchmark simulation.
    
        Note:
        - This method updates the operational location for both the simulation instance (`self.sim`)
            and, if available, the benchmark simulation instance (`self.benchmark_sim`).
        - The operational location is essential for configuring the simulation environment
            and influences various simulation parameters and behaviors.

        Arguments:
            locations: An optional list of `BurnMDOperationalLocation` objects. If 
                provided, one of these locations will be selected based on an even
                distribution algorithm that considers the number of environments per
                worker. This argument is mutually exclusive with `location_to_use`.
            num_envs_per_worker: The number of environments contained within each worker.
                This is used to calculate the index of the operational location to use
                from the `locations` list, ensuring an even distribution of locations
                across environments. This arg is required if `locations` is provided.
            location_to_use: An optional `BurnMDOperationalLocation` object. If provided,
                this location is directly used as the operational location for the
                current environment. This arg is mutually exclusive with `locations`.

        Returns:
            The `uid` of the operational location that was selected or provided for the
            current environment. The `uid` is structured as "state_year_fireName", 
            e.g., "Oregon_2020_White_River".

        Raises:
            ValueError: If neither `locations` nor `location_to_use` is provided.
        """
        if locations is None and location_to_use is None:
            raise ValueError("Either `locations` or `location_to_use` must be provided.")
        elif locations is not None:
            logger.info("Selecting operational location from provided locations...")
            # Get the index of the operational location to use for the current environment.
            logger.debug(f"There are {len(locations)} operational locations provided.")
            loc_idx = self._get_even_distribution_index(
                data_length=len(locations), num_envs_per_worker=num_envs_per_worker
            )
            logger.debug(f"Operational location at index {loc_idx} will be used.")
            location_to_use = locations[loc_idx]
        else:
            logger.info("Using provided operational location...")

        # Prepare the environment and simulation for the selected operational location.
        w_idx, v_idx = (
            self.rllib_env_context.worker_index,
            self.rllib_env_context.vector_index,
        )
        logger.info(
            f"({w_idx}, {v_idx}) Setting self._burnmd_op_loc to {location_to_use}..."
        )
        self._burnmd_op_loc = location_to_use

        # TODO: Create MR for simfire to add `set_operational_location` method and
        # optimize/update the logic of `reset_terrain()`.
        logger.info("Updating self.sim with selected operational location...")
        self.sim = env_utils.set_operational_location(self.sim, self._burnmd_op_loc)
        if self.benchmark_sim:
            logger.info(
                "Updating self.benchmark_sim with selected operational location..."
            )
            self.benchmark_sim = env_utils.set_operational_location(
                self.benchmark_sim, self._burnmd_op_loc
            )

        return self._burnmd_op_loc.uid

    def _set_fire_initial_position(
        self,
        *,
        data: Optional[np.recarray] = None,
        num_envs_per_worker: int = None,
        loc_to_idx: Dict[str, int] = None,
        position: Optional[Tuple[int, int]] = None,
    ) -> Tuple[str, Tuple[int, int]]:
        """Update the `fire_initial_position` for the `FireSimulation` instance.

        This method selects the initial position of the fire based on either a provided
        position or by selecting from a dataset of fire scenarios. If a dataset is
        provided, it selects a scenario based on the environment index to ensure even
        distribution across different environments. The selected fire position is then
        set for both the simulation and, if present, the benchmark simulation.

        Note:
            - If `data` is a 2D array, it is assumed that the first axis is used to
                select an array for the respective operation location.
            - This method updates the `fire_initial_position` for both the simulation
                instance and, if available, the benchmark simulation instance.

        Arguments:
            data: An optional numpy recarray containing the sample of fire scenarios to
                choose from. If provided, the method selects the fire initial position
                based on the scenarios in this dataset. If not provided, the `position`
                argument must be specified.
            num_envs_per_worker: The number of environments that are contained within
                each worker. This is used to determine the index of the fire scenario
                that should be used for the current environment, ensuring an even
                distribution of scenarios across environments.
            loc_to_idx: An optional dictionary mapping operation location identifiers to
                indices within the `data` array. This is required if `data` is a 2D
                array, to select the appropriate row for the operation location.
            position: An optional tuple specifying the (x, y) coordinates of the fire's
                initial position. If provided, this position is used directly to set the
                fire's initial position. If not provided, the `data` argument must be
                specified.

        Returns:
            A tuple containing the operation location UID and the selected initial
            position of the fire as a tuple of (x, y) coordinates.

        Raises:
            ValueError: If neither `data` nor `position` is provided.
            NotImplementedError: If the `data` shape is not 2D, and the method is
                expected to handle such cases in future implementations.
        """
        if data is None and position is None:
            raise ValueError("Either `data` or `position` must be provided.")
        elif data is not None:
            logger.info("Selecting fire initial position from provided data...")
            # Get the index of the fire scenario to use for the current environment.
            fire_idx = self._get_even_distribution_index(
                data_length=data.shape[-1], num_envs_per_worker=num_envs_per_worker
            )

            # TODO: Maybe create custom recarray to use for type hints on attributes?
            # If 2D array, assume first axis is used to get arr for respective op location
            if len(data.shape) == 2:
                op_loc_idx = loc_to_idx[self._burnmd_op_loc.uid]
                fire_pos_arr = data[op_loc_idx, fire_idx]
            else:
                # FIXME: We should probably raise an error if the shape is not 2D?
                # But, we also might want to allow usage of this method from elsewhere,
                # ie. not solely from the `InitializeSimfire` callback.
                fire_pos_arr = data[fire_idx]

            # Use the fire scenario to initialize the `FireSimulation`.
            fire_pos_arr = fire_pos_arr.view(type=np.recarray)
            position = (fire_pos_arr.x, fire_pos_arr.y)
        else:
            logger.info(f"Using provided fire initial position: {position}...")

        logger.debug(f"Setting simulation fire initial position to {position}...")
        self.sim.set_fire_initial_position(position)
        if self.benchmark_sim:
            logger.debug(f"Setting benchmark_sim fire initial position to {position}...")
            self.benchmark_sim.set_fire_initial_position(position)

        self._new_fire_scenario = True

        return self._burnmd_op_loc.uid, position

    def _get_even_distribution_index(
        self, *, data_length: int, num_envs_per_worker: int
    ) -> int:
        """Calculates an index for even distribution of data across all environments.

        The index is used to sample the data in a way that spreads it across all
        environments contained within the respective `WorkerSet` as evenly as possible.

        TODO: Decide if this method should be made static, and passed the env context
        data. This may make it easier to test and validate the method logic.

        Arguments:
            data_length: The length of the data being distributed.
            num_envs_per_worker: The number of sub-environments within each worker.

        Returns:
            The index to use for the current environment when sampling the data.
        """
        env_context = self.rllib_env_context
        w_i, v_i = env_context.worker_index, env_context.vector_index
        # NOTE: We use the modulo operator to ensure that the `fire_idx` is within the
        # available indices of the provided data.
        if env_context.num_workers == 0:
            # Sub-environment(s) contained within only the `local_worker`.
            idx = ((w_i + 1) * v_i) % data_length
        else:
            # Sub-environment(s) contained within only the `remote_worker`(s).
            idx = ((num_envs_per_worker * w_i) + v_i) % data_length

        return idx

    def _build_fire_context(self) -> env_utils.EnvFireContext:
        """Build fire-specific environment context. See EnvFireContext for details."""
        # If context has not been set, return values of (-1, -1) and log error.
        try:
            env_context = self.rllib_env_context
            w_i, v_i = env_context.worker_index, env_context.vector_index
        except AttributeError as err:
            logger.error(f"Error: {err}. Setting `w_i` and `v_i` to -1.")
            w_i, v_i = -1, -1

        # Ensure operational location data is present if data layers are used.
        fuel_type = self.sim.config.terrain.fuel_type
        topo_type = self.sim.config.terrain.topography_type
        if fuel_type == "operational" or topo_type == "operational":
            if not hasattr(self, "_burnmd_op_loc") or self._burnmd_op_loc is None:
                raise AttributeError(
                    f"The fuel type ({fuel_type}) or topography type ({topo_type}) is "
                    "set to 'operational', but the SimHarness environment does not have "
                    f"self._burnmd_op_loc set. The FireHarness._set_operational_location() "
                    "method can be used to set this attribute."
                )
            burnmd_op_loc = self._burnmd_op_loc
            op_config = self.sim.config.operational
        # TODO: "Fallback" behavior in case functional is used - decide default values.
        else:
            burnmd_op_loc = None
            op_config = None

        # Get the initial position of the fire for the current environment.
        fire_init_pos = self.sim.config.fire.fire_initial_position
        self.fire_context = env_utils.EnvFireContext(
            worker_index=w_i,
            vector_index=v_i,
            fire_initial_position=fire_init_pos,
            burnmd_operational_location=burnmd_op_loc,
            operational_config=op_config,
        )
        return self.fire_context

    def _configure_fire_for_recreated_worker(
        self,
        env_to_fire_context: Dict[Tuple[int, int], env_utils.EnvFireContext],
    ) -> None:
        """Update the `FireSimulation` with the configuration data for the current env.

        Arguments:
            env_to_simfire_config: A dictionary mapping the environment uid (namely, the
                pair of worker and vector indices) to a snapshot of the fire context that
                was present prior to worker failure.
        """
        env_context = self.rllib_env_context
        env_id = env_context.worker_index, env_context.vector_index
        fire_context = env_to_fire_context[env_id]

        logger.info(f"Updating `self.sim` with fire context for env_id={env_id}...")
        # Handle op location and fire initial position for the current environment.
        self._set_operational_location(
            location_to_use=fire_context.burnmd_operational_location
        )
        self._set_fire_initial_position(position=fire_context.fire_initial_position)
        # FIXME: Below would be doing the above, but manually instead of calling method.
        # position = fire_context.fire_initial_position
        # logger.info(f"Setting simulation fire initial position to {position}...")
        # self.sim.set_fire_initial_position(position)
        # if self.benchmark_sim:
        #     logger.info(f"Setting benchmark_sim fire initial position to {position}...")
        #     self.benchmark_sim.set_fire_initial_position(position)

        # Ensure that harness attributes are correctly updated.
        self._new_fire_scenario = True

        # Verify that the operational configs match.
        if self.sim.config.operational != fire_context.operational_config:
            raise AssertionError("Operational configs do not match!")

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
                    # TODO: Define `benchmark_sim` in `DamageAwareReactiveHarness`.
                    benchmark_sim=self.benchmark_sim,
                    agent_ids=self._agent_ids,
                )
            except Exception as e:
                raise e
        else:
            self.harness_analytics = None

    def _setup_reward_cls(self, reward_init_cfg: Dict[str, Any]) -> None:
        """Instantiates the reward class used to perform reward calculation each episode.

        This method must be called AFTER `self._setup_harness_analytics()`, as the reward
        class requires `self.harness_analytics` to be passed as an argument to its
        constructor.

        Arguments:
            reward_init_cfg: A dictionary that contains the reward class to use for
                reward calculation after each timestep, and it's respective input
                arguments are expected under the 'kwargs' key.
        Raises:
            TypeError: If `harness_analytics_partial.keywords` does not contain a
                `sim_data_partial` key with value of type `functools.partial`.
            AttributeError: If `self` does not have a `harness_analytics` attribute.
                See the above message for more details.

        """
        reward_cls = reward_init_cfg["reward_cls"]
        reward_kwargs = reward_init_cfg["kwargs"]
        if reward_kwargs is None:
            reward_kwargs = {}

        reward_kwargs.update({"harness_analytics": self.harness_analytics})
        self.reward_cls = reward_cls(**reward_kwargs)

    def _get_non_interaction_disaster_categories(self) -> Dict[str, int]:
        """Get disaster categories that aren't interactions.

        Arguments:
            disaster_categories (List[str]): List of potential Simulation space categories

        Returns:
            A list containing disaster categories (str), with interactions removed.
        """
        categories = {}
        enum_names_for_sim_actions = [v.name for v in self.sim.get_actions().values()]
        for enum_name, enum_val in self.sim.get_disaster_categories().items():
            if enum_name not in enum_names_for_sim_actions:
                categories[enum_name] = enum_val
        return categories


class ReactiveHarness(FireHarness[AnyFireSimulation]):
    def __init__(self, **kwargs):
        # TODO: Verify this call to super init when using **kwargs
        super().__init__(**kwargs)

        # TODO: Make this check more general, ie. not included in every subclass?
        # Validate that the attributes provided are supported by this harness.
        curr_cls = self.__class__
        bad_attributes = get_unsupported_attributes(self.attributes, curr_cls)
        if bad_attributes:
            msg = (
                f"The {curr_cls.__name__} class does not support the "
                f"following attributes: {bad_attributes}."
            )
            raise AssertionError(msg)

        if self.num_agents > 1:
            msg = f"{self.__class__.__name__} only supports a single agent."
            raise NotImplementedError(msg)

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


class DamageAwareReactiveHarness(ReactiveHarness[AnyFireSimulation]):
    def __init__(
        self,
        *,
        terminate_if_greater_damage: bool = True,
        max_bench_length: int = 600,
        **kwargs,
    ):
        # TODO: Verify this call to super init when using **kwargs
        super().__init__(**kwargs)
        # Bool to toggle the ability to terminate the agent simulation early if at the
        # current timestep of the agent simulation, the agents have caused more burn
        # damage (burned + burning) than the final state of the benchmark fire map.
        # TODO Have this value set in the configs
        self._terminate_if_greater_damage = terminate_if_greater_damage

        # TODO: Define `benchmark_sim` here, and make it required (ie. never None).
        if self.benchmark_sim:
            # FIXME: Remove this once external env seeding is fully integrated.
            # Validate that benchmark and sim match seeds
            assert self.sim.get_seeds() == self.benchmark_sim.get_seeds()

            # Store the firemaps from the benchmark simulation if used in the state.
            if "bench_fire_map" in self.attributes:
                # Create static list to store the episode benchsim firemaps
                self._max_bench_length = max_bench_length
                self._bench_firemaps = [0] * self._max_bench_length

        # TODO: Make this check more general, ie. not included in every subclass?
        # Validate that the attributes provided are supported by this harness.
        curr_cls = self.__class__
        bad_attributes = get_unsupported_attributes(self.attributes, curr_cls)
        if bad_attributes:
            msg = (
                f"The {curr_cls.__name__} class does not support the "
                f"following attributes: {bad_attributes}."
            )
            raise AssertionError(msg)

        if self.num_agents > 1:
            msg = f"{self.__class__.__name__} only supports a single agent."
            raise NotImplementedError(msg)

    def _should_terminate(self) -> bool:
        # Retrieve original value, based on `FireHarness` definition of terminated.
        terminated = super()._should_terminate()

        # Terminate episode early if burn damage in Agent Sim > final bench firemap
        if self.benchmark_sim:
            if self._terminate_if_greater_damage:
                total_area = self.sim.fire_map.size

                sim_data = self.harness_analytics.sim_analytics.data
                sim_damaged_total = sim_data.burned + sim_data.burning
                benchsim_data = self.harness_analytics.benchmark_sim_analytics.data
                benchsim_damaged_total = total_area - benchsim_data.unburned

                if sim_damaged_total > benchsim_damaged_total:
                    terminated = True
                    # TODO potentially add a static negative penalty for making the fire worse

        return terminated

    def _update_state(self) -> None:
        super()._update_state()

        # Modify the state to contain the bench fire map at that sim step
        if "bench_fire_map" in self.attributes:
            bench_fire_map_idx = self.attributes.index("bench_fire_map")
            # if the simulation has lasted longer that the benchmark sim, use the final state of the benchsim fire map
            if (
                self.harness_analytics.benchmark_sim_analytics.num_sim_steps
                < self.harness_analytics.sim_analytics.num_sim_steps
            ):
                self.state[..., (bench_fire_map_idx)] = self._bench_firemaps[
                    (self.harness_analytics.benchmark_sim_analytics.num_sim_steps) - 1
                ]
            # else get the benchmark sim fire map from the same sim step as the simulation fire map
            else:
                self.state[..., (bench_fire_map_idx)] = self._bench_firemaps[
                    (self.harness_analytics.sim_analytics.num_sim_steps) - 1
                ]

        # Modify the state to contain the final state of bench fire map
        if "bench_fire_map_final" in self.attributes:
            bench_fire_map_final_idx = self.attributes.index("bench_fire_map_final")
            self.state[..., (bench_fire_map_final_idx)] = self._bench_firemaps[
                (self.harness_analytics.benchmark_sim_analytics.num_sim_steps) - 1
            ]

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, dict]:
        """Reset environment to an initial state, returning an initial obs and info."""
        # Use the following line to seed `self.np_random`
        super().reset(seed=seed, options=options)

        # Run the new benchsim to completion to obtain the benchsim data used to generate
        # the rewards and policy
        self.benchmark_sim.reset()
        # TODO: Only call _run_benchmark if fire scenario differs from previous episode.
        # This is somewhat tricky - must ensure that analytics.data is NOT reset!
        self._run_benchmark()

        return self.state, {}

    def _run_benchmark(self):
        """Run the entire benchmark sim and store the data needed for the rewards and bench fire maps within each episode."""
        # TODO: We can remove the benchmark_sim entirely, and get this behavior by simply
        # running self.sim until it terminates, then reset self.sim to the initial state.

        # if self.sim.elapsed_steps > 0:
        if self.benchmark_sim.elapsed_steps > 0:
            raise RuntimeError(
                "Benchmark simulation must be reset before running it again."
            )

        timesteps = 0
        while self.benchmark_sim.active:
            run_sim = timesteps % self.agent_speed == 0

            if run_sim:
                # TODO: Refactor logic into a method, and call it here.
                # Run for one timestep, then update respective metrics.
                self.benchmark_sim.run(1)
                # FIXME: This method call is VERY redundant (see method logic)
                self.harness_analytics.update_bench_after_one_simulation_step(
                    timestep=timesteps
                )

                curr_step = self.harness_analytics.benchmark_sim_analytics.num_sim_steps
                # Store the bench fire map at the sim step
                if curr_step < self._max_bench_length - 1:
                    self._bench_firemaps[curr_step - 1] = np.copy(
                        self.benchmark_sim.fire_map
                    )
                else:
                    self._bench_firemaps.append(np.copy(self.benchmark_sim.fire_map))
                    self._max_bench_length += 1

            timesteps += 1

    def get_nonsim_attribute_data(self) -> OrderedDict[str, np.ndarray]:  # noqa
        nonsim_data = super().get_nonsim_attribute_data()
        fire_map_kwargs = {"place_agents": False, "set_agent_positions": False}
        nonsim_data["bench_fire_map"] = self.prepare_fire_map(**fire_map_kwargs)
        nonsim_data["bench_fire_map_final"] = self.prepare_fire_map(**fire_map_kwargs)
        return nonsim_data

    def get_nonsim_attribute_bounds(self) -> OrderedDict[str, Dict[str, int]]:  # noqa
        nonsim_min_maxes = super().get_nonsim_attribute_bounds()
        bench_values = self._get_non_interaction_disaster_categories().values()
        min_max_dict = {"min": min(bench_values), "max": max(bench_values)}
        # TODO: Verify that using same min-max dict ref for both bench fire maps is ok.
        nonsim_min_maxes.update(
            {"bench_fire_map": min_max_dict, "bench_fire_map_final": min_max_dict}
        )
        return nonsim_min_maxes

    @staticmethod
    def supported_attributes() -> List[str]:
        return FireHarness.supported_attributes() + BENCHMARK_ATTRIBUTES
