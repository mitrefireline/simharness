import logging
from collections import OrderedDict
from typing import List, Optional, Tuple, TypeVar

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
        bad_attributes = get_unsupported_attributes(self.attributes, curr_cls)
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
        return [FIRE_MAP_KEY, AGENT_POSITION_KEY]

    def get_initial_state(self) -> np.ndarray:
        """TODO."""
        fire_map = self.prepare_fire_map(place_agents=False)
        fire_map = np.expand_dims(fire_map, axis=-1).astype(np.float32)
        # Build MARL obs - position array will be different for each agent.
        marl_obs = {}
        for ag_id in self._agent_ids:
            curr_agent = self.agents[ag_id]
            pos_state = curr_agent.initial_position
            marl_obs[ag_id] = OrderedDict(
                {
                    FIRE_MAP_KEY: fire_map,
                    AGENT_POSITION_KEY: np.asarray(pos_state),
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
        obs_shape = self.sim.fire_map.shape + (1,)
        return spaces.Box(low=low, high=high, shape=obs_shape, dtype=np.float32)

    def _get_position_observation_space(self) -> spaces.Box:
        """TODO."""
        row_max, col_max = self.sim.fire_map.shape
        return spaces.Box(low=np.array([0, 0]), high=np.array([row_max - 1, col_max - 1]))

    def _update_state(self):
        """Modify environment's state to contain updates from the current timestep."""
        # Copy the fire map from the simulation so we don't overwrite it.
        fire_map = np.expand_dims(np.copy(self.sim.fire_map), axis=-1).astype(np.float32)

        # Build MARL obs - position array will be different for each agent.
        marl_obs = {}
        for ag_id in self._agent_ids:
            curr_agent = self.agents[ag_id]
            pos_state = curr_agent.current_position
            marl_obs[ag_id] = OrderedDict(
                {
                    FIRE_MAP_KEY: fire_map,
                    AGENT_POSITION_KEY: np.asarray(pos_state),
                }
            )
        # Note: Setting to ordered dict bc spaces.Dict.sample() returns ordered dict.
        # self.state = OrderedDict(marl_obs)
        self.state = marl_obs


# FIXME: Consider effect of long time horizons, ie. see "max_bench_length" (this is a
# note to discuss this later).
# FIXME: Discuss harness naming convention with team - I hate how long this is...
class MultiAgentComplexObsDamageAwareReactiveHarness(
    MultiAgentComplexObsReactiveHarness[AnyFireSimulation]
):
    def __init__(
        self,
        *,
        terminate_if_greater_damage: bool = True,
        max_bench_length: int = 600,
        **kwargs,
    ):
        super().__init__(**kwargs)
        # Bool to toggle the ability to terminate the agent simulation early if at the
        # current timestep of the agent simulation, the agents have caused more burn
        # damage (burned + burning) than the final state of the benchmark fire map.
        self._terminate_if_greater_damage = terminate_if_greater_damage

        # TODO: Define `benchmark_sim` here, and make it required (ie. never None).
        # Store the firemaps from the benchmark simulation if used in the state
        if self.benchmark_sim is None:
            # The benchmark_sim is required for rewards and termination!
            raise ValueError(
                "The benchmark simulation must be provided to use this harness."
            )
        else:
            # Create static list to store the episode benchsim firemaps
            self._max_bench_length = max_bench_length
            # FIXME: Suboptimal implementation and should be refactored later.
            self._bench_firemaps = [0] * self._max_bench_length

        # Validate that the reward class provided is supported by this harness.
        reward_cls_name = self.reward_cls.__class__.__name__
        supported_rewards = ["AreaSavedPropReward", "AreaSavedPropRewardV2", "ForwardRewardV2"]
        if reward_cls_name not in supported_rewards:
            # FIXME: Raise a more specific error message.
            msg = (
                f"The {self.__class__.__name__} class does not support usage of the "
                f"{reward_cls_name} reward class."
            )
            raise AssertionError(msg)

    def _should_terminate(self) -> bool:
        # Retrieve original value, based on `FireHarness` definition of terminated.
        terminated = super()._should_terminate()

        # Terminate episode early if burn damage in Agent Sim > final bench firemap
        if self.benchmark_sim:
            if self._terminate_if_greater_damage:
                # breakpoint()
                total_area = self.sim.fire_map.size

                sim_data = self.harness_analytics.sim_analytics.data
                sim_damaged_total = sim_data.burned + sim_data.burning
                benchsim_data = self.harness_analytics.benchmark_sim_analytics.data
                benchsim_damaged_total = total_area - benchsim_data.unburned

                logger.debug(f"sim_damaged_total: {sim_damaged_total}")
                logger.debug(f"benchsim_damaged_total: {benchsim_damaged_total}")
                if sim_damaged_total > benchsim_damaged_total:
                    # TODO: add static negative penalty for making the fire worse?
                    logger.info(
                        "Terminating episode early because the agents have caused more "
                        "burn damage than the final state of the benchmark fire map."
                    )
                    terminated = True

        return terminated

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, dict]:
        """Reset environment to an initial state, returning an initial obs and info."""
        # Use the following line to seed `self.np_random`
        initial_state, infos = super().reset(seed=seed, options=options)

        # Verify initial fire scenario is the same for both the sim and benchsim.
        sim_fire_init_pos = self.sim.config.fire.fire_initial_position
        benchmark_fire_init_pos = self.benchmark_sim.config.fire.fire_initial_position
        if sim_fire_init_pos != benchmark_fire_init_pos:
            raise ValueError(
                "The initial fire scenario for the simulation and benchmark simulation "
                "must be the same."
            )

        # TODO: Only call _run_benchmark if fire scenario differs from previous episode.
        # This is somewhat tricky - must ensure that analytics.data is NOT reset!
        if self._new_fire_scenario:
            # Run new benchsim to completion to obtain data for reward and policy.
            self.benchmark_sim.reset()
            # NOTE: The call below will do a few things:
            #   - Run bench sim to completion (self.benchmark_sim.run(1))
            #   - Update bench sim analytics (update_bench_after_one_simulation_step)
            #   - Store each bench fire map at the sim step in self._bench_firemaps
            self._run_benchmark()
            # Don't rerun benchsim until _initialize_simfire() is called again.
            self._new_fire_scenario = False

        return initial_state, infos

    def _run_benchmark(self):
        """Run benchmark sim until fire propagation is complete."""
        # TODO: We can remove the benchmark_sim entirely, and get this behavior by simply
        # running self.sim until it terminates, then reset self.sim to the initial state.

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
