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
from typing import Callable, Dict, Optional, OrderedDict, Tuple, TypeVar

import numpy as np
from gymnasium import spaces
from ray.rllib.env import MultiAgentEnv
from ray.rllib.utils.typing import MultiAgentDict
from simfire.sim.simulation import FireSimulation

from simharness2.environments.fire_harness import FireHarness


logger = logging.getLogger(__name__)

AnyFireSimulation = TypeVar("AnyFireSimulation", bound=FireSimulation)


class MultiAgentFireHarness(FireHarness[AnyFireSimulation], MultiAgentEnv):
    """TODO."""

    # Provide full (preferred format) observation- and action-spaces as Dicts
    # mapping agent IDs to the individual agents' spaces.
    action_space: spaces.Dict
    observation_space: spaces.Dict

    def get_action_space(self, action_space_cls: Callable) -> spaces.Dict:
        if action_space_cls is spaces.Discrete:
            input_arg = len(self.movements) * len(self.interactions)
        elif action_space_cls is spaces.MultiDiscrete:
            input_arg = [len(self.movements), len(self.interactions)]
        else:
            raise NotImplementedError

        self._action_space_in_preferred_format = True
        agent_action_space = action_space_cls(input_arg)
        return spaces.Dict({agent_id: agent_action_space for agent_id in self._agent_ids})

    def get_observation_space(self) -> spaces.Dict:
        """TODO."""
        # # NOTE: calling `reshape()` to switch to channel-minor format.
        # self._channel_lows = np.array(
        #     [[[self.min_maxes[channel]["min"]]] for channel in self.attributes]
        # ).reshape(1, 1, len(self.attributes))
        # self._channel_highs = np.array(
        #     [[[self.min_maxes[channel]["max"]]] for channel in self.attributes]
        # ).reshape(1, 1, len(self.attributes))

        # obs_shape = (
        #     self.sim.fire_map.shape[0],
        #     self.sim.fire_map.shape[1],
        #     len(self.attributes),
        # )
        # low = np.broadcast_to(self._channel_lows, obs_shape)
        # high = np.broadcast_to(self._channel_highs, obs_shape)
        # FIXME: Verify the super() call works as desired!
        agent_obs_space = super().get_observation_space()
        self._obs_space_in_preferred_format = True
        return spaces.Dict({agent_id: agent_obs_space for agent_id in self._agent_ids})

    def step(
        self, action_dict: MultiAgentDict
    ) -> Tuple[
        MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict
    ]:
        """Returns observations from ready agents."""
        # TODO: Can we parallelize this method? If so, how? I'm not sure if that
        # will make sense wrt updating the sim, etc.?
        for agent_id, agent in self.agents.items():
            self._do_one_agent_step(agent=agent, action=action_dict[agent_id])

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

        # Calculate the reward for the current timestep
        # TODO pass `terminated` into `get_reward` method
        # FIXME: Update reward for MARL case!!
        # TODO: Give each agent the "same" simple reward for now.
        reward = self.reward_cls.get_reward(self.timesteps, sim_run)

        # Terminate episode early if burn damage in Agent Sim is larger than final bench fire map
        self._terminate_if_greater_damage = False #FIXME get rid of this line when the fixme in the if statement below is implemented
        if self.benchmark_sim:
            if self._terminate_if_greater_damage:
                total_area = self.sim.fire_map.size

                sim_damaged_total = self.harness_analytics.sim_analytics.data.burned + self.harness_analytics.sim_analytics.data.burning
                # FIXME Fix this damage calculation if needed to account for damage across all the agent sims
                benchsim_damaged_total = total_area - self.harness_analytics.benchmark_sim_analytics.data.unburned

                if sim_damaged_total > benchsim_damaged_total:
                    terminated = True
                    # TODO potentially add a static negative penalty for making the fire worse

        # TODO account for below updates in the reward_cls.calculate_reward() method
        # "End of episode" reward
        if terminated:
            reward += 10

        if self.harness_analytics:
            self.harness_analytics.update_after_one_harness_step(
                sim_run, terminated, reward, timestep=self.timesteps
            )

        rewards, truncateds, terminateds, infos = {}, {}, {}, {}
        truncs = set()
        terms = set()
        for agent_id, agent in self.agents.items():
            # FIXME: All agents receive the SAME reward !!!
            rewards[agent_id] = reward
            # FIXME: Trunc/Term logic is the SAME for all agents.
            # We may not always want this, but it's a good starting point.
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

        return self.state, rewards, terminateds, truncateds, infos

    def _parse_action(self, action: np.ndarray) -> Tuple[int, int]:
        """Parse the action into movement and interaction."""
        # NOTE: Assuming that all agents are homogeneous
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

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[MultiAgentDict, MultiAgentDict]:
        """TODO."""
        # NOTE: Since super().reset() calls `self.get_initial_state()`, we can simply
        # return `initial_state` and `infos` as is.
        initial_state, infos = super().reset(seed=seed, options=options)
        return initial_state, infos

    # FIXME: Naive way of disabling the usage of min/maxes in harness. This will be
    # addressed in a future MR that refactors use of normalizing and min/maxes.
    def _get_min_maxes(self) -> OrderedDict[str, Dict[str, Tuple[int, int]]]:
        return {}
