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
from typing import Callable, Optional, Tuple, TypeVar

import numpy as np
from gymnasium import spaces
from ray.rllib.env import MultiAgentEnv
from ray.rllib.utils.typing import MultiAgentDict
from simfire.sim.simulation import FireSimulation

from simharness2.environments import FireHarness

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
        # TODO: Verify this call to `super().reset()` works as desired!
        initial_state, infos = super().reset(seed=seed, options=options)

        # FIXME: Will need to update creation of `marl_obs` to handle POMDP.
        marl_obs = {ag_id: initial_state for ag_id in self._agent_ids}
        infos = {ag_id: {} for ag_id in self._agent_ids}

        return marl_obs, infos
