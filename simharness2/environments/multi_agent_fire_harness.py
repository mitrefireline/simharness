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
        # Ensure agent (s) start position is rendered on simfire fire_map.
        if self.timesteps == 0:
            agent_positions = [
                (agent.col, agent.row, agent.sim_id) for agent in self.agents.values()
            ]
            self.sim.update_agent_positions(points=agent_positions)
        # TODO: Can we parallelize this method? If so, how? I'm not sure if that
        # will make sense wrt updating the sim, etc.?
        for agent_id in action_dict.keys():
            curr_agent = self.agents.get(agent_id)
            self._do_one_agent_step(agent=curr_agent, action=action_dict[agent_id])

        # After handling ready agents, we need to update progress for busy agents.
        # This is crucial for us to know when the agent is done and needs an action
        # at the next timestep.
        busy_agent_ids = self._agent_ids - set(action_dict.keys())
        self._handle_busy_agents(busy_agent_ids)

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

        if terminated:
            logger.info(
                f"Episode has terminated with {len(busy_agent_ids)} busy agents. "
                "All agents will be 'forced' ready to ensure they are included in the "
                "final observation state returned."
            )
            # Force any busy agents to be ready.
            for agent_id in busy_agent_ids:
                self.agents[agent_id].is_ready = True
            # Re call update state to ensure all agent states are included.
            self._update_state()

            # TODO: We can probably do this somewhere else.
            total_mitigations = 0
            for _, agent in self.agents.items():
                total_mitigations += agent.completed_mitigations

            logger.info(
                f"Total mitigations completed across all agents: {total_mitigations}"
            )

        # Calculate the timestep reward for each agent.
        # TODO: Rllib states that "rewards for indiv agents will be added
        # up to the point where a new action for that agent is needed", so
        # we can include "busy" agents in the rewards dict, if desired.
        rewards = self.reward_cls.get_reward(
            timestep=self.timesteps,
            sim_run=sim_run,
            done_episode=terminated or truncated,
            agents=self.agents,
            agent_speed=self.agent_speed,
        )

        # FIXME: Refactor logic to ensure all rewards return expected types
        if np.isscalar(rewards):
            if self.timesteps < 1:
                logger.warning("Calculated reward value is scalar. Converting to Dict.")
            rewards = {agent_id: rewards for agent_id in self.agents}

        # FIXME: We are passing the TIMESTEP reward, not CUMULATIVE reward!!
        if self.harness_analytics:
            # FIXME: Decide if we should pass all agent rewards. For now, use the sum.
            cumulative_reward = sum(rewards.values())
            self.harness_analytics.update_after_one_harness_step(
                sim_run, terminated, cumulative_reward, timestep=self.timesteps
            )

        # TODO: Override _should_truncate() etc. to return Dict instead of single value.
        truncateds, terminateds, infos = {}, {}, {}
        truncs = set()
        terms = set()
        for agent_id, agent in self.agents.items():
            # Only return those agents' names that require actions at next timestep.
            if not agent.is_ready:
                continue
            # FIXME: Trunc/Term logic is the SAME for all agents.
            # We may not always want this, but it's a good starting point.
            truncateds[agent_id] = truncated
            terminateds[agent_id] = terminated
            infos[agent_id] = {}

            if truncated:
                truncs.add(agent_id)
            if terminated:
                terms.add(agent_id)

        terminateds["__all__"] = len(terms) == self.num_agents
        truncateds["__all__"] = len(truncs) == self.num_agents

        self.timesteps += 1  # increment AFTER method logic is performed (convention).

        # FIXME: The current return of self.state will not account for busy agent (s).
        # We need to update/override self._update_state to fix this.
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

    def _handle_busy_agents(self, agent_ids: set):
        for agent_id in agent_ids:
            agent = self.agents.get(agent_id)

            # Validate that agent's latest interaction is not None.
            if self.interactions[agent.latest_interaction] != "none":
                mitigation_complete = agent.dig_for_one_minute()
                # Once mitigation is complete, the agent is allowed to:
                # - place it on the fire map, and
                # - update it's position on the map.
                if agent.is_ready and mitigation_complete:
                    logger.info(
                        f"Agent {agent.agent_id} has completed the current mitigation. "
                        "The current mitigation count for this agent is: "
                        f"{agent.completed_mitigations}"
                    )
                    self._update_mitigation(agent)
                else:
                    # Overwrite value from previous timestep.
                    agent.mitigation_placed = False

                # Update agent location on map
                move_str = self.movements[agent.latest_movement]
                if agent.is_ready and move_str != "none":
                    logger.info(f"Agent {agent.agent_id} is now moving {move_str}.")
                    self._update_agent_position(agent)
