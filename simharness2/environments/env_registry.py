from typing import Any, Dict

from gymnasium.envs.registration import register
from ray.tune.registry import register_env

from simharness2.environments.reactive import (
    ReactiveDiscreteHarness,
    ReactiveHarness,
)
from simharness2.environments.rl_harness import RLHarness


def reactive_multidiscrete_env_creator(env_config: Dict[str, Any]) -> RLHarness:
    """Environment creator for RLlib.

    Arguments:
        env_config: A dictionary containing the environment configuration.

    Returns:
        An instance of the ReactiveHarness (environment) class.
    """
    register(
        id="ReactiveHarness-v0",
        entry_point="simharness2.environments.reactive:ReactiveHarness",
        max_episode_steps=2000,  # TODO make this a configurable parameter
    )
    return ReactiveHarness(**env_config)


def reactive_discrete_env_creator(env_config: Dict[str, Any]) -> RLHarness:
    """Environment creator for RLlib.

    Arguments:
        env_config: A dictionary containing the environment configuration.

    Returns:
        An instance of the ReactiveDiscreteHarness (environment) class.
    """
    register(
        id="ReactiveHarness-v1",
        entry_point="simharness2.environments.reactive:ReactiveDiscreteHarness",
        max_episode_steps=2000,  # TODO make this a configurable parameter
    )
    return ReactiveDiscreteHarness(**env_config)


register_env(name="ReactiveHarness-v0", env_creator=reactive_multidiscrete_env_creator)

register_env(name="ReactiveHarness-v1", env_creator=reactive_discrete_env_creator)
