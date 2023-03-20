from ray.tune.registry import register_env
from simharness2.environments.rl_harness import RLHarness
from simharness2.environments.reactive import (
    ReactiveDiscreteHarness,
    ReactiveHarness,
)

def reactive_multidiscrete_env_creator(env_config: str) -> RLHarness:
    """Environment creator for RLlib.

    Arguments:
        env_config: A dictionary containing the environment configuration.

    Returns:
        An instance of the ReactiveHarness (environment) class.
    """
    return ReactiveHarness(**env_config)


def reactive_discrete_env_creator(env_config: str) -> RLHarness:
    """Environment creator for RLlib.

    Arguments:
        env_config: A dictionary containing the environment configuration.

    Returns:
        An instance of the ReactiveDiscreteHarness (environment) class.
    """
    return ReactiveDiscreteHarness(**env_config)


register_env(
        name="ReactiveHarness-v0", env_creator=reactive_multidiscrete_env_creator
    )

register_env(
        name="ReactiveHarness-v1", env_creator=reactive_discrete_env_creator
    )