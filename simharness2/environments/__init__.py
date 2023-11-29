from .fire_harness import AnyFireSimulation, FireHarness, ReactiveHarness
from .harness import AnySimulation, Harness
from .multi_agent_fire_harness import MultiAgentFireHarness

# from .reactive import ReactiveHarness
from .reactive_marl import MARLReactiveHarness
from .rl_harness import RLHarness

__all__ = [
    "AnyFireSimulation",
    "AnySimulation",
    "FireHarness",
    "Harness",
    "MultiAgentFireHarness",
    "ReactiveHarness",
    "MARLReactiveHarness",
    "RLHarness",
]
# from ray.tune.registry import register_env
# from ray.rllib.env import EnvContext

# import gymnasium as gym
# from gymnasium.envs.registration import register


# def marl_reactive_harness_env_creator(env_config: EnvContext) -> MARLReactiveHarness:
#     """Environment creator for RLlib.

#     Arguments:
#         env_config: A dictionary containing the environment configuration.

#     Returns:
#         An instance of the ReactiveHarness (environment) class.
#     """
#     # register(
#     #     id="ReactiveHarness-v0",
#     #     entry_point="simharness2.environments.reactive:ReactiveHarness",
#     #     max_episode_steps=2000,  # TODO make this a configurable parameter
#     # )
#     return MARLReactiveHarness(env_config)


# register_env(
#     name="marl_reactive_harness_env", env_creator=marl_reactive_harness_env_creator
# )
