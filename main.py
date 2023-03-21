import argparse
import logging
import os
from typing import Any, Dict

import gymnasium as gym
import ray
import yaml
from ray import air, tune
from ray.rllib.utils.test_utils import check_learning_achieved
from ray.tune.logger import pretty_print
from ray.tune.registry import get_trainable_cls

from simharness2.sim_registry import get_simulation_from_name

import simharness2.environments.env_registry

from omegaconf import DictConfig, OmegaConf
import hydra

# from hydra.core.config_store import ConfigStore

# cs = ConfigStore.instance()
# cs.store(name="reactive", node=ReactiveConfig)

# def setup_args():
#     """Parse command line options (mode and config)."""
#     parser = argparse.ArgumentParser(description="Test custom environment with RLlib.")
#     help_s = "Path to (harness) config file."
#     parser.add_argument("--config", required=True, type=str, help=help_s)
#     help_s, choices = "Environment type.", ["train", "eval"]
#     parser.add_argument(
#         "--env-type", required=True, type=str, help=help_s, choices=choices
#     )
#     parser.add_argument(
#         "--run",
#         type=str,
#         default="PPO",
#         help="The RLlib-registered algorithm to use.",
#     )
#     parser.add_argument(
#         "--stop-iters", type=int, default=50, help="Number of iterations to train."
#     )
#     parser.add_argument(
#         "--stop-timesteps", type=int, default=100000, help="Number of timesteps to train."
#     )
#     parser.add_argument(
#         "--stop-reward", type=float, default=0.1, help="Reward at which we stop training."
#     )
#     parser.add_argument(
#         "--no-tune",
#         action="store_true",
#         help="Run without Tune using a manual train loop instead. In this case,"
#         "use PPO without grid search and no TensorBoard.",
#     )
#     return parser.parse_args()


# def get_config(cfg_path: str) -> Dict[str, Any]:
#     """Load the YAML config file from the given path.

#     Arguments:
#         cfg_path: A string indicating the file path to load the YAML file from.

#     Returns:
#         A dictionary containing the contents of the YAML configuration file.
#     """
#     with open(cfg_path, "r") as f:
#         return yaml.safe_load(f)


# def reactive_multidiscrete_env_creator(env_config: str) -> RLHarness:
#     """Environment creator for RLlib.

#     Arguments:
#         env_config: A dictionary containing the environment configuration.

#     Returns:
#         An instance of the ReactiveHarness (environment) class.
#     """
#     return ReactiveHarness(**env_config)


# def reactive_discrete_env_creator(env_config: str) -> RLHarness:
#     """Environment creator for RLlib.

#     Arguments:
#         env_config: A dictionary containing the environment configuration.

#     Returns:
#         An instance of the ReactiveDiscreteHarness (environment) class.
#     """
#     return ReactiveDiscreteHarness(**env_config)


# def main():
#     args = setup_args()
#     config = get_config(args.config)

#     # Initialize ray in local mode for easier debugging.
#     # FIXME When using `num_gpus=1`, simfire crashes with a CUDA error.
#     ray.init(local_mode=True, num_gpus=0, num_cpus=1)

#     # Get the simulator and configs from the simulation registry
#     sim, train_config, eval_config = get_simulation_from_name(config["SIMULATION"])

#     # Prepare kwargs to pass to the custom environment constructor.
#     if args.env_type == "eval":
#         # Make a deterministic environment to use for evaluation.
#         env_cfg = {
#             "simulation": sim(eval_config),
#             "deterministic": True,
#             **config["RLHARNESS"],
#         }
#     else:
#         env_cfg = {"simulation": sim(train_config), **config["RLHARNESS"]}

#     # Register custom enviornment (s) for use with RLlib.
#     register_env(
#         name="ReactiveHarness-v0", env_creator=reactive_multidiscrete_env_creator
#     )
#     register_env(name="ReactiveHarness-v1", env_creator=reactive_discrete_env_creator)
#     # List of [out_channels, kernel, stride] for each filter.
#     conv_filters = [
#         [16, [8, 8], 2],  # Output: 41x41x16
#         [32, [4, 4], 2],  # Output: 19x19x32
#         [256, [16, 16], 1],  # Output: 9x9x256
#     ]
#     post_fcnet_hiddens = []  # [256]
#     # Input: 64x64x6 -> Output: 15x15x16
#     # Input: 15x15x16 -> Output: 6x6x32
#     # Input: 6x6x32 -> Output:
#     # Create a default configuration for the specified RLlib algorithm.
#     config = (
#         get_trainable_cls(args.run)
#         .get_default_config()
#         .environment("ReactiveHarness-v0", env_config=env_cfg)
#         .framework("torch")
#         .rollouts(num_rollout_workers=1)
#         # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
#         .resources(num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "0")))
#         .training(
#             model={"conv_filters": conv_filters, "post_fcnet_hiddens": post_fcnet_hiddens}
#         )
#     )

#     stop = {
#         "training_iteration": args.stop_iters,
#         "timesteps_total": args.stop_timesteps,
#         "episode_reward_mean": args.stop_reward,
#     }

#     if args.no_tune:
#         # manual training with train loop using PPO and fixed learning rate
#         if args.run != "PPO":
#             raise ValueError("Only support --run PPO with --no-tune.")
#         print("Running manual train loop without Ray Tune.")
#         # use fixed learning rate instead of grid search (needs tune)
#         config.lr = 1e-3
#         algo = config.build()
#         # run manual training loop and print results after each iteration
#         for _ in range(args.stop_iters):
#             result = algo.train()
#             print(pretty_print(result))
#             # stop training of the target train steps or reward are reached
#             if (
#                 result["timesteps_total"] >= args.stop_timesteps
#                 or result["episode_reward_mean"] >= args.stop_reward
#             ):
#                 break
#         algo.stop()
#     else:
#         # automated run with Tune and grid search and TensorBoard
#         print("Training automatically with Ray Tune")
#         tuner = tune.Tuner(
#             args.run,
#             param_space=config.to_dict(),
#             run_config=air.RunConfig(stop=stop),
#         )
#         results = tuner.fit()

#         if args.as_test:
#             print("Checking if learning goals were achieved")
#             check_learning_achieved(results, args.stop_reward)

#     ray.shutdown()


def run():
    return


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    print(cfg)
    ray.init(local_mode=True, num_gpus=0, num_cpus=1)

    sim, train_config, eval_config = get_simulation_from_name(cfg.environment.simulation)
    train_sim = sim(train_config)
    # convert the config to a dictionary
    env_cfg = OmegaConf.to_container(cfg.environment.config)
    # add the (train) simulation object that will be used on environment creation
    env_cfg.update({"simulation": train_sim})

    conv_filters = [
        [16, [8, 8], 2],  # Output: 41x41x16
        [32, [4, 4], 2],  # Output: 19x19x32
        [256, [16, 16], 1],  # Output: 9x9x256
    ]
    post_fcnet_hiddens = []  # [256]

    config = (
        get_trainable_cls(cfg.algo.name)
        .get_default_config()
        .environment(cfg.environment.name, env_config=env_cfg)
        .framework("torch")
        .rollouts(num_rollout_workers=1)
        # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
        .resources(num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "0")))
        .training(
            model={"conv_filters": conv_filters, "post_fcnet_hiddens": post_fcnet_hiddens}
        )
    )

    # use fixed learning rate instead of grid search (needs tune)
    config.lr = 1e-3
    algo = config.build()
    # run manual training loop and print results after each iteration
    for _ in range(10):
        result = algo.train()
        print(pretty_print(result))
    algo.stop()

    ray.shutdown()
    return


if __name__ == "__main__":
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    numba_logger = logging.getLogger("numba")
    numba_logger.setLevel(logging.WARNING)
    main()
