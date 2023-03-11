# noqa : D212,D415
"""
To avoid an ImportError and/or ModueNotFoundError, run this script as a module:

python -m simharness2.environments.tests.check_reactive_environments \
    --config <path_to_config_file> --env-type <train|eval>

(above command should be executed from the root of the repository)
"""
import argparse
import logging
import os
from typing import Any, Dict

import yaml
from ray.rllib.utils.pre_checks.env import check_gym_environments

from simharness2.environments.reactive import (
    ReactiveDiscreteHarness,
    ReactiveHarness,
)
from simharness2.sim_registry import get_simulation_from_name


def setup_args():
    """Parse command line options (mode and config)."""
    parser = argparse.ArgumentParser(description="Test custom environment with RLlib.")
    help_s = "Path to (harness) config file."
    parser.add_argument("--config", required=True, type=str, help=help_s)
    help_s, choices = "Environment type.", ["train", "eval"]
    parser.add_argument(
        "--env-type", required=True, type=str, help=help_s, choices=choices
    )
    return parser.parse_args()


def get_config(cfg_path: str) -> Dict[str, Any]:
    """Load the YAML config file from the given path.

    Arguments:
        cfg_path: A string indicating the file path to load the YAML file from.

    Returns:
        A dictionary containing the contents of the YAML configuration file.
    """
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)


def reactive_multidiscrete_env_creator(env_config: Dict[str, Any]) -> ReactiveHarness:
    """Environment creator for RLlib.

    Arguments:
        env_config: A dictionary containing the environment configuration.

    Returns:
        An instance of the ReactiveHarness (environment) class.
    """
    return ReactiveHarness(**env_config)


def reactive_discrete_env_creator(env_config: str) -> ReactiveDiscreteHarness:
    """Environment creator for RLlib.

    Arguments:
        env_config: A dictionary containing the environment configuration.

    Returns:
        An instance of the ReactiveDiscreteHarness (environment) class.
    """
    return ReactiveDiscreteHarness(**env_config)


def main():  # noqa: D103
    args = setup_args()
    config = get_config(args.config)

    # Initialize ray in local mode for easier debugging.
    # ray.init(local_mode=True, num_gpus=0, num_cpus=4)

    # Get the simulator and configs from the simulation registry
    sim, train_config, eval_config = get_simulation_from_name(config["SIMULATION"])

    # Prepare kwargs to pass to the custom environment constructor.
    if args.env_type == "eval":
        # Make a deterministic environment to use for evaluation.
        env_cfg = {
            "simulation": sim(eval_config),
            "deterministic": True,
            **config["RLHARNESS"],
        }
    else:
        env_cfg = {"simulation": sim(train_config), **config["RLHARNESS"]}

    reactive_env = reactive_multidiscrete_env_creator(env_cfg)
    check_gym_environments(reactive_env)

    reactive_discrete_env = reactive_discrete_env_creator(env_cfg)
    check_gym_environments(reactive_discrete_env)


if __name__ == "__main__":
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    numba_logger = logging.getLogger("numba")
    numba_logger.setLevel(logging.WARNING)
    try:
        main()
    except Exception as e:
        import traceback

        print(f"SimHarness has crashed unexpectedly with a {repr(e)}")
        print(e)
        traceback.print_exc()
