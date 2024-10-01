"""FIXME: A one line summary of the module or program.

Leave one blank line.  The rest of this docstring should contain an
overall description of the module or program.  Optionally, it may also
contain a brief description of exported classes and functions and/or usage
examples.

Typical usage example:

  foo = ClassFoo()
  bar = foo.FunctionBar()
"""

import logging
import os
from typing import TYPE_CHECKING

import hydra
import ray
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from ray.rllib.algorithms.algorithm import Algorithm

if TYPE_CHECKING:
    from simharness2.callbacks.initialize_simfire import InitializeSimfire


os.environ["HYDRA_FULL_ERROR"] = "1"
# Register custom resolvers that are used within the config files
OmegaConf.register_new_resolver("operational_screen_size", lambda x: int(x * 30))
OmegaConf.register_new_resolver("calculate_half", lambda x: int(x / 2))
OmegaConf.register_new_resolver("square", lambda x: x**2)

LOGGER = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main entry-point for training a SimHarness model with RLlib.

    Args:
        cfg (DictConfig): Hydra config with all required parameters for training.
    """
    # Start the Ray runtime
    ray.init(address="auto")

    hydra_cfg = HydraConfig.get()
    storage_path = hydra_cfg.run.dir
    output_subdir = hydra_cfg.output_subdir
    outdir = os.path.join(storage_path, output_subdir)
    LOGGER.info(f"Configuration files for this job can be found at {outdir}.")

    # Validate provided checkpoint path, then create an algorithm instance from it.
    ckpt_path = cfg.algo.checkpoint_path
    if ckpt_path is not None:
        if not os.path.isdir(ckpt_path):
            raise ValueError(f"{ckpt_path} is not a valid directory path.")

    LOGGER.info(f"Creating an algorithm instance from {ckpt_path}.")
    algo = Algorithm.from_checkpoint(cfg.algo.checkpoint_path)
    algo_callbacks = algo.callbacks._callback_list
    init_simfire = algo_callbacks[0]
    if "InitializeSimfire" not in init_simfire.__class__.__name__:
        raise ValueError("The first callback in the algorithm is not InitializeSimfire!!")

    # FIXME: Update out dir so gifs can be loaded from the flask app after generation.
    # algo.logdir = "NEW LOG DIR"

    # TODO: Replace dummy data with user input from the flask app.
    op_location_type = "eval"  # User selected location type
    selected_op_location = "California_2020_Apple"  # User selected location
    try:
        selected_location = init_simfire.op_locations[op_location_type][
            selected_op_location
        ]

    except KeyError as e:
        LOGGER.error(f"Error: {e}")
        raise ValueError(f"Error: {e}")

    ray.shutdown()


if __name__ == "__main__":
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    main()
