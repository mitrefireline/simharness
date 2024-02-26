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
import sys
import warnings


if f"{os.environ['HOME']}/simharness2" not in sys.path:
    sys.path.append(os.path.join(os.environ["HOME"], "simharness2"))

import hydra
import ray
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from simfire.sim.simulation import FireSimulation

from simharness2.callbacks.initalize_simfire import (
    _check_fire_init_pos_is_static,
    _prepare_fire_map_data,
    _validate_fire_init_config,
)


OmegaConf.register_new_resolver("operational_screen_size", lambda x: int(x * 39))
OmegaConf.register_new_resolver("calculate_half", lambda x: int(x / 2))
OmegaConf.register_new_resolver("square", lambda x: x**2)

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
handler.setFormatter(
    logging.Formatter("%(asctime)s\t%(levelname)s %(filename)s:%(lineno)s -- %(message)s")
)
logger.addHandler(handler)
logger.propagate = False


@hydra.main(
    version_base=None,
    config_path=f"{os.environ['HOME']}/simharness2/conf",
    config_name="config",
)
def main(cfg: DictConfig) -> None:
    """Generate the fire init position database.

    Args:
        cfg (DictConfig): Hydra config with all required parameters for training.
    """
    # Start the Ray runtime
    ray.init(address="auto")

    outdir = os.path.join(cfg.run.storage_path, HydraConfig.get().output_subdir)
    logger.info(f"Configuration files for this job can be found at {outdir}.")
    logger.info(f"Stdout for this job can be found at {outdir}/")  # FIXME!!

    executed_command = " ".join(["%s" % arg for arg in sys.argv])
    logger.info(f"Executed command: \n{executed_command}")

    env_cfg = instantiate(cfg.environment.env_config, _convert_="partial")
    sim: FireSimulation = env_cfg.get("sim")
    _check_fire_init_pos_is_static(sim)
    fire_pos_cfg = env_cfg.get("fire_initial_position")
    _validate_fire_init_config(fire_pos_cfg, sim.fire_map.size)

    # Retrieve the train/eval data using the provided fire initial position config.
    # save_dir =
    train_data, eval_data = _prepare_fire_map_data(sim, fire_pos_cfg)


if __name__ == "__main__":
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    main()
