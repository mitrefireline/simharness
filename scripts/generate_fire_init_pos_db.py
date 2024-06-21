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

sh_path = os.path.join(os.environ["HOME"], "simharness")
if sh_path not in sys.path:
    sys.path.append(sh_path)

import hydra
import ray
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from simfire.sim.simulation import Simulation, FireSimulation
from simfire.utils.config import Config
from simharness2.environments.utils import create_fire_simulation_from_config

import simharness2.environments.utils as env_utils


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
    config_path=f"{os.environ['HOME']}/simharness/conf",
    config_name="config",
)
def main(cfg: DictConfig) -> None:
    """Generate the fire init position database.

    Args:
        cfg (DictConfig): Hydra config with all required parameters for training.
    """
    # Start the Ray runtime
    ray.init(address="local", num_cpus=80)

    outdir = os.path.join(cfg.run.storage_path, HydraConfig.get().output_subdir)
    logger.info(f"Configuration files for this job can be found at {outdir}.")
    logger.info(f"Stdout for this job can be found at {outdir}/")  # FIXME!!

    executed_command = " ".join(["%s" % arg for arg in sys.argv])
    logger.info(f"Executed command: \n{executed_command}")

    # FIXME: If the env_config has an operational location that will not be used, we end
    # up (possibly) downloading unnecessary data and building the sim object, which will
    # just be updated later anyways. Okay for now, but fix later if time permits.
    env_cfg = instantiate(cfg.environment.env_config, _convert_="partial")
    # Use provided simulation info to create a simulation object.
    sim: FireSimulation = create_fire_simulation_from_config(env_cfg.get("sim_init_cfg"))
    # Validate the configuration for the `FireSimulation` object.
    env_utils.check_terrain_is_operational(sim)
    env_utils.check_fire_init_pos_is_static(sim)
    op_locs_cfg = env_cfg.get("operational_locations")
    fire_pos_cfg = env_cfg.get("fire_initial_position")
    env_utils.validate_fire_init_config(fire_pos_cfg, sim.fire_map.size)

    # TODO: Add check to ensure each location is valid. For more info, see:
    # https://github.com/mitrefireline/simfire/blob/0d46451db183a58d209ef789c509f00eca0daedf/simfire/utils/config.py#L306
    # TODO: Do we want to 'track' locations used for training and evaluation?
    # Seed each respective env with the operational locations.
    num_train_locs = op_locs_cfg.get("sample_size").get("train")
    num_eval_locs = op_locs_cfg.get("sample_size").get("eval")
    train_locs, eval_locs = env_utils.get_operational_locations(
        cfg=op_locs_cfg,
        num_train_locs=num_train_locs,
        num_eval_locs=num_eval_locs,
        seed=cfg.debugging.seed,
        fire_year=2020,
    )

    # Retrieve the train/eval data using the provided fire initial position config.
    logger.info(f"Shape of fire map before preparing data: {sim.fire_map.shape}")
    for loc in train_locs + eval_locs:
        logger.info(f"Preparing data for location: {loc}")
        train_data, eval_data = env_utils.prepare_fire_map_data(
            sim,
            fire_pos_cfg,
            location=loc,
            return_train_data=loc in train_locs,
            return_eval_data=loc in eval_locs,
            # TODO: Should we specify a logdir??
            # logdir=??
        )
        logger.info(
            f"Using location {loc} resulted in fire map shape: {sim.fire_map.shape}"
        )


if __name__ == "__main__":
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    main()
