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
from hydra.core.hydra_config import HydraConfig

os.environ["HYDRA_FULL_ERROR"] = "1"

from simharness2.config import SimHarnessConfig
from simharness2.config import register_configs
from simharness2 import utils

if TYPE_CHECKING:
    from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
    from ray.tune.result_grid import ResultGrid


register_configs()
# FIXME: Use same logger name convention across project
LOGGER = logging.getLogger(__name__)


def train_with_tune(algo_cfg: "AlgorithmConfig", cfg: SimHarnessConfig) -> "ResultGrid":
    """Iterate through combinations of hyperparameters to find optimal training runs.

    Args:
        algo_cfg (AlgorithmConfig): Algorithm config for RLlib.
        cfg (DictConfig): Hydra config with all required parameters.

    Returns:
        ResultGrid: Set of Results objects from running Tuner.fit()
    """
    from ray import air, tune
    from simharness2.logger.aim import AimLoggerCallback

    trainable_algo_str = cfg.trainable_class
    param_space = algo_cfg

    # Override the variables we want to tune on
    if cfg.tunables:
        utils.set_variable_hyperparameters(algo_cfg=param_space, cfg=cfg)

    # Configs for this specific trial run
    run_config = air.RunConfig(
        name=cfg.run.name,
        storage_path=cfg.run.storage_path,
        # failure_config=
        checkpoint_config=air.CheckpointConfig(**cfg.checkpoint),
        stop={**cfg.stop_conditions},
        callbacks=[AimLoggerCallback(cfg=cfg, **cfg.aim)],
        failure_config=None,
        sync_config=tune.SyncConfig(syncer=None),  # Disable syncing
    )

    # TODO make sure 'reward' is reported with tune.report()
    # TODO add this to config
    # Config for the tuning process (used for all trial runs)
    # tune_config = tune.TuneConfig(num_samples=4)

    # Create a Tuner
    tuner = tune.Tuner(
        trainable=trainable_algo_str,
        param_space=param_space,
        run_config=run_config,
        # tune_config=tune_config,
    )

    results = tuner.fit()
    result_df = results.get_dataframe()

    logging.info(result_df)
    return results


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: SimHarnessConfig) -> None:
    """Main entry-point for training a SimHarness model with RLlib.

    Args:
        cfg (DictConfig): Hydra config with all required parameters for training.
    """
    import ray

    # Start the Ray runtime
    ray.init(**cfg.ray_init)

    outdir = os.path.join(cfg.run.storage_path, HydraConfig.get().output_subdir)
    LOGGER.info(f"Configuration files for this job can be found at {outdir}.")

    # Build the algorithm config.
    algo_cfg = utils.build_algo_cfg(cfg)

    if cfg.mode == "train":
        from simharness2.train import train

        algo = algo_cfg.build()
        if cfg.checkpoint_path:
            ckpt_path = cfg.checkpoint_path
            LOGGER.info(f"Creating an algorithm instance from {ckpt_path}.")

            if not os.path.isfile(ckpt_path):
                raise ValueError(f"{ckpt_path} is not a valid file path.")

            algo.restore(checkpoint_path=ckpt_path)

        LOGGER.info(f"Training model on {cfg.environment.env}.")
        train(algo, cfg)

    if cfg.mode == "tune":
        LOGGER.info(f"Tuning model on {cfg.environment.env}.")
        train_with_tune(algo_cfg, cfg)

    ray.shutdown()


if __name__ == "__main__":
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    main()
