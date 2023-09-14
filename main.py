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
from importlib import import_module
from typing import Any, Dict, Tuple

import hydra
import ray
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from ray import air, tune
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
from ray.tune.logger import pretty_print
from ray.tune.registry import get_trainable_cls, register_env
from ray.tune.result_grid import ResultGrid

# from simharness2.utils.evaluation_fires import get_default_operational_fires
import simharness2.models  # noqa
from simharness2.callbacks.render_env import RenderEnv
from simharness2.logger.aim import AimLoggerCallback

# from simharness2.callbacks.set_env_seeds_callback import SetEnvSeedsCallback

os.environ["HYDRA_FULL_ERROR"] = "1"
# Register custom resolvers that are used within the config files
OmegaConf.register_new_resolver("operational_screen_size", lambda x: int(x * 39))
OmegaConf.register_new_resolver("calculate_half", lambda x: int(x / 2))
OmegaConf.register_new_resolver("square", lambda x: x**2)

LOGGER = logging.getLogger(__name__)


def _set_variable_hyperparameters(algo_cfg: AlgorithmConfig, cfg: DictConfig) -> None:
    """Override the algo_cfg hyperparameters we would like to tune over.

    Args:
        algo_cfg (AlgorithmConfig): Config used for training our model.
        cfg (DictConfig): Hydra config with all required parameters.
    """
    tunables = OmegaConf.to_container(cfg.tunables, resolve=True)

    for section_key, param_dict in tunables.items():
        for key, value in param_dict.items():
            if value["type"] == "loguniform":
                sampler = tune.loguniform(value["values"][0], value["values"][1])
            elif value["type"] == "uniform":
                sampler = tune.uniform(value["values"][0], value["values"][1])
            elif value["type"] == "random":
                sampler = tune.randint(value["values"][0], value["values"][1])
            elif value["type"] == "choice":
                sampler = tune.choice(value["values"])
            else:
                LOGGER.error(f"Invalid value type {value['type']} given - skipping.")

            tunables[section_key][key] = sampler

    algo_cfg.training(**tunables["training"])


def train_with_tune(algo_cfg: AlgorithmConfig, cfg: DictConfig) -> ResultGrid:
    """Iterate through combinations of hyperparameters to find optimal training runs.

    Args:
        algo_cfg (AlgorithmConfig): Algorithm config for RLlib.
        cfg (DictConfig): Hydra config with all required parameters.

    Returns:
        ResultGrid: Set of Results objects from running Tuner.fit()
    """
    trainable_algo_str = cfg.algo.name
    param_space = algo_cfg

    # Override the variables we want to tune on
    if cfg.tunables:
        _set_variable_hyperparameters(algo_cfg=param_space, cfg=cfg)

    # Configs for this specific trial run
    run_config = air.RunConfig(
        name=cfg.runtime.name or None,
        local_dir=cfg.runtime.local_dir,
        stop={**cfg.stop_conditions},
        callbacks=[AimLoggerCallback(cfg=cfg, **cfg.aim)],
        failure_config=None,
        sync_config=tune.SyncConfig(syncer=None),  # Disable syncing
        checkpoint_config=air.CheckpointConfig(**cfg.checkpoint),
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


def train(algo: Algorithm, cfg: DictConfig) -> None:
    """Train the given algorithm within RLlib.

    Args:
        algo (Algorithm): Algorithm to train with.
        cfg (DictConfig): Hydra config with all required parameters for training.
    """
    stop_cond = cfg.stop_conditions
    # Run training loop and print results after each iteration
    for i in range(stop_cond.training_iteration):
        LOGGER.info(f"Training iteration {i}.")
        result = algo.train()
        LOGGER.info(f"{pretty_print(result)}\n")

        if i % cfg.checkpoint.checkpoint_frequency == 0:
            ckpt_path = algo.save()
            log_str = f"A checkpoint has been created inside directory: {ckpt_path}.\n"
            LOGGER.info(log_str)

        if (
            result["timesteps_total"] >= stop_cond.timesteps_total
            or result["episode_reward_mean"] >= stop_cond.episode_reward_mean
        ):
            LOGGER.warning(f"Training stopped short at iteration {i}.\n")
            ts = result["timesteps_total"]
            mean_rew = result["episode_reward_mean"]
            LOGGER.info(f"Timesteps: {ts}\nEpisode_Mean_Rewards: {mean_rew}\n")
            break

    model_path = algo.save()
    LOGGER.info(f"The final model has been saved inside directory: {model_path}.")
    algo.stop()


def _instantiate_config(
    cfg: DictConfig,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Instantiate the algorithm config used to build the RLlib training algorithm.

    Args:
        cfg (DictConfig): Hydra config with all required parameters.

    Returns:
        Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        env_settings: Parameters needed for instantiating the environment
        eval_settings: Parameters needed for running the evaluation code.
        debug_settings: Settings needed for debugging.
        exploration_cfg: RLlib exploration configurations.
    """
    # Instantiate the env and eval settings objects from the config.
    # NOTE: We are instantiating to a NEW object on purpose; otherwise a
    # `TypeError` will be raised when attempting to log the cfg to Aim.
    env_settings = instantiate(cfg.environment, _convert_="partial")
    eval_settings = instantiate(cfg.evaluation, _convert_="partial")

    # FIXME: Fire scenario configuration disabled for now. Fix this in new MR.
    # Get the operational fires we want to run evaluation with
    # operational_fires = get_default_operational_fires(cfg)

    # Inject operational fires into the evaluation settings
    # eval_settings["evaluation_config"]["env_config"].update(
    #     {"scenarios": operational_fires}
    # )

    # Prepare exploration options for the algorithm
    exploration_cfg = OmegaConf.to_container(
        cfg=cfg.exploration.exploration_config, resolve=True
    )

    # If no `type` is given, tune's `UnifiedLogger` is used as follows:
    # DEFAULT_LOGGERS = (JsonLogger, CSVLogger, TBXLogger)
    # `UnifiedLogger(config, self._logdir, loggers=DEFAULT_LOGGERS)`
    # - The `logger_config` defined below is used here:
    # https://github.com/ray-project/ray/blob/863928c4f13b66465399d63e01df3c446b4536d9/rllib/algorithms/algorithm.py#L423
    # - The `Trainable._create_logger` method can be found here:
    # https://github.com/ray-project/ray/blob/8d2dc9a3997482100034b60568b06aad7fd9fc59/python/ray/tune/trainable/trainable.py#L1067

    debug_settings = instantiate(cfg.debugging, _convert_="partial")

    # Register the environment with Ray
    # NOTE: Assume that same environment cls is used for training and evaluation.
    # TODO: This blocks us from being able to have `view()` can we change this?
    env_module, env_cls = cfg.environment.env.rsplit(".", 1)
    env_cls = getattr(import_module(env_module), env_cls)
    register_env(cfg.environment.env, lambda config: env_cls(config))

    return env_settings, eval_settings, debug_settings, exploration_cfg


def _build_algo_cfg(cfg: DictConfig) -> Tuple[Algorithm, AlgorithmConfig]:
    """Build the algorithm config and object for training an RLlib model.

    Args:
        cfg (DictConfig): Hydra config with all required parameters.

    Returns:
        Tuple(Algorithm, AlgorithmConfig): Training algorithm and associated config.
    """
    # Instantiate everything necessary for creating the algorithm config.
    env_settings, eval_settings, debug_settings, explor_cfg = _instantiate_config(cfg)

    algo_cfg = (
        get_trainable_cls(cfg.algo.name)
        .get_default_config()
        .training(**cfg.training)
        .environment(**env_settings)
        .framework(**cfg.framework)
        .rollouts(**cfg.rollouts)
        .evaluation(**eval_settings)
        .exploration(explore=cfg.exploration.explore, exploration_config=explor_cfg)
        .resources(**cfg.resources)
        .debugging(**debug_settings)
        .callbacks(RenderEnv)
    )

    return algo_cfg


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main entry-point for training a SimHarness model with RLlib.

    Args:
        cfg (DictConfig): Hydra config with all required parameters for training.
    """
    # NOTE: We are disabling logging to the driver. For reference, see
    # https://docs.ray.io/en/latest/ray-observability/user-guides/configure-logging.html#disable-logging-to-the-driver
    # Thus, to use an existing ray cluster, we must set address="auto".
    # Start the Ray runtime
    ray.init(address="auto", log_to_driver=False)

    outdir = os.path.join(cfg.runtime.local_dir, HydraConfig.get().output_subdir)
    LOGGER.info(f"Configuration files for this job can be found at {outdir}.")

    # Build the algorithm config.
    algo_cfg = _build_algo_cfg(cfg)

    if cfg.cli.mode == "train":
        algo = algo_cfg.build()
        if cfg.algo.checkpoint_path:
            ckpt_path = cfg.algo.checkpoint_path
            LOGGER.info(f"Creating an algorithm instance from {ckpt_path}.")

            if not os.path.isfile(ckpt_path):
                raise ValueError(f"{ckpt_path} is not a valid file path.")

            algo.restore(checkpoint_path=ckpt_path)

        LOGGER.info(f"Training model on {cfg.environment.env}.")
        train(algo, cfg)

    if cfg.cli.mode == "tune":
        LOGGER.info(f"Tuning model on {cfg.environment.env}.")
        train_with_tune(algo_cfg, cfg)

    ray.shutdown()


if __name__ == "__main__":
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    main()
