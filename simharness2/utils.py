import logging
import os
from importlib import import_module
from typing import Any, Dict, Tuple, TYPE_CHECKING

from hydra.utils import instantiate
from omegaconf import OmegaConf

from simharness2.config import SimHarnessConfig

if TYPE_CHECKING:
    from ray.rllib.algorithms.algorithm_config import AlgorithmConfig

# FIXME: Use same logger name convention across project
LOGGER = logging.getLogger(__name__)


def set_variable_hyperparameters(
    algo_cfg: "AlgorithmConfig", cfg: SimHarnessConfig
) -> None:
    """Override the algo_cfg hyperparameters we would like to tune over.

    Args:
        algo_cfg (AlgorithmConfig): Config used for training our model.
        cfg (DictConfig): Hydra config with all required parameters.
    """
    from ray.tune import loguniform, uniform, randint, choice

    tunables = OmegaConf.to_container(cfg.tunables, resolve=True)

    for section_key, param_dict in tunables.items():
        for key, value in param_dict.items():
            if value["type"] == "loguniform":
                sampler = loguniform(value["values"][0], value["values"][1])
            elif value["type"] == "uniform":
                sampler = uniform(value["values"][0], value["values"][1])
            elif value["type"] == "random":
                sampler = randint(value["values"][0], value["values"][1])
            elif value["type"] == "choice":
                sampler = choice(value["values"])
            else:
                LOGGER.error(f"Invalid value type {value['type']} given - skipping.")

            tunables[section_key][key] = sampler

    algo_cfg.training(**tunables["training"])


def build_algo_cfg(cfg: SimHarnessConfig) -> "AlgorithmConfig":
    """Build the algorithm config and object for training an RLlib model.

    Args:
        cfg (DictConfig): Hydra config with all required parameters.

    Returns:
        Tuple(Algorithm, AlgorithmConfig): Training algorithm and associated config.
    """
    # FIXME: Callbacks should be modularized and specified from the config.
    # FIXME: Below import is required to register custom model (s). Find better way.
    import simharness2.models  # noqa
    from ray.tune.registry import get_trainable_cls
    from simharness2.callbacks.render_env import RenderEnv

    # Instantiate everything necessary for creating the algorithm config.
    env_settings, eval_settings, debug_settings, explore_cfg = instantiate_config(cfg)

    trainable_cls = get_trainable_cls(cfg.trainable_class)
    default_cfg: AlgorithmConfig = trainable_cls.get_default_config()
    algo_cfg = (
        default_cfg.training(**cfg.training)
        .environment(**env_settings)
        .framework(**cfg.framework)
        .rollouts(**cfg.rollouts)
        .evaluation(**eval_settings)
        .exploration(explore=cfg.exploration.explore, exploration_config=explore_cfg)
        .resources(**cfg.resources)
        .debugging(**debug_settings)
        .callbacks(RenderEnv)
        # .multi_agent(
        #     policies=agent_ids,
        #     policy_mapping_fn=(lambda agent_id, *args, **kwargs: agent_id),
        # )
    )
    return algo_cfg


def instantiate_config(
    cfg: SimHarnessConfig,  # TODO: Fix return type annotations
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
    from ray.tune.registry import register_env

    # Instantiate the env and eval settings objects from the config.
    # NOTE: We are instantiating to a NEW object on purpose; otherwise a
    # `TypeError` will be raised when attempting to log the cfg to Aim.
    env_settings = instantiate(cfg.environment, _convert_="partial")
    eval_settings = instantiate(cfg.evaluation, _convert_="partial")

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

    debug_settings = instantiate(cfg.debugging, _convert_="all")

    # Register the environment with Ray
    # NOTE: Assume that same environment cls is used for training and evaluation.
    # TODO: This blocks us from being able to have `view()` can we change this?
    env_module, env_cls = cfg.environment.env.rsplit(".", 1)
    env_cls = getattr(import_module(env_module), env_cls)
    register_env(cfg.environment.env, lambda config: env_cls(**config))

    return env_settings, eval_settings, debug_settings, exploration_cfg
