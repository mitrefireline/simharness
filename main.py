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

import gymnasium as gym
import hydra
import ray
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from ray import air, tune
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
from ray.rllib.utils.typing import ResultDict
from ray.tune.logger import pretty_print
from ray.tune.registry import get_trainable_cls, register_env
from simfire.sim.simulation import Simulation  # noqa: F401

from simharness2.logger.aim import AimLoggerCallback
from simharness2.utils.evaluation_fires import get_default_operational_fires
from simharness2.callbacks.set_env_seeds_callback import SetEnvSeedsCallback

os.environ["HYDRA_FULL_ERROR"] = "1"
# Register custom resolvers that are used within the config files
OmegaConf.register_new_resolver("operational_screen_size", lambda x: int((x / 64) * 1920))
OmegaConf.register_new_resolver("calculate_half", lambda x: int(x / 2))


def train_with_tune(algo_cfg: AlgorithmConfig, cfg: DictConfig) -> ResultDict:
    """FIXME: Docstring for train_with_tune."""
    # automated run with Tune and grid search and TensorBoard
    tuner = tune.Tuner(
        cfg.algo.name,
        param_space=algo_cfg,
        # TODO add `tune_config` argument with `tune.TuneConfig`
        run_config=air.RunConfig(
            name=cfg.runtime.name or None,
            local_dir=cfg.runtime.local_dir,
            # callbacks=[
            #     AimLoggerCallback(
            #         repo="/home/jovyan/aim",
            #         experiment="aim_test",
            #         system_tracking_interval=None,
            #         log_system_params=False,
            #     )
            # ],
            stop={**cfg.stop_conditions},
            callbacks=[AimLoggerCallback(cfg=cfg, **cfg.aim)],
            failure_config=None,
            sync_config=tune.SyncConfig(syncer=None),  # Disable syncing
            checkpoint_config=air.CheckpointConfig(**cfg.checkpoint),
        ),
    )
    results = tuner.fit()
    return results


def train(algo: Algorithm, cfg: DictConfig, log: logging.Logger):
    """FIXME: Docstring for train."""
    stop_cond = cfg.stop_conditions
    # Run manual training loop and print results after each iteration
    for i in range(stop_cond.training_iteration):
        log.info(f"Training iteration {i}")
        result = algo.train()
        log.info(pretty_print(result))

        if i % cfg.checkpoint.checkpoint_frequency == 0:
            ckpt_path = algo.save()
            log.info(f"A checkpoint has been created inside directory: {ckpt_path}.")

        if (
            result["timesteps_total"] >= stop_cond.timesteps_total
            or result["episode_reward_mean"] >= stop_cond.episode_reward_mean
        ):
            log.info(f"Training stopped short at iteration {i}")
            ts = result["timesteps_total"]
            mean_rew = result["episode_reward_mean"]
            log.info(f"Timesteps: {ts}\nEpisode_Mean_Rewards: {mean_rew}")
            break

    model_path = algo.save()
    log.info(f"The final model has been saved inside directory: {model_path}.")
    algo.stop()


def view(algo: Algorithm, cfg: DictConfig, view_sim: Simulation, log: logging.Logger):
    """FIXME: Docstring for view."""
    log.info("Collecting gifs of trained model...")
    env_name = cfg.evaluation.evaluation_config.env

    env_cfg = OmegaConf.to_container(cfg.environment.env_config)
    env_cfg.update({"simulation": view_sim})

    env = gym.make(env_name, **env_cfg)

    for _ in range(2):
        env.simulation.rendering = True
        obs, _ = env.reset()
        done = False

        fire_loc = env.simulation.fire_manager.init_pos
        agent_pos = env.agent_pos  # type: ignore
        info = f"Agent Start Location: {agent_pos}, Fire Start Location: {fire_loc}"

        total_reward = 0.0
        while not done:
            action = algo.compute_single_action(obs)

            obs, reward, done, _, _ = env.step(action)
            total_reward += reward
        info = info + f", Final Reward: {total_reward}"
        log.info(info)

        head_path, checkpoint_dir = os.path.split(cfg.algo.checkpoint_path)
        save_dir = os.path.join(head_path, "gifs", checkpoint_dir)
        env.simulation.save_gif(save_dir)
        env.simulation.rendering = False


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    """FIXME: Docstring for main."""
    # Start the Ray runtime
    ray.init()
    # Fetch logger, which is configured in `conf/hydra/job_logging`
    log = logging.getLogger(__name__)
    outdir = os.path.join(cfg.runtime.local_dir, HydraConfig.get().output_subdir)
    log.warning(f"Configuration files for this job can be found at {outdir}")

    # assume for now that operational fires are the default
    operational_fires = get_default_operational_fires(cfg)

    model_available = False
    if cfg.algo.checkpoint_path:
        log.info(f"Creating an algorithm instance from {cfg.algo.checkpoint_path}")
        # TODO raise error if checkpoint_path is not a valid path
        algo = Algorithm.from_checkpoint(cfg.algo.checkpoint_path)
        model_available = True

    if cfg.cli.mode == "train" or cfg.cli.mode == "tune":
        log.info(f"Training model on {cfg.environment.env}")
        if not model_available:
            # Instantiate objects based on the provided settings
            # TODO: Move this to a utility function (ie `instantiate_from_config`)?
            # NOTE: We are instantiating to a NEW object on purpose; otherwise a
            # `TypeError` will be raised when attempting to log the cfg to Aim.
            env_settings = instantiate(cfg.environment, _convert_="partial")
            eval_settings = instantiate(cfg.evaluation, _convert_="partial")
            # TODO: Move (both) NOTE below to docs and remove from code
            # NOTE: Need to convert OmegaConf container to dict to avoid `TypeError`.
            # Prepare exploration options for the algorithm
            explore = cfg.exploration.explore
            exploration_cfg = OmegaConf.to_container(cfg.exploration.exploration_config)

            # Prepare debugging settings
            # If no `type` is given, tune's `UnifiedLogger` is used as follows:
            # DEFAULT_LOGGERS = (JsonLogger, CSVLogger, TBXLogger)
            # `UnifiedLogger(config, self._logdir, loggers=DEFAULT_LOGGERS)`
            # - The `logger_config` defined below is used here:
            # https://github.com/ray-project/ray/blob/863928c4f13b66465399d63e01df3c446b4536d9/rllib/algorithms/algorithm.py#L423
            # - The `Trainable._create_logger` method can be found here:
            # https://github.com/ray-project/ray/blob/8d2dc9a3997482100034b60568b06aad7fd9fc59/python/ray/tune/trainable/trainable.py#L1067
            debug_settings = OmegaConf.to_container(cfg.debugging)
            # TODO make options passed to `logger_config` configurable from the CLI
            debug_settings.update(
                {
                    "logger_config": {
                        "type": tune.logger.TBXLogger,
                        "logdir": cfg.runtime.local_dir,
                    }
                }
            )
            # Register the environment with Ray
            # TODO: Move this to a function (ie `register_env`)?
            # NOTE: Assume that same environment cls is used for training and evaluation.
            env_module, env_cls = cfg.environment.env.rsplit(".", 1)
            env_cls = getattr(import_module(env_module), env_cls)
            register_env(cfg.environment.env, lambda config: env_cls(config))

            # Build the `AlgorithmConfig` object using the provided settings.
            algo_cfg = (
                get_trainable_cls(cfg.algo.name)
                .get_default_config()
                .training(**cfg.training)
                .environment(**env_settings)
                .framework(**cfg.framework)
                .rollouts(**cfg.rollouts)
                .evaluation(**eval_settings)
                .exploration(explore=explore, exploration_config=exploration_cfg)
                .resources(**cfg.resources)
                .debugging(**debug_settings)
                .callbacks(SetEnvSeedsCallback)
            )

            if cfg.cli.mode == "tune":
                train_with_tune(algo_cfg, cfg)
            else:
                algo = algo_cfg.build()
                model_available = True

                train(algo, cfg, log)

    # elif cfg.cli.mode == "view":
    #     if not model_available:
    #         raise ValueError("No model is available for viewing.")

    #     view(algo, cfg, sim(view_cfg))
    else:
        raise ValueError(f"Invalid mode: {cfg.cli.mode}")

    ray.shutdown()


if __name__ == "__main__":
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    main()
