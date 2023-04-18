"""FIXME: A one line summary of the module or program.

Leave one blank line.  The rest of this docstring should contain an
overall description of the module or program.  Optionally, it may also
contain a brief description of exported classes and functions and/or usage
examples.

Typical usage example:

  foo = ClassFoo()
  bar = foo.FunctionBar()
"""
import os

import gymnasium as gym
import hydra
import ray
from omegaconf import DictConfig, OmegaConf
from ray import air, tune
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
from ray.rllib.utils.typing import ResultDict
from ray.tune.logger import pretty_print
from ray.tune.registry import get_trainable_cls
from simfire.sim.simulation import Simulation
from simfire.utils.log import create_logger

import simharness2.environments.env_registry  # noqa
from simharness2.sim_registry import get_simulation_from_name

# from simharness2.logger.aim import AimLoggerCallback
# from ray.rllib.utils.test_utils import check_learning_achieved


os.environ["HYDRA_FULL_ERROR"] = "1"
log = create_logger(__name__)


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
            sync_config=tune.SyncConfig(syncer=None),  # Disable syncing
            checkpoint_config=air.CheckpointConfig(**cfg.checkpoint),
        ),
    )
    results = tuner.fit()
    return results



def train(algo: Algorithm, cfg: DictConfig):
    """FIXME: Docstring for train."""
    stop_cond = cfg.stop_conditions
    # Run manual training loop and print results after each iteration
    for i in range(stop_cond.training_iteration):
        log.info(f"Training iteration {i}")
        result = algo.train()
        log.info(pretty_print(result))

        if i % cfg.checkpoint.frequency == 0:
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


def view(algo: Algorithm, cfg: DictConfig, view_sim: Simulation):
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
    # FIXME decide if/what should be passed to `ray.init()`
    # ray.init(num_gpus=cfg.resources.num_gpus)
    ray.init()

    log.info(f"Loading simulation {cfg.environment.env_config.simulation}...")
    sim, train_cfg, eval_cfg, view_cfg = get_simulation_from_name(
        cfg.environment.env_config.simulation
    )

    model_available = False
    if cfg.algo.checkpoint_path:
        log.info(f"Creating an algorithm instance from {cfg.algo.checkpoint_path}")
        # TODO raise error if checkpoint_path is not a valid path
        algo = Algorithm.from_checkpoint(cfg.algo.checkpoint_path)
        model_available = True

    if cfg.cli.mode == "train" or cfg.cli.mode == "tune":
        log.info(f"Training model on {cfg.environment.env}")
        if not model_available:
            # Convert the immutable configs to mutable dictionaries
            env_settings = OmegaConf.to_container(cfg.environment)
            eval_settings = OmegaConf.to_container(cfg.evaluation)
            # Update the value of `sim` from the name of the requested simulation, ie.
            # `Fire-v0` to the actual simulation object itself, ie. `FireSimulation`.
            # train_sim, eval_sim = sim(train_cfg), sim(eval_cfg)
            env_settings["env_config"].update({"simulation": sim(train_cfg)})
            eval_settings["evaluation_config"]["env_config"].update(
                {"simulation": sim(eval_cfg)}
            )

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
            )

            if cfg.cli.mode == "tune":
                train_with_tune(algo_cfg, cfg)
            else:
                algo = algo_cfg.build()
                model_available = True

                train(algo, cfg)

    elif cfg.cli.mode == "view":
        if not model_available:
            raise ValueError("No model is available for viewing.")

        view(algo, cfg, sim(view_cfg))
    else:
        raise ValueError(f"Invalid mode: {cfg.cli.mode}")

    ray.shutdown()


if __name__ == "__main__":
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    main()
