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
from ray.rllib.algorithms.algorithm import Algorithm

from simharness2.sim_registry import get_simulation_from_name

import simharness2.environments.env_registry

from omegaconf import DictConfig, OmegaConf
import hydra

os.environ["HYDRA_FULL_ERROR"] = "1"

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    logging.info(cfg)
    ray.init()
    
    logging.info(f"Loading simulation {cfg.sim_harness.simulation} for training...")
    sim, train_config, eval_config = get_simulation_from_name(cfg.sim_harness.simulation)
    train_sim = sim(train_config)
    
    # Convert the inmutable config to a mutable dictionary
    env_cfg = OmegaConf.to_container(cfg.sim_harness.config)
    # Add the training simulation object for environment creation
    env_cfg.update({"simulation": train_sim})

    expl_cfg = OmegaConf.to_container(cfg.exploration.exploration_config)

    if cfg.model_load.load_path is not None:
        logging.info(f"Loading previous model from {cfg.model_load.load_path}")
        algo = Algorithm.from_checkpoint(cfg.model_load.load_path)
    else:
        config = (
            get_trainable_cls(cfg.algo.name)
            .get_default_config()
            .environment(cfg.sim_harness.name, env_config=env_cfg)
            .rollouts(**cfg.rollouts)
            .training(**cfg.training)
            .resources(**cfg.resources)
            .framework("torch")
            .exploration(exploration_config=expl_cfg)
        )

        algo = config.build()
    
    stop_cond = cfg.stop_conditions
    # Run manual training loop and print results after each iteration
    for i in range(stop_cond.iterations):
        logging.info(f"Training iteration {i}")
        result = algo.train()
        logging.info(pretty_print(result))
        
        if i % cfg.model_save.checkpoint_freq == 0:
            ckpt_path = algo.save(cfg.model_save.save_dir)
            logging.info(f"A checkpoint has been created inside directory: {ckpt_path}.")
        
        if (result["timesteps_total"] >= stop_cond.timesteps or 
            result["episode_reward_mean"] >= stop_cond.ep_mean_rew):
                logging.info(f"Training stopped short at iteration {i}")
                ts = result["timesteps_total"]
                mean_rew = result["episode_reward_mean"]
                logging.info(f"Timesteps: {ts}\nEpisode_Mean_Rewards: {mean_rew}")
                break
    
    model_path = algo.save(cfg.model_save.save_dir)
    logging.info(f"The final model has been saved inside directory: {model_path}.")
    algo.stop()

    ray.shutdown()
    return


if __name__ == "__main__":
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    numba_logger = logging.getLogger("numba")
    numba_logger.setLevel(logging.WARNING)
    main()
