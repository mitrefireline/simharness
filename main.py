import logging
import os
from typing import Any, Dict

import gymnasium as gym
import ray
from ray import air, tune
from ray.rllib.utils.test_utils import check_learning_achieved
from ray.tune.logger import pretty_print
from ray.tune.registry import get_trainable_cls
from ray.rllib.algorithms.algorithm import Algorithm

from simharness2.sim_registry import get_simulation_from_name
from simfire.sim.simulation import Simulation

import simharness2.environments.env_registry

from omegaconf import DictConfig, OmegaConf
import hydra

os.environ["HYDRA_FULL_ERROR"] = "1"

def train(algo: Algorithm, cfg: DictConfig):
    stop_cond = cfg.stop_conditions
    save_path = os.path.join(cfg.model_save.save_dir, cfg.model_save.trial_name)
    # Run manual training loop and print results after each iteration
    for i in range(stop_cond.iterations):
        logging.info(f"Training iteration {i}")
        result = algo.train()
        logging.info(pretty_print(result))
        
        if i % cfg.model_save.checkpoint_freq == 0:
            ckpt_path = algo.save(save_path)
            logging.info(f"A checkpoint has been created inside directory: {ckpt_path}.")
        
        if (result["timesteps_total"] >= stop_cond.timesteps or 
            result["episode_reward_mean"] >= stop_cond.ep_mean_rew):
                logging.info(f"Training stopped short at iteration {i}")
                ts = result["timesteps_total"]
                mean_rew = result["episode_reward_mean"]
                logging.info(f"Timesteps: {ts}\nEpisode_Mean_Rewards: {mean_rew}")
                break

    model_path = algo.save(save_path)
    logging.info(f"The final model has been saved inside directory: {model_path}.")
    algo.stop()
    
    
def view(algo: Algorithm, cfg: DictConfig, eval_sim: Simulation):
    logging.info(f"Collecting gifs of trained model...")
    env_name = cfg.sim_harness.name
    
    env_cfg = OmegaConf.to_container(cfg.sim_harness.config)
    env_cfg.update({"simulation": eval_sim})
    
    env = gym.make(env_name, **env_cfg)
    
    for _ in range(2):
        env.simulation.rendering = True
        obs, _ = env.reset()
        done = False

        fire_loc = env.simulation.fire_manager.init_pos
        info = f"Agent Start Location: {env.agent_pos}, Fire Start Location: {fire_loc}"

        total_reward = 0.0
        while not done:
            action = algo.compute_single_action(obs)

            obs, reward, done, _, _ = env.step(action)
            total_reward += reward
        info = info + f", Final Reward: {total_reward}"
        logging.info(info)
        
        save_dir = os.path.join(os.getcwd(), "gifs", cfg.model_save.trial_name)
        env.simulation.save_gif(save_dir)
        env.simulation.rendering = False


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    
    logging.info(cfg)
    ray.init()
    
    logging.info(f"Loading simulation {cfg.sim_harness.simulation}...")
    sim, train_config, eval_config = get_simulation_from_name(cfg.sim_harness.simulation)
    
    model_available = False
    if cfg.model_load.load_path is not None:
        logging.info(f"Loading previous model from {cfg.model_load.load_path}")
        algo = Algorithm.from_checkpoint(cfg.model_load.load_path)
        model_available = True
    
    if cfg.cli.train:
        logging.info(f"Training model on {cfg.sim_harness.name}")
        if not model_available:
            # Convert the inmutable configs to mutable dictionaries
            env_cfg = OmegaConf.to_container(cfg.sim_harness.config)
            expl_cfg = OmegaConf.to_container(cfg.exploration.exploration_config)
            
            train_sim = sim(train_config)
            env_cfg.update({"simulation": train_sim})

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
            model_available = True
            
        train(algo, cfg)
    
    if cfg.cli.view:
        if not model_available:
            raise ValueError("No model is available for viewing.")
        
        eval_sim = sim(eval_config)
        view(algo, cfg, eval_sim)
        
    ray.shutdown()


if __name__ == "__main__":
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    main()
