# Developing

## Configuration Management with Hydra
Configuration management in `simharness` is done using the [Hydra](https://hydra.cc/docs/intro/#introduction) framework.
> Hydra is an open-source Python framework that simplifies the development of research and other complex applications. The key feature is the ability to dynamically create a hierarchical configuration by composition and override it through config files and the command line. The name Hydra comes from its ability to run multiple similar jobs - much like a Hydra with multiple heads.
>
> Key features:​
> - Hierarchical configuration composable from multiple sources
> - Configuration can be specified or overridden from the command line
> - Dynamic command line tab completion

Notice that the `main()` method in [main.py](simharness2/main.py) is decorated with:
```python
@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
```
- Hydra needs to know where to find the config. This is done by specifying the directory containing the config (relative to `simharness`, our *application*) by passing `config_path`. In our case, the [conf](simharness2/conf) directory is used.
- The *top-level* configuration file is specified by passing the `config_name` parameter. In our case, [config.yaml](simharness2/conf/config.yaml) will be used as the default configuration file. Note that you should omit the `.yaml` extension.
    - To override the `config_name` specified in `hydra.main()`, set the `--config-name` (or `-cn`) flag . For example, to run `main.py` using the `dqn.yaml` configuration file, we can do:
    ```shell
    python main.py --config-name dqn
    ```
    - See [Hydra's command line flags](https://hydra.cc/docs/1.1/advanced/hydra-command-line-flags/) to learn more.


## Customizing an RLlib `Algorithm`
As mentioned above, configuration files for `simharness` are stored in the [conf](simharness2/conf) directory. To customize the configuration of an RLlib `Algorithm`, an `AlgorithmConfig` object will be built using the settings provided in the specified hierarchical configuration. 

The current structure of `conf` is as follows:
```shell
├── conf                      <- Hydra configs directory
│   ├── config.yaml              <- Main config
│   ├── environment              <- AlgorithmConfig.environment() settings
│   │   └── env_config              <- AlgorithmConfig.env_config
│   ├── evaluation               <- AlgorithmConfig.evaluation() settings
│   │   └── evaluation_config       <- AlgorithmConfig.evaluation_config
│   │       └── env_config             <- AlgorithmConfig.evaluation_config.env_config
│   ├── exploration              <- AlgorithmConfig.exploration() settings
│   │   └── exploration_config      <- AlgorithmConfig.exploration_config
│   ├── framework                <- AlgorithmConfig.framework() settings
│   ├── hydra                    <- Hydra-specific configs
│   ├── resources                <- AlgorithmConfig.resources() settings
│   ├── rollouts                 <- AlgorithmConfig.rollouts() settings
│   ├── simulation               <- Simulation configs
│   │   └── simfire                 <- simfire.sim.simulation.Simulation configs
│   └── training                 <- AlgorithmConfig.training() settings
```

See below for more information on each of level of the configuration hierarchy:

<details>
  <summary>Main config</summary>

  - `cli`
    - `mode`: The run mode to use. I recommend using `tune` to train an `Algorithm`, as it is (currently) the only
    way to track experiments with `Aim`. The `view` mode is intended to be used to do inference with a trained 
    `Algorithm` on a fixed evaluation simulation and save a `.gif` of the agent acting in the simulation.
        - Options: `train`, `tune`, `view`
        - Default: `???`
  - `algo`
    - `name`: The desired `Algorithm` class to use. See [Available Algorithms - Overview](https://docs.ray.io/en/latest/rllib/rllib-algorithms.html#available-algorithms-overview) for the algorithms currently available in RLlib.
        - Default: `DQN`
  - `runtime`: *Some* of the configuration that will be used to create a [ray.air.RunConfig](https://docs.ray.io/en/latest/ray-air/api/doc/ray.air.RunConfig.html#ray-air-runconfig) object.
    - `name`: Name of the trial or experiment. If not provided, will be deduced from the Trainable.
        - Default: `null`
    - `local_dir`: Local dir to save training results to.
        - Default: `${hydra:run.dir}`
  - `checkpoint`: The configuration that will be used to create a [ray.air.CheckpointConfig](https://docs.ray.io/en/latest/ray-air/api/doc/ray.air.CheckpointConfig.html#ray-air-checkpointconfig) object.
    - `checkpoint_frequency`: Number of iterations between checkpoints. Checkpointing is disabled when set to `0`.
        - Default: `20` (TODO: decide default value)
    - `num_to_keep`: The number of checkpoints to keep on disk for this run. If a checkpoint is persisted to disk after there are already this many checkpoints, then an existing checkpoint will be deleted. If this is `None`, checkpoints will not be deleted. Must be >= 1.
        - Default: `null`
  - `stop_conditions`: The stop conditions to consider. This will be used to set the `stop` argument when initializing the `ray.air.RunConfig` object. **Note**: The specified values **must** match keys contained in the `ray.rllib.utils.typing.ResultDict` (which represents the result dict returned by `Algorithm.train()`; see below for the default keys).
    - `training_iteration`: 
        - Default: `1000000` (1 million)
    - `timesteps_total`:
        - Default: `2000000000` (2 billion)
    - `episode_reward_mean`:
        - Default: `100` (currently, this value is arbitrary)
  - `debugging`: *Some* of the options that will be passed to configure `AlgorithmConfig.debugging()` settings.
    - `log_level`: Set the `ray.rllib.*` log level for the agent process and its workers. The `DEBUG` level will also periodically print out summaries of relevant internal dataflow (this is also printed out once at startup at the `INFO` level).
        - Options: `DEBUG`, `INFO`, `WARN`, `ERROR`
        - Default: `WARN`
    - `log_sys_usage`: Log system resource metrics to results. This requires `psutil` to be installed for sys stats, and `gputil` for GPU metrics.
        - Default: `True`
    - `seed`: This argument, in conjunction with worker_index, sets the random seed of each worker, so that **identically configured trials will have identical results. This makes experiments reproducible.**
        - Default: `2000`
    
        

</details>

<details>
  <summary>AlgorithmConfig.environment() settings</summary>
  
  ### TODO
  
</details>

<details>
  <summary>AlgorithmConfig.evaluation() settings</summary>
  
  ### TODO
  
</details>

<details>
  <summary>AlgorithmConfig.exploration() settings</summary>
  
  ### TODO
  
</details>

<details>
  <summary>AlgorithmConfig.framework() settings</summary>
  
  ### TODO
  
</details>

<details>
  <summary>AlgorithmConfig.resources() settings</summary>
  
  ### TODO
  
</details>

<details>
  <summary>AlgorithmConfig.rollouts() settings</summary>
  
  ### TODO
  
</details>

<details>
  <summary>Simulation configs</summary>
  
  ### TODO
  `simfire.sim.simulation.Simulation` configs


</details>

<details>
  <summary>Hydra-specific configs</summary>
  
  ### TODO
  
</details>

<details>
  <summary>Default keys in ray.rllib.utils.typing.ResultDict</summary>

````python
['episode_reward_max', 'episode_reward_min', 'episode_reward_mean', 'episode_len_mean', 'episodes_this_iter', 'num_faulty_episodes', 'num_healthy_workers', 'num_in_flight_async_reqs', 'num_remote_worker_restarts', 'num_agent_steps_sampled', 'num_agent_steps_trained', 'num_env_steps_sampled', 'num_env_steps_trained', 'num_env_steps_sampled_this_iter', 'num_env_steps_trained_this_iter', 'timesteps_total', 'num_steps_trained_this_iter', 'agent_timesteps_total', 'done', 'episodes_total', 'training_iteration', 'trial_id', 'experiment_id', 'date', 'timestamp', 'time_this_iter_s', 'time_total_s', 'pid', 'hostname', 'node_ip', 'time_since_restore', 'timesteps_since_restore', 'iterations_since_restore', 'warmup_time', 'info/num_env_steps_sampled', 'info/num_env_steps_trained', 'info/num_agent_steps_sampled', 'info/num_agent_steps_trained', 'sampler_results/episode_reward_max', 'sampler_results/episode_reward_min', 'sampler_results/episode_reward_mean', 'sampler_results/episode_len_mean', 'sampler_results/episodes_this_iter', 'sampler_results/num_faulty_episodes', 'hist_stats/episode_reward', 'hist_stats/episode_lengths', 'sampler_perf/mean_raw_obs_processing_ms', 'sampler_perf/mean_inference_ms', 'sampler_perf/mean_action_processing_ms', 'sampler_perf/mean_env_wait_ms', 'sampler_perf/mean_env_render_ms', 'connector_metrics/ObsPreprocessorConnector_ms', 'connector_metrics/StateBufferConnector_ms', 'connector_metrics/ViewRequirementAgentConnector_ms', 'timers/training_iteration_time_ms', 'counters/num_env_steps_sampled', 'counters/num_env_steps_trained', 'counters/num_agent_steps_sampled', 'counters/num_agent_steps_trained', 'config/num_gpus', 'config/num_cpus_per_worker', 'config/num_gpus_per_worker', 'config/_fake_gpus', 'config/num_trainer_workers', 'config/num_gpus_per_trainer_worker', 'config/num_cpus_per_trainer_worker', 'config/placement_strategy', 'config/eager_tracing', 'config/eager_max_retraces', 'config/env', 'config/observation_space', 'config/action_space', 'config/env_task_fn', 'config/render_env', 'config/clip_rewards', 'config/normalize_actions', 'config/clip_actions', 'config/disable_env_checking', 'config/is_atari', 'config/auto_wrap_old_gym_envs', 'config/num_envs_per_worker', 'config/sample_collector', 'config/sample_async', 'config/enable_connectors', 'config/rollout_fragment_length', 'config/batch_mode', 'config/remote_worker_envs', 'config/remote_env_batch_wait_ms', 'config/validate_workers_after_construction', 'config/ignore_worker_failures', 'config/recreate_failed_workers', 'config/restart_failed_sub_environments', 'config/num_consecutive_worker_failures_tolerance', 'config/preprocessor_pref', 'config/observation_filter', 'config/synchronize_filters', 'config/compress_observations', 'config/enable_tf1_exec_eagerly', 'config/sampler_perf_stats_ema_coef', 'config/worker_health_probe_timeout_s', 'config/worker_restore_timeout_s', 'config/gamma', 'config/lr', 'config/train_batch_size', 'config/max_requests_in_flight_per_sampler_worker', 'config/rl_trainer_class', 'config/_enable_rl_trainer_api', 'config/_rl_trainer_hps', 'config/explore', 'config/policy_states_are_swappable', 'config/actions_in_input_normalized', 'config/postprocess_inputs', 'config/shuffle_buffer_size', 'config/output', 'config/output_compress_columns', 'config/output_max_file_size', 'config/offline_sampling', 'config/evaluation_interval', 'config/evaluation_duration', 'config/evaluation_duration_unit', 'config/evaluation_sample_timeout_s', 'config/evaluation_parallel_to_training', 'config/ope_split_batch_by_episode', 'config/evaluation_num_workers', 'config/always_attach_evaluation_results', 'config/enable_async_evaluation', 'config/in_evaluation', 'config/sync_filters_on_rollout_workers_timeout_s', 'config/keep_per_episode_custom_metrics', 'config/metrics_episode_collection_timeout_s', 'config/metrics_num_episodes_for_smoothing', 'config/min_time_s_per_iteration', 'config/min_train_timesteps_per_iteration', 'config/min_sample_timesteps_per_iteration', 'config/export_native_model_files', 'config/checkpoint_trainable_policies_only', 'config/logger_creator', 'config/log_level', 'config/log_sys_usage', 'config/fake_sampler', 'config/seed', 'config/worker_cls', 'config/rl_module_class', 'config/_enable_rl_module_api', 'config/_tf_policy_handles_more_than_one_loss', 'config/_disable_preprocessor_api', 'config/_disable_action_flattening', 'config/_disable_execution_plan_api', 'config/simple_optimizer', 'config/replay_sequence_length', 'config/horizon', 'config/soft_horizon', 'config/no_done_at_end', 'config/target_network_update_freq', 'config/num_steps_sampled_before_learning_starts', 'config/store_buffer_in_checkpoints', 'config/lr_schedule', 'config/adam_epsilon', 'config/grad_clip', 'config/tau', 'config/num_atoms', 'config/v_min', 'config/v_max', 'config/noisy', 'config/sigma0', 'config/dueling', 'config/hiddens', 'config/double_q', 'config/n_step', 'config/before_learn_on_batch', 'config/training_intensity', 'config/td_error_loss_fn', 'config/categorical_distribution_temperature', 'config/__stdout_file__', 'config/__stderr_file__', 'config/input', 'config/callbacks', 'config/create_env_on_driver', 'config/custom_eval_function', 'config/framework', 'config/num_cpus_for_driver', 'config/num_workers', 'perf/cpu_util_percent', 'perf/ram_util_percent', 'perf/gpu_util_percent0', 'perf/vram_util_percent0', 'sampler_results/hist_stats/episode_reward', 'sampler_results/hist_stats/episode_lengths', 'sampler_results/sampler_perf/mean_raw_obs_processing_ms', 'sampler_results/sampler_perf/mean_inference_ms', 'sampler_results/sampler_perf/mean_action_processing_ms', 'sampler_results/sampler_perf/mean_env_wait_ms', 'sampler_results/sampler_perf/mean_env_render_ms', 'sampler_results/connector_metrics/ObsPreprocessorConnector_ms', 'sampler_results/connector_metrics/StateBufferConnector_ms', 'sampler_results/connector_metrics/ViewRequirementAgentConnector_ms', 'config/tf_session_args/intra_op_parallelism_threads', 'config/tf_session_args/inter_op_parallelism_threads', 'config/tf_session_args/log_device_placement', 'config/tf_session_args/allow_soft_placement', 'config/local_tf_session_args/intra_op_parallelism_threads', 'config/local_tf_session_args/inter_op_parallelism_threads', 'config/env_config/simulation', 'config/env_config/movements', 'config/env_config/interactions', 'config/env_config/attributes', 'config/env_config/normalized_attributes', 'config/env_config/agent_speed', 'config/env_config/deterministic', 'config/model/_disable_preprocessor_api', 'config/model/_disable_action_flattening', 'config/model/fcnet_hiddens', 'config/model/fcnet_activation', 'config/model/conv_filters', 'config/model/conv_activation', 'config/model/post_fcnet_hiddens', 'config/model/post_fcnet_activation', 'config/model/free_log_std', 'config/model/no_final_linear', 'config/model/vf_share_layers', 'config/model/use_lstm', 'config/model/max_seq_len', 'config/model/lstm_cell_size', 'config/model/lstm_use_prev_action', 'config/model/lstm_use_prev_reward', 'config/model/_time_major', 'config/model/use_attention', 'config/model/attention_num_transformer_units', 'config/model/attention_dim', 'config/model/attention_num_heads', 'config/model/attention_head_dim', 'config/model/attention_memory_inference', 'config/model/attention_memory_training', 'config/model/attention_position_wise_mlp_dim', 'config/model/attention_init_gru_gate_bias', 'config/model/attention_use_n_prev_actions', 'config/model/attention_use_n_prev_rewards', 'config/model/framestack', 'config/model/dim', 'config/model/grayscale', 'config/model/zero_mean', 'config/model/custom_model', 'config/model/custom_action_dist', 'config/model/custom_preprocessor', 'config/model/lstm_use_prev_action_reward', 'config/model/_use_default_native_models', 'config/exploration_config/type', 'config/exploration_config/initial_epsilon', 'config/exploration_config/final_epsilon', 'config/exploration_config/epsilon_timesteps', 'config/policies/default_policy', 'config/evaluation_config/explore', 'config/evaluation_config/env', 'config/logger_config/type', 'config/logger_config/logdir', 'config/replay_buffer_config/type', 'config/replay_buffer_config/prioritized_replay', 'config/replay_buffer_config/capacity', 'config/replay_buffer_config/prioritized_replay_alpha', 'config/replay_buffer_config/prioritized_replay_beta', 'config/replay_buffer_config/prioritized_replay_eps', 'config/replay_buffer_config/replay_sequence_length', 'config/replay_buffer_config/worker_side_prioritization', 'config/multiagent/policy_mapping_fn', 'config/multiagent/policies_to_train', 'config/multiagent/policy_map_capacity', 'config/multiagent/policy_map_cache', 'config/multiagent/count_steps_by', 'config/multiagent/observation_fn', 'config/tf_session_args/gpu_options/allow_growth', 'config/tf_session_args/device_count/CPU', 'config/evaluation_config/env_config/simulation', 'config/evaluation_config/env_config/movements', 'config/evaluation_config/env_config/interactions', 'config/evaluation_config/env_config/attributes', 'config/evaluation_config/env_config/normalized_attributes', 'config/evaluation_config/env_config/agent_speed', 'config/evaluation_config/env_config/deterministic', 'config/multiagent/policies/default_policy']
```

</details>
