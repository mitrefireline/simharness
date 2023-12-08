# Developing

SimHarness provides a modular structure to allow users to customize the library's code
for their specific use cases. This includes configuration files, reward functions, and
environmental parameters.

## Configs

SimHarness utilizes [Hydra](https://github.com/facebookresearch/hydra) as a hierarchical
configuration management tool to allow users to configure the training parameters of
SimHarness.

> Hydra is an open-source Python framework that simplifies the development of research and
other complex applications. The key feature is the ability to dynamically create a
hierarchical configuration by composition and override it through config files and the
command line. The name Hydra comes from its ability to run multiple similar jobs - much
like a Hydra with multiple heads.
>
> Key features:​
>
> - Hierarchical configuration composable from multiple sources
> - Configuration can be specified or overridden from the command line
> - Dynamic command line tab completion

The config directories provided by SimHarness mirror the structure of the
[AlgorithmConfigs](https://github.com/ray-project/ray/blob/ac4229200b77d89ce5624501469de35b7733c976/rllib/algorithms/algorithm_config.py#L118)
used by RLlib for model training, such as `training`, `evaluation`, and `resources`.
The `environment`, `hydra`, `simulation`, and `tunables` config directories are not part
of the RLlib structure, and are instead specific to the SimHarness repository.

### Main Config
Notice that the `main()` method in [main.py](simharness2/main.py) is decorated with:

```python
@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
```

- Hydra needs to know where to find the config. This is done by specifying the directory
containing the config (relative to `simharness`, our *application*) by passing
`config_path`. In our case, the [conf](simharness2/conf) directory is used.
- The *top-level* configuration file is specified by passing the `config_name` parameter.
In our case, [config.yaml](simharness2/conf/config.yaml) will be used as the default
configuration file. Note that you should omit the `.yaml` extension.
- To override the `config_name` specified in `hydra.main()`, set the `--config-name` (or
`-cn`) flag . For example, to run `main.py` using the `dqn.yaml` configuration file, we
can do:

```shell
python main.py --config-name dqn
```

- See [Hydra's command line flags](https://hydra.cc/docs/1.1/advanced/hydra-command-line-flags/)
to learn more.

### Customizing an RLlib Algorithm
Configuration files for `simharness` are stored in the [conf](simharness2/conf) directory.
To customize the configuration of an RLlib `Algorithm`, an `AlgorithmConfig` object will
be built using the settings provided in the specified hierarchical configuration.

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
│   ├── training                 <- AlgorithmConfig.training() settings
│   └── tunables                 <- Hyperparameters to be tuned during tuning
```

Detailed analysis of each hyperparameter file and parameters is in progress.

### Hydra Config Command Line Modifications

Hydra allows for parameters within the config to be changed from the command line when
running the code or submitting a job. To do so, first identify which parameter(s) will be
modified. The key for each parameter follows the hierarchical structure of the config
folders. For example, a parameter such as the `cli.mode` is found within the main
`config.yaml` file in the `cli` section, with the parameter to set being `mode`.
Parameters from within a deeper directory, such as the number of GPUs in the `resources`
would be the key `resources.num_gpus`. To set the value of the parameter, simply add
`=X` where `X` is the value to set to.

## Reward Function

SimHarness utilizes a modular reward function to allow users to customize the priorities
of the trained agent(s). Different reward functions can be selected within the
`environment` config to modify the behavior of the agents to better match the intent of
the user. For example, some users may prioritize total land saved while not prioritizing
the fire distance to the wildland urban interface (WUI), while other users may prioritize
the opposite.

To customize the agent's reward function, users must add a new reward class to the
reward file `simharness2.rewardds.base_reward.py` that inherits from the `BaseReward`
class. This enforces the implemented class to include two functions: `get_reward()` and
`get_timestep_intermediate_reward()`. `get_reward()` provides the logic for calculating
the reward when the simulation is run, while `get_timestep_intermediate_reward()` provides
the logic for steps when the simulation is *not* run. The split allows users to provide
different rewards depending on if the action is taken when the fire spread is being
applied or not. More complex reward functions, such as `BenchmarkReward` pass in a
`HarnessAnalytics` object, which is an object that stores information about the current
training run and simulation object such as number of burned squares total or number of
squares burned only in the last timestep. This object can be used to develop more complex
reward functions based on previous steps and time.

### Analytics

The `ReactiveHarness` environment has a `harness_analytics` attribute. To leverage the
`analytics` subpackage, update the underlying config:
  - For training environments: `conf/environment/env_config/<config_name>.yaml`
  - For evaluation environments: `conf/evaluation/evaluation_config/env_config/<config_name>.yaml`


The corresponding section within the default training environment config file is below:
```yaml
# Defines the class that will be used to monitor and track `ReactiveHarness`.
harness_analytics_partial:
  _target_: simharness2.analytics.harness_analytics.ReactiveHarnessAnalytics
  _partial_: true
  # Defines the class that will be used to monitor and track `FireSimulation`.
  sim_analytics_partial:
    _target_: simharness2.analytics.simulation_analytics.FireSimulationAnalytics
    _partial_: true
    # Defines the class that will be used to monitor and track agent behavior.
    agent_analytics_partial:
      _target_: simharness2.analytics.agent_analytics.ReactiveAgentAnalytics
      _partial_: true
      movement_types: ${....movements}
      interaction_types: ${....interactions}
```

### Accessing Analytics Data

The `ReactiveHarnessAnalytics` object stores additional analytics classes for the the
`sim_analytics` and, optionally, the `benchmark_sim`. Both objects are of the
`FireSimulationAnalytics` class, and store information about the state of the `fire_map`
within the `FireSimulation`. `sim_analytics` holds information about the current
simulation run, while `benchmark_sim` is the same simulation but without any mitigations
placed, to act as a benchmark for the fire spread and analytics.

Within each `FireSimulationAnalytics` object, data about the current simulation is stored,
 along with agent specific information within the `agent_analytics` object, which stores
 agent-specific information like movements taken and interactions applied. The
 `SimulationData` class under the `data` attribute exposes the following attributes:
  - `burned`
  - `unburned`
  - `burning`
  - (optionally) `mitigated`
  - `agent_interactions`
  - `agent_movements`

To access the data, such as number of burning squares at the current timestep, the user
can use:

```python
sim_analytics.data.burned
```

### Per-Timestep Analytics
Both `sim_analytics` and `agent_analytics` can be configured to store the history of
each timestep within an episode. To enable, set `save_history: true` under the respective
`sim_analytics_partial` and `agent_analytics_partial` attributes. When enabled, the
underlying `data` attribute will add the data from each timestep to `_history`,
implemented as a `collections.deque` object, enabling the data to be aggregated and saved
on episode completion.

For example, the default evaluation configuration has the following:
```yaml

harness_analytics_partial:
  _target_: simharness2.analytics.harness_analytics.ReactiveHarnessAnalytics
  _partial_: true
  sim_analytics_partial:
    _target_: simharness2.analytics.simulation_analytics.FireSimulationAnalytics
    _partial_: true
    save_history: true # <--- HERE
    agent_analytics_partial:
      _target_: simharness2.analytics.agent_analytics.ReactiveAgentAnalytics
      _partial_: true
      movement_types: ${....movements}
      interaction_types: ${....interactions}
      save_history: true # <--- HERE
```

## Environments

The current `ReactiveHarness` inherits from the base `RLHarness` class, which handles
most of the environment setup. To create a new harness, users can inherit from the
`RLHarness` base class and implement the `reset()`, and `step()` methods according to
their specific use case. A `SimFire` `Simulation` object is required to be passed into the
harness, as the `Simulation` is the basis for the training environment.
