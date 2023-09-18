# Harness Analytics

## Overview
- The `ReactiveHarness` environment has a `harness_analytics` attribute. 
- To leverage the `analytics` subpackage, update the underlying config:
  - For training environments: `conf/environment/env_config/<config_name>.yaml`
  - For evaluation environments: `conf/evaluation/evaluation_config/env_config/<config_name>.yaml`

## Default Training Configuration
Here is the respective section within the default training environment config file:
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
### Configuration Details
#### `harness_analytics_partial`
- The `harness_analytics_partial` defines the `RLHarnessAnalytics` subclass that will be
  used to store and moniotor the behavior of the respective `RLHarness` subclass, such as
    the `ReactiveHarness` environment.
  - The subclass specified (see `_target_`) will be instantiated within the `RLHarness.__init__()` (FIXME:
    currently within the `ReactiveHarness.__init__()`) method
    and is stored as the `harness_analytics` attribute (FIXME: rename to `analytics`?),
    found via `ReactiveHarness.harness_analytics`.
  - Use `_target_` to specify the dotpath to the subclass to be used. By default, the
    `ReactiveHarnessAnalytics` subclass is used.
  - The usage of `_partial_: true` is required, otherwise an error will be raised.

##### `sim_analytics_partial`
- The `harness_analytics` instance requires a `SimulationAnalytics` subclass to be
  specified under the `sim_analytics_partial` attribute. This subclass will be used to
  monitor and track the behavior of the `Simulation` subclass used by the `RLHarness`, 
  such as the `FireSimulation` instance used by the `ReactiveHarness`.
  - The subclass specified will be instantiated within the 
    `RLHarnessAnalytics.__init__()` method and is stored as the `sim_analytics` 
    attribute, found via `ReactiveHarness.harness_analytics.sim_analytics` (FIXME).
  - Use `_target_` to specify the dotpath to the subclass to be used. By default, the
    `FireSimulationAnalytics` subclass is used.
  - The usage of `_partial_: true` is required, otherwise an error will be raised.

###### `agent_analytics_partial`
- The `sim_analytics` instance can be configured to use an `AgentAnalytics` subclass to
  track the behavior of each agent interacting with the `Simulation` subclass.
  as the `ReactiveAgent` instance used by the `FireSimulation`.
  - The subclass specified will be instantiated within the 
    `SimulationAnalytics.__init__()` method and is stored as the `agent_analytics` 
    attribute, found via `ReactiveHarness.harness_analytics.sim_analytics.agent_analytics`
    (FIXME).
  - Use `_target_` to specify the dotpath to the subclass to be used. By default, the
    `ReactiveAgentAnalytics` subclass is used.
  - The usage of `_partial_: true` is required, otherwise an error will be raised.
  - The `agent_analytics` instance requires the `movement_types` and `interaction_types`
    attributes to be specified.


## Accessing Analytics Data
### `harness_analytics`
- The `harness_analytics` instance exposes the following attributes:
  - `sim_analytics`: An object using the `FireSimulationAnalytics` API.
  - (optionally) `benchmark_sim_analytics`: An object using the `FireSimulationAnalytics` API.
  - `best_episode_performance`: An object storing the best performance across all
    episodes in a trial (wrt the reactive fire scenario). TODO: Finish implementation.

### `sim_analytics`
- The `sim_analytics` object stores an instance of the `SimulationData` class under the
  `data` attribute. This class exposes the following attributes:
  - `burned`
  - `unburned`
  - `burning`
  - (optionally) `mitigated`
  - `agent_interactions`
  - `agent_movements`

For example, assuming that `sim_analytics.update()` has been called, we can get the total
number of burning squares at the current timestep via:
```python
sim_analytics.data.burned
```


Interplay with `rewards`


## Saving Episode History
- Both `sim_analytics` and `agent_analytics` can be configured to store the history of 
each timestep within an episode. To enable, set `save_history: true` under the respective
`sim_analytics_partial` and `agent_analytics_partial` attributes.
- When enabled, the underlying `data` attribute will add the data from each timestep to
`_history`, implemented as a `collections.deque` object, enabling the data to be
  aggregated and saved on episode completion.

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

This will