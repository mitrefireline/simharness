# Notes for `analytics` subpackage

## Usage
- The `ReactiveHarness` environment has a `harness_analytics` attribute. Maybe
we should rename this to simply `analytics`?
- To leverage the `analytics` subpackage, update the underlying config:
    - For training environments: `conf/environment/env_config/<config_name>.yaml`
    - For evaluation environments: `conf/evaluation/evaluation_config/env_config/<config_name>.yaml`

### Example
For example, the respective section within the default training environment config file:
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
- Notice the usage of `_partial_: true`. This is required, otherwise an error will be raised.
