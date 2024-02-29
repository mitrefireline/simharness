# Environment Observations

## Example Usage of `supported_attributes()`

Quickly run the following python code to see the supported attributes for each environment:
````python
import inspect
import simharness2.environments as envs

for name, obj in inspect.getmembers(envs):
    if inspect.isclass(obj):
        try:
            print(obj.__name__)
            print(obj.supported_attributes())
        except NotImplementedError as e:
            print(e)
````

and the output will look similar to:

````shell
ComplexObsReactiveHarness
['fire_map', 'agent_pos']
DamageAwareReactiveHarness
['fire_map', 'fire_map_with_agents', 'w_0', 'sigma', 'delta', 'M_x', 'elevation', 'wind_speed', 'wind_direction', 'bench_fire_map', 'bench_fire_map_final']
FireHarness
['fire_map', 'fire_map_with_agents', 'w_0', 'sigma', 'delta', 'M_x', 'elevation', 'wind_speed', 'wind_direction']
Harness

MultiAgentComplexObsReactiveHarness
['fire_map', 'agent_pos']
MultiAgentFireHarness
['fire_map', 'fire_map_with_agents', 'w_0', 'sigma', 'delta', 'M_x', 'elevation', 'wind_speed', 'wind_direction']
ReactiveHarness
['fire_map', 'fire_map_with_agents', 'w_0', 'sigma', 'delta', 'M_x', 'elevation', 'wind_speed', 'wind_direction']
````
