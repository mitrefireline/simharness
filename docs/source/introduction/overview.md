# Repository Overview

SimHarness is a Python repository designed to support the training of
[RLlib](https://docs.ray.io/en/latest/rllib/index.html) RL algorithms within simulated
disaster environments defined by
[SimFire](https://github.com/mitrefireline/simfire/tree/main). SimHarness takes as input
an instance of the
[SimFire Simulation class](https://github.com/mitrefireline/simfire/blob/39abc5a34b103a306c776a3c2972c10a87d0e652/simfire/sim/simulation.py#L37),
such as SimFire's
[FireSimulation](https://github.com/mitrefireline/simfire/blob/39abc5a34b103a306c776a3c2972c10a87d0e652/simfire/sim/simulation.py#L173),
as the training environment.
The [Simulation](https://github.com/mitrefireline/simfire/blob/39abc5a34b103a306c776a3c2972c10a87d0e652/simfire/sim/simulation.py#L37)
object provides an API that allows SimHarness to move agents around the simulated
environment and interact with it by placing mitigations. The
[FireSimulation](https://github.com/mitrefireline/simfire/blob/39abc5a34b103a306c776a3c2972c10a87d0e652/simfire/sim/simulation.py#L173)
agents represent firefighters moving through an environment as a wildfire spreads,
placing mitigations such as firelines to limit the spread of the fire within the area.

SimHarness utilizes [Hydra](https://github.com/facebookresearch/hydra) as a hierarchical
configuration management tool to allow users to configure the training parameters of
SimHarness. The configuration files provided by SimHarness mirror the structure of the
[AlgorithmConfigs](https://github.com/ray-project/ray/blob/ac4229200b77d89ce5624501469de35b7733c976/rllib/algorithms/algorithm_config.py#L118)
used by RLlib for model training, such as `training`, `evaluation`, and `resources`.
Hyperparameter tuning can be applied to the RLlib algorithm hyperparameters to help
identify the optimal settings for training the agent(s).

Users can also configure the parameters used for initializing the `Simulation` and the agents
within the environment. For example, users can configure the `agent_speed`, which
controls the number of actions an agent can take before the simulation is run,
`interactions`, which are the mitigation techniques an agent can apply to the landscape,
and `attributes`, which determine which attributes are passed as an input dimension to the
RL model during training and inference.

Another configurable aspect of the SimHarness environment is the reward function. Users
can create a custom reward function for training that emphasizes user-specific goals. This
allows for tuning of the agent policy to better suit the user's goals. For example, some
users may want policies that prioritize ending the fire as quickly as possible, while
others may focus more on limiting the fire spread to specific areas.

<p align="center">
  <img src="../../images/workflow.png" />
  <b>Figure 1.</b> Conceptual workflow for training an RL model using SimHarness within the SimFire environment.
</p>

The SimHarness training loop functions similarly to a traditional RL training loop, except
it expects the passed-in environment to be a child class of `Simulation` as opposed to a
[gymnasium](https://gymnasium.farama.org) environment. `Simulation`` is currently a class
within the SimFire package, but is expected to be moved to a separate,
non-disaster-specific package in the future. The simulated environment outputs training
signals such as observations and rewards to the SimHarness agent(s) which use the
observations to predict optimal actions. The actions produced by the model provide both
`movement` and `interaction` information. `Movements` are how the agent is traversing
across the environment, such as `[nothing, up, down, left, right]`. `Interactions` are how
the agent is changing the environment itself. In the case of SimFire, this can be
`[nothing, fireline, wetline, scratchline]`. These actions are relayed back to the
simulated environment, which then affects the overall disaster scenario simulated by the
environment.
