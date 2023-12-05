# Running the Code

SimHarness provides an easy-to-use API for the training and tuning of DRL models. On the
backend, SimHarness uses [RLlib](https://docs.ray.io/en/latest/rllib/index.html) for
algorithm training which uses [Ray](https://www.ray.io) as a compute framework.

# Ray Cluster Creation

Ray allows users to specify the resources currently available for use either locally or
through a remote connection using Ray *clusters*. If using a remote connection, users can
send jobs to the remote cluster for completion. A *cluster* is a set of worker nodes
connected to a common head node. Each worker node can help run processes in a distributed
manner to speed up training and maximize resource efficiency.

**NOTE:** The number of GPUs and CPUs specified in cluster creation will affect the number
of GPUs and CPUs the user can specify in the `resources` config. RLlib will not run
properly if the user requests a higher number of GPUs or CPUs for training than is
specified within the cluster.

## Developing Locally

Users have the ability to develop on their local machine with the resources available to
the current machine. To do so, users will need to create a local Ray cluster for the
training scripts to attach to.

### Creating a Ray Cluster

Run the below code to create a new Ray cluster with the specified resources on the current
machine, where `X` is the number of GPUs and `Y` is the number of CPUs that Ray can use
when training the algorithm.

```bash
ray start --num-gpus X --num-cpus Y --head
```

This cluster can be accessed either locally (from the same machine) or remotely (from
a different machine). To access remotely, the IP address of the host machine and port
number exposing the cluster need to be known.

### Running on a Local Ray Cluster

Running SimHarness on a local cluster is simple and mimics how the code is run normally,
where `X` is either `train` or `tune`.

```bash
python3 main.py cli.mode=X
```

### Running on a Remote Ray Cluster

Running SimHarness on a remote cluster requires the IP address of the host machine and
exposed Ray cluster port to be known. As the code is not being run on the local machine,
the user needs to submit a Ray *job* to the remote cluster to be run. See the Ray docs
for running on a remote cluster
[here](https://docs.ray.io/en/latest/cluster/running-applications/job-submission/quickstart.html#using-a-remote-cluster).

```bash
ray job submit --address X --working-dir ./ -- python main.py cli.mode=Y
```

Where `X` is the HTTP address of the ray cluster (IP and port number) and Y is either
`train` or `tune`.

### Train vs. Tune

SimHarness supports two types of algorithm training - `train` and `tune`. Running `train`
trains the algorithm using the currently supplied hyperparameters within the config
*as is*. This is useful when the user wants to runa  quick test of the code or knows
exactly which hyperparameters they would like the use for training (ie hyperparameter
tuning is already completed). Running `tune` will run hyperparameter tuning over a set of
predefined hyperparameter bounds/choices. Users can specify how many different trials they
would like to run and Ray will train the given number of algorithms with different
combinations of hyperparameters. Each algorithm will be saved by Ray and the overall
performance can be viewed either in the command line output through Ray or in the
[aim](https://aimstack.io) dashboard associated with the given trial. Learn more about
Ray's *tune* capabilities [here](https://docs.ray.io/en/latest/tune/key-concepts.html).

### Tuning Hyperparameters

To select hyperparameters for tuning, SimHarness uses a separate
config - `conf/tunables/default.yaml`. Within the file, users can specify different
training parameters to overwrite with tunable values. The top level keys in the yaml file
correspond to the directories within the larger config, ie `training`. The next level keys
represent the hyperparameters within the file, ie `lr` or `gamma`. The values associated
with the next level keys are information relevant to the tuning of the hyperparameter.
Users can select different *types* of tuning structures with the `type` key. This can be
one of: `uniform`, `loguniform`, or `choice`. If `logunifom` or `uniform` is chosen, then
the `values` key represents the bounds of the value (`[high, low]`). If `choice` is
chosen, then the `values` key represents all possible choices of the value. During
runtime, the tune algorithm will select a value for each hyperparameter and run a trial
with the specified values. At the end, a report will showcase the overall performance of
each trial and give insight into the optimal hyperparameter combination for the given
task. Running with `tune` is currently the only option that can provide detailed analysis
through the `aim` dashboard, so it is the recommended route for training an algorithm -
even if there is only a single trail and no tunable parameters.

**NOTE:** Currently, the tunables file can only support hyperparameters within the
`training` file. We expect the addition of additional files soon.

### Hydra Config Command Line Modifications

Hydra allows for parameters within the config to be changed from the command line when
running the code or submitting a job. To do so, first identify which parameter(s) will be
modified. The key for each parameter follows the hierarchical structure of the config
folders. For example, a parameter such as the `cli.mode` is found within the main
`config.yaml` file in the `cli` section, with the parameter to set being `mode`.
Parameters from within a deeper directory, such as the number of GPUs in the `resources`
would be the key `resources.num_gpus`. To set the value of the parameter, simply add
`=X` where `X` is the value to set to.
