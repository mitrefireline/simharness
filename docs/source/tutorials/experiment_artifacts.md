# What will be saved during an experiment?

## Table of Contents
- [Overview](#overview)
- [Quick Reference Guide](#quick-reference-guide)
- [Detailed Artifact Descriptions](#detailed-artifact-descriptions)
    - [Hydra Artifacts](#hydra-artifacts)
### Overview
When running experiments with SimHarness, several config keys dictate where training artifacts will be stored:
- `root_storage_path`: This is the **base directory** for all training artifacts. Default values for `aim.repo` and `hydra.run.dir` are set **relative** to this path, ensuring a consistent and organized structure for storing and accessing training artifacts.
    - **Note**: Make sure this path has **enough disk space**. Training artifacts can take up significant storage. Choose a path on a drive or partition with ample free space. Consider using an external storage solution or cloud storage for large-scale experiments. For more information on supported scenarios, see [Storage Options in Tune](https://docs.ray.io/en/latest/tune/tutorials/tune-storage.html#storage-options-in-tune).
- `experiment_subdir`: This is an intermediary subdirectory under `root_storage_path` that **helps categorize different types of experiments**. It doesn't directly store the experiments, but serves as a parent directory for specific experiment directories. By default, it is set to `experiments`, but users can change it to any other value, such as `debug_experiments`, `test_runs`, etc. This allows for more flexibility in organizing and accessing different types of experiments.
- `hydra.run.dir`: In single-run mode (i.e., when the `--multirun` command-line flag is omitted), this specifies the **output directory** for the respective run. This location stores log files and saves YAML configs (see `Hydra Artifacts` section below for more details). By default, it is set to `${root_storage_path}/${experiment_subdir}/${now:%Y-%m-%d_%H:%M:%S}`.
    - [Configuration for run](https://hydra.cc/docs/configure_hydra/workdir/#configuration-for-run) provides examples of custom configurations for the output directory.
    - **Note**: Hydra "multi-run mode" (i.e., when the `--multirun` command-line flag is given) has not been tested with SimHarness.

When `mode=tune` (use [Tuner.fit](https://docs.ray.io/en/latest/tune/api/execution.html#tune-run-ref) to execute and manage hyperparameter tuning and generate your trials), additional artifacts are generated for the experiment. The `Tuner.fit()` function provides features such as [logging](https://docs.ray.io/en/latest/tune/tutorials/tune-output.html#tune-logging), [checkpointing](https://docs.ray.io/en/latest/tune/tutorials/tune-trial-checkpoints.html#tune-trial-checkpoint), and [early stopping](https://docs.ray.io/en/latest/tune/tutorials/tune-stopping.html#tune-stopping-ref).
- For a complete outline of the types of data stored by Tune, see [this](https://docs.ray.io/en/latest/tune/tutorials/tune-trial-checkpoints.html#appendix-types-of-data-stored-by-tune) Appendix in the Ray docs.
With the above notes on Tune in mind, we have:
- `run.storage_path`: Path to store results for tuning runs. Can be a local directory or a cloud storage destination. If not provided, this defaults to the local `~/ray_results` directory.
    - **Note**: We recommend using `${hydra:run.dir}` to specify where to store results for tuning runs. This ensures Hydra artifacts and Tune artifacts are stored in a central location.
- `aim.repo`: Aim repository directory or a `Repo` object where the Run object will log results. If not provided, a default repo will be set up in the experiment directory (one level above trial directories).
    - **Note**: When `mode=tune`, SimHarness automatically sets the callbacks specified in the runtime configuration for tuning runs to `AimLoggerCallback`. This ensures that tune metrics are logged in the Aim format. We are currently focusing on other features, but making the callbacks more configurable is on our roadmap for future updates.

### Quick Reference Guide
In summary, these are the key configuration keys for controlling the storage of training artifacts:
* `root_storage_path`: The base directory for all training artifacts.
* `experiment_subdir`: An intermediary subdirectory under `root_storage_path` for categorizing experiments.
* `hydra.run.dir`: The output directory for a single run, storing Hydra log files and YAML configs.
* `run.storage_path`: The storage path for tuning runs.
* `aim.repo`: The Aim repository directory or a `Repo` object where the `Run` object logs results.
Remember, when `mode=tune`, SimHarness sets the callbacks for tuning runs to `AimLoggerCallback` to log tune metrics in the Aim format. Choosing the right paths for these keys is crucial for efficient storage and access of training artifacts. Consider the size of the artifacts and the scale of your experiments when setting these paths.

### Detailed Artifact Descriptions
In this section, we delve into more specific details about the different types of artifacts saved during an experiment.
#### Hydra Artifacts
The `hydra.run.dir` is used to store the Hydra output directory. Specifically, the `config.yaml`, `hydra.yaml`, and `overrides.yaml` can be found at the path resolved by `${hydra:run.dir}/${hydra:output_subdir}`.

The Hydra output directory (`.hydra` by default) and the application log file contain:
- `config.yaml`: A dump of the user-specified configuration.
- `hydra.yaml`: A dump of the Hydra configuration.
- `overrides.yaml`: The command line overrides used.
In the main output directory, you'll find:
- `my_app.log`: A log file created for this run.
	- **Note**: This log file is created when `hydra/job_logging=default`. If a different configuration is used for `hydra/job_logging`, then the file may not be created. To ensure that the file will be created, the logging configuration under `hydra.job_logging.handlers` must provide a `file` section, like the example below:
```yaml
handlers:
  ...
  file:
    class: logging.FileHandler
    formatter: simple
    # absolute file path
    filename: ${hydra:runtime.output_dir}/${hydra:job.name}.log
```

* **IMPORTANT**: When running experiments with SimHarness, it's typically best practice to disable creation of the log file, as we can access the console logs that are captured by Ray (and potentially Aim).
	* The Hydra log file typically contains very few lines, as most of the logic performed in the experiment is done outside of the main process. Nonetheless, create a log file for the experiment run if it provides important information or is useful for debugging.

The default name for the Hydra output directory is `.hydra`. To change this, override `hydra.output_subdir` with the new name, ie. `hydra.output_subdir=hydra_output` (see [Changing or disabling Hydra's output subdir](https://hydra.cc/docs/tutorials/basic/running_your_app/working_directory/#changing-or-disabling-hydras-output-subdir) for more details).
* **Note:** You can further configure the name of the output directory using the [customizing the working directory](https://hydra.cc/docs/configure_hydra/workdir/) pattern.
