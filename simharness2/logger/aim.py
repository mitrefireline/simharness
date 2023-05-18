"""Module for using AIM with simharness2."""
import logging
from typing import TYPE_CHECKING, Dict, List, Optional, Union

import numpy as np
from hydra.utils import instantiate
from omegaconf import DictConfig
from ray.tune.logger.logger import LoggerCallback
from ray.tune.result import (
    EPISODES_TOTAL,
    TIME_TOTAL_S,
    TIMESTEPS_TOTAL,
    TRAINING_ITERATION,
)
from ray.tune.utils import flatten_dict
from ray.util.annotations import PublicAPI

if TYPE_CHECKING:
    from ray.tune.experiment.trial import Trial

try:
    from aim.sdk import Repo, Run
except ImportError:
    Repo, Run = None, None

logger = logging.getLogger(__name__)

VALID_SUMMARY_TYPES = [int, float, np.float32, np.float64, np.int32, np.int64]


@PublicAPI
class AimLoggerCallback(LoggerCallback):
    """Aim Logger: logs metrics in Aim format.

    Aim is an open-source, self-hosted ML experiment tracking tool.
    It's good at tracking lots (thousands) of training runs, and it allows you to
    compare them with a performant and well-designed UI.

    Source: https://github.com/aimhubio/aim

    Args:
        repo: Aim repository directory or a `Repo` object that the Run object will
            log results to. If not provided, a default repo will be set up in the
            experiment directory (one level above trial directories).
        experiment: Sets the `experiment` property of each Run object, which is the
            experiment name associated with it. Can be used later to query
            runs/sequences.
            If not provided, the default will be the Tune experiment name set
            by `RunConfig(name=...)`.
        metrics: List of metric names (out of the metrics reported by Tune) to
            track in Aim. If no metric are specified, log everything that
            is reported.
        aim_run_kwargs: Additional arguments that will be passed when creating the
            individual `Run` objects for each trial. For the full list of arguments,
            please see the Aim documentation:
            https://aimstack.readthedocs.io/en/latest/refs/sdk.html
    """

    VALID_HPARAMS = (str, bool, int, float, list, type(None))
    VALID_NP_HPARAMS = (np.bool8, np.float32, np.float64, np.int32, np.int64)

    def __init__(
        self,
        repo: Optional[Union[str, "Repo"]] = None,
        metrics: Optional[List[str]] = None,
        cfg: Optional[DictConfig] = None,
        **aim_run_kwargs,
    ):
        """
        See help(AimLoggerCallback) for more information about parameters.
        """
        assert Run is not None, (
            "aim must be installed!. You can install aim with"
            " the command: `pip install aim`."
        )
        self._repo_path = repo
        if not (bool(metrics) or metrics is None):
            raise ValueError(
                "`metrics` must either contain at least one metric name, or be None, "
                "in which case all reported metrics will be logged to the aim repo."
            )
        self._metrics = metrics
        # NOTE: I think a shallow copy is okay here; better to use a copy for safety?
        self._cfg = cfg.copy() if cfg else None
        self._aim_run_kwargs = aim_run_kwargs
        self._trial_to_run: Dict["Trial", Run] = {}

    def _create_run(self, trial: "Trial") -> Run:
        """Initializes an Aim Run object for a given trial.

        Args:
            trial: The Tune trial that aim will track as a Run.

        Returns:
            Run: The created aim run for a specific trial.
        """
        experiment_dir = trial.local_experiment_path
        run = Run(
            repo=self._repo_path or experiment_dir,
            **self._aim_run_kwargs,
        )
        # Attach a few useful trial properties
        run["trial_id"] = trial.trial_id
        run["trial_logdir"] = trial.logdir

        # Log the (hydra) config if it exists
        if self._cfg:
            self._log_hydra_config(run)
        trial_ip = trial.get_runner_ip()
        if trial_ip:
            run["trial_ip"] = trial_ip
        return run

    def log_trial_start(self, trial: "Trial"):
        """Execute on trial start.

        Args:
            trial: The Tune trial that aim will track as a Run.
        """
        if trial in self._trial_to_run:
            # Cleanup an existing run if the trial has been restarted
            self._trial_to_run[trial].close()

        trial.init_logdir()
        self._trial_to_run[trial] = self._create_run(trial)

        if trial.evaluated_params:
            self._log_trial_hparams(trial)

    def log_trial_result(self, iteration: int, trial: "Trial", result: Dict):
        """Log a result.

        Args:
            iteration: The iteration number
            trial: The Tune trial that aim will track as a Run.
            result: Dictionary containing key:value information to log
        """
        tmp_result = result.copy()

        step = result.get(TIMESTEPS_TOTAL, None) or result[TRAINING_ITERATION]
        episode = result.get(EPISODES_TOTAL, None)
        for k in ["config", "pid", "timestamp", TIME_TOTAL_S, TRAINING_ITERATION]:
            tmp_result.pop(k, None)  # not useful to log these

        # `context` and `epoch` are special keys that users can report,
        # which are treated as special aim metrics/configurations.
        context = tmp_result.pop("context", None)
        epoch = tmp_result.pop("epoch", None)

        trial_run = self._trial_to_run[trial]
        path = ["ray", "tune"]

        flat_result = flatten_dict(tmp_result, delimiter="/")
        valid_result = {}

        for attr, value in flat_result.items():
            if self._metrics and attr not in self._metrics:
                continue

            full_attr = "/".join(path + [attr])
            if isinstance(value, tuple(VALID_SUMMARY_TYPES)) and not (
                np.isnan(value) or np.isinf(value)
            ):
                valid_result[attr] = value
                trial_run.track(
                    value=value,
                    name=full_attr,
                    epoch=epoch,
                    step=episode or step,
                    context=context,
                )
            elif (isinstance(value, (list, tuple, set)) and len(value) > 0) or (
                isinstance(value, np.ndarray) and value.size > 0
            ):
                valid_result[attr] = value

    def log_trial_end(self, trial: "Trial", failed: bool = False):
        """Execute on trial end.

        Args:
            trial: The Tune trial that aim will track as a Run.
            failed: Flag indicating whether or not the trial failed
        """
        trial_run = self._trial_to_run.pop(trial)
        trial_run.close()

    def _log_trial_hparams(self, trial: "Trial"):
        """Log Hyperparameters.

        Args:
            trial: The Tune trial that aim will track as a Run.
        """
        params = flatten_dict(trial.evaluated_params, delimiter="/")
        flat_params = flatten_dict(params)

        scrubbed_params = {
            k: v for k, v in flat_params.items() if isinstance(v, self.VALID_HPARAMS)
        }

        np_params = {
            k: v.tolist()
            for k, v in flat_params.items()
            if isinstance(v, self.VALID_NP_HPARAMS)
        }

        scrubbed_params.update(np_params)
        removed = {
            k: v
            for k, v in flat_params.items()
            if not isinstance(v, self.VALID_HPARAMS + self.VALID_NP_HPARAMS)
        }
        if removed:
            logger.info(
                "Removed the following hyperparameter values when " "logging to aim: %s",
                str(removed),
            )

        run = self._trial_to_run[trial]
        run["hparams"] = scrubbed_params

    def _log_hydra_config(self, run: Run):
        """Log a subset of the hydra config to Aim as `Run Params`."""
        for cfg_k, cfg_v in self._cfg.items():
            if cfg_k == "simulation":
                self._log_simulation_config(run, cfg_v)
            elif cfg_k == "environment":
                self._log_environment_config(run, cfg_v)
            elif cfg_k == "evaluation":
                self._log_evaluation_config(run, cfg_v)
            else:
                # Simple case, just log the config key and its contents.
                run[cfg_k] = instantiate(cfg_v)
            continue
            # run[k] = v
        # run["cfg"] = self._cfg

    def _log_evaluation_config(self, run: Run, cfg: DictConfig):
        """Log the evaluation config to Aim as `Run Params`."""
        eval_cfg_settings = instantiate(cfg.evaluation_config)

        if "simulation" in eval_cfg_settings.env_config.keys():
            sim_obj = eval_cfg_settings.env_config.simulation
            # Intention: create a (dotpath) string representation of `sim_obj`.
            if not isinstance(sim_obj, str):
                sim_obj = ".".join(
                    [sim_obj.__class__.__module__, sim_obj.__class__.__name__]
                )
            eval_cfg_settings.env_config.simulation = sim_obj

        cfg.evaluation_config = eval_cfg_settings
        run["evaluation"] = cfg

    def _log_environment_config(self, run: Run, cfg: DictConfig):
        """Log the environment config to Aim as `Run Params`."""
        env_settings = instantiate(cfg)

        if "simulation" in env_settings.env_config.keys():
            sim_obj = env_settings.env_config.simulation
            # Intention: create a (dotpath) string representation of `sim_obj`.
            if not isinstance(sim_obj, str):
                sim_obj = ".".join(
                    [sim_obj.__class__.__module__, sim_obj.__class__.__name__]
                )
            env_settings.env_config.simulation = sim_obj

        run["environment"] = env_settings

    def _log_simulation_config(self, run: Run, cfg: DictConfig):
        """Log the simulation config to Aim as `Run Params`."""
        # NOTE: Both `train` and `eval` configs are logged, even if they are the same. In
        # TODO: In future, log `train` config and only log parameters within `eval`
        # config that differ from the `train` config (to reduce redundancy).
        run["simulation"] = instantiate(cfg)
        # sim_cfg_flat = flatten_dict(instantiate(cfg_v), delimiter=".")
        # Log all training simulation parameters, and only log the evaluation
        # simulation parameters that are different from the training ones.
        # train_cfg = OmegaConf.to_container(instantiate(cfg_v["train"]))
        # eval_cfg = OmegaConf.to_container(instantiate(cfg_v["eval"]))
        # train_cfg_flat = flatten_dict(train_cfg, delimiter=".")
        # eval_cfg_flat = flatten_dict(eval_cfg, delimiter=".")

        # for k, tr_v in train_cfg_flat.items():
        #     # Check if evaluation simulation has a different value for parameter.
        #     eval_v = eval_cfg_flat.get(k, None)
        #     if eval_v and tr_v != eval_v:
        #         params_dict["simulation"]["eval"].update({k: eval_v})
        #     # Always log the training simulation parameters.
        #     params_dict["simulation"]["train"].update({k: tr_v})

        # Remove the `eval` dict if it is empty.
        # if not params_dict["simulation"]["eval"]:
        #     params_dict["simulation"].pop("eval")

        # train_set = set(train_cfg_flat.items())
        # eval_set = set(eval_cfg_flat.items())
