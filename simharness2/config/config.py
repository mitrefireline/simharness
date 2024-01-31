import logging
from typing import Any, Dict, List, Union, Optional, TYPE_CHECKING

from dataclasses import dataclass
from dataclasses import field

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING
from omegaconf import II  # , SI

# TODO: Use constants.py for the group and name of each config

from dataclasses import field

# TODO: Explore using rllib types
# from ray.rllib.utils.typing import LearningRateOrSchedule
# from ray.rllib.utils.typing import PartialAlgorithmConfigDict
# from ray.rllib.models import MODEL_DEFAULTS

# from simfire.utils.config import Config as SimfireConfig

from simharness2.config.environment import EnvironmentConfig

# TODO: Test if using `NotProvided` as default value works.
# from ray.rllib.utils.from_config import NotProvided


@dataclass
class AimConfig:
    """TODO."""

    # configuration passed to the `AimLoggerCallback`
    repo: str = MISSING
    experiment: str = MISSING
    # TODO: Enable writing to aim when training from a checkpoint
    run_hash: Optional[str] = None
    system_tracking_interval: Optional[int] = None
    log_system_params: Optional[bool] = None
    capture_terminal_logs: Optional[bool] = True
    log_hydra_config: Optional[bool] = False


# FIXME! Implement logic
@dataclass
class TunablesConfig:
    """TODO."""

    training: Optional[dict] = None
    exploration: Optional[dict] = None


@dataclass
class TrainRunConfig:
    """TODO."""

    storage_path: str = MISSING
    name: Optional[str] = None
    verbose: Optional[int] = None
    log_to_file: Union[bool, str] = False


@dataclass
class TrainCheckpointConfig:
    """TODO."""

    # Number of checkpoints to keep on disk for this run.
    num_to_keep: Optional[int] = None
    # The attribute that will be used to "score" checkpoints.
    checkpoint_score_attribute: Optional[str] = None
    # Either "max" or "min".
    checkpoint_score_order: Optional[str] = None
    # Number of (training) iterations between checkpoints. Set to 0 to disable.
    checkpoint_frequency: Optional[int] = 20
    # If True, will save a checkpoint at the end of training.
    checkpoint_at_end: Optional[bool] = None


@dataclass
class RayInitConfig:
    """Arguments passed to `ray.init()`, which will connect to or start a Ray cluster."""

    address: Optional[str] = None
    configure_logging: bool = True
    # FIXME: is this default value "okay"?
    logging_level: int = logging.INFO
    logging_format: Optional[str] = None
    log_to_driver: bool = True
    runtime_env: Optional[Dict[str, Any]] = None


# FIXME: Find a "better" name for this class
@dataclass
class RLlibLoggerConfig:
    """TODO."""

    # FIXME: Decide how to handle this default value. Maybe force user to provide?
    type: Dict[str, str] = field(
        default_factory=lambda: {
            "_target_": "hydra.utils.get_class",
            "path": "ray.tune.logger.UnifiedLogger",
        }
    )
    logdir: str = II("hydra:run.dir")


@dataclass
class DebuggingConfig:
    """TODO."""

    # Define logger-specific configuration to be used inside Logger. Default value,
    # `None`, allows overwriting with nested dicts.
    logger_config: RLlibLoggerConfig = field(default_factory=RLlibLoggerConfig)
    # Set the ray.rllib.* log level for the algorithm process and its workers.
    # Should be one of "DEBUG", "INFO", "WARN", or "ERROR".
    log_level: str = "WARN"
    # Log system resource metrics to results. This requires `psutil` to be installed for
    # sys stats, and `gputil` for GPU metrics.
    log_sys_usage: bool = True
    # This argument, in conjunction with worker_index, sets the random seed of each
    # worker, so that identically configured trials will have identical results. This
    # makes experiments reproducible!!
    # TODO: Make note in documenetation that highlights the importance of this argument!
    seed: Optional[int] = None


@dataclass
class SimulationConfig:
    # train: SimfireConfig = field(default_factory=SimfireConfig)
    # eval: SimfireConfig = field(default_factory=SimfireConfig)
    train: dict = field(default_factory=dict)  # FIXME: No nested validation!!
    eval: dict = field(default_factory=dict)  # FIXME: No nested validation!!

    # FIXME: Do we want to keep these args? Decide then remove this comment.
    screen_size: int = 128
    screen_height: int = II(".screen_size")
    screen_width: int = II(".screen_size")
    fire_start_seed: int = 2


@dataclass
class TrainingConfig:
    gamma: Optional[float] = None
    # TODO: Enable user to specify a lr schedule.
    # NOTE: Using `LearningRateOrSchedule` type ann will raise:
    # omegaconf.errors.ConfigValueError: Unions of containers are not supported
    # lr: Optional[LearningRateOrSchedule] = None
    lr: Optional[float] = None
    grad_clip: Optional[float] = None
    grad_clip_by: str = "global_norm"
    train_batch_size: Optional[int] = None
    # FIXME: Create a `ModelConfig` class for verifying nested config.
    model: Optional[dict] = field(default_factory=dict)
    optimizer: Optional[dict] = field(default_factory=dict)
    max_requests_in_flight_per_sampler_worker: Optional[int] = None
    # learner_class: Optional[Type["Learner"]] = NotProvided,


@dataclass
class FrameworkConfig:
    """DL framework settings."""

    # Specify the framework to use. Supported options:
    #   - torch: PyTorch
    #   - tf2: TensorFlow 2.x (eager execution or traced if eager_tracing=True)
    #   - tf: TensorFlow (static-graph)
    framework: Optional[str] = None
    # TODO (later): Enable user to specify all the other framework options in rllib.
    # Enable tracing in eager mode.
    # eager_tracing: Optional[bool] = None
    # Maximum number of tf.function re-traces before a runtime error is raised.
    # eager_max_retraces: Optional[int] = None
    # tf_session_args: Optional[Dict[str, Any]] = None


@dataclass
class RolloutsConfig:
    num_rollout_workers: int = 0
    num_envs_per_worker: int = 1
    create_env_on_local_worker: bool = False
    enable_connectors: bool = True

    rollout_fragment_length: Optional[Union[int, str]] = None
    batch_mode: Optional[str] = None

    remote_worker_envs: bool = False
    remote_env_batch_wait_ms: int = 0

    validate_workers_after_construction: bool = True
    preprocessor_pref: Optional[str] = None
    # Either "NoFilter" or "MeanStdFilter"
    observation_filter: Optional[str] = None
    compress_observations: bool = False

    # FIXME: Ommitting these for now, but we should enable user to specify these later.
    # use_worker_filter_stats: Optional[bool] = None
    # update_worker_filter_stats: Optional[bool] = None
    # sampler_perf_stats_ema_coef: Optional[float] = None


@dataclass
class EvaluationConfig:
    evaluation_interval: Optional[int] = None
    # TODO: Should we "force" the user to use "episodes" as the unit?
    evaluation_duration: Optional[int] = None
    evaluation_duration_unit: Optional[str] = None
    evaluation_sample_timeout_s: Optional[float] = None
    evaluation_parallel_to_training: Optional[bool] = None
    evaluation_num_workers: Optional[int] = None
    always_attach_evaluation_results: Optional[bool] = None
    evaluation_config: Optional[dict] = None
    enable_async_evaluation: Optional[bool] = None
    # TODO: Enable below once simharness supports reading offline experiences.
    # off_policy_estimation_methods: Optional[Dict] = None
    # ope_split_batch_by_episode: Optional[bool] = None


@dataclass
class ExplorationConfig:
    explore: Optional[bool] = None
    # TODO: Can we make a schema to validate the exploration config dict?
    exploration_config: Optional[dict] = None


@dataclass
class ResourcesConfig:
    num_gpus: Union[float, int] = 0
    _fake_gpus: bool = False
    num_cpus_per_worker: Union[float, int] = 1
    num_gpus_per_worker: Union[float, int] = 0
    num_cpus_for_local_worker: int = 1
    num_learner_workers: int = 0
    num_cpus_per_learner_worker: Union[float, int] = 1
    num_gpus_per_learner_worker: Union[float, int] = 0
    local_gpu_idx: int = 0
    custom_resources_per_worker: Optional[dict] = None
    placement_strategy: str = "PACK"


@dataclass
class SimHarnessConfig:
    """TODO."""

    # Specify the run mode. Supported options: train, tune
    mode: str = MISSING
    # Specify the root directory used to save data for the experiment.
    data_dir: str = MISSING
    # Specify the trainable function or class used to train the policy.
    trainable_class: str = MISSING
    # The path (str) to the checkpoint directory to use.
    checkpoint_path: Optional[str] = None

    # TODO: The default must provide the correct keys that are accessed in main.py.
    stop_conditions: Dict[str, Any] = field(default_factory=dict)
    ray_init: RayInitConfig = field(default_factory=RayInitConfig)
    debugging: DebuggingConfig = field(default_factory=DebuggingConfig)

    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    framework: FrameworkConfig = field(default_factory=FrameworkConfig)
    rollouts: RolloutsConfig = field(default_factory=RolloutsConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    exploration: ExplorationConfig = field(default_factory=ExplorationConfig)
    resources: ResourcesConfig = field(default_factory=ResourcesConfig)

    # NOTE: Currently, below configs are used when `mode == tune` (for the most part).
    # TODO: How can we make this optional for user to provide?
    aim: AimConfig = field(default_factory=AimConfig)
    # FIXME: TunablesConfig is incomplete
    tunables: TunablesConfig = field(default_factory=TunablesConfig)
    run: TrainRunConfig = field(default_factory=TrainRunConfig)
    checkpoint: TrainCheckpointConfig = field(default_factory=TrainCheckpointConfig)


def register_configs() -> None:
    cs = ConfigStore.instance()
    cs.store(name="simharness_base_config", node=SimHarnessConfig)
    # cs.store(name="aim", node=AimConfig)
    cs.store(
        group="environment",
        name="base_environment",
        node=EnvironmentConfig,
    )
    # cs.store(
    #     group="database_lib/db",
    #     name="postgresql",
    #     node=PostGreSQLConfig,
    # )
