from typing import List, Union, Optional, Tuple

from dataclasses import dataclass
from dataclasses import field

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING
from omegaconf import II  # , SI

from simharness2.config.utils import GetClassConf

# TODO: Use constants.py for the group and name of each config
from simharness2.constants import *


@dataclass
class SimfireConfig:
    # FIXME: Using magic number to retrieve config_dict data. Use constants.py instead.
    _target_: Optional[str] = None
    config_dict: dict = II("simulation.train")

    def __post_init__(self):
        if self._target_ is None:
            self._target_ = DEFAULT_SIMFIRE_CONFIG_CLASS


@dataclass
class FireSimulationConfig:
    # _target_: Optional[str] = None
    _target_: str = DEFAULT_SIMFIRE_SIMULATION_CLASS
    config: SimfireConfig = field(default_factory=SimfireConfig)

    # def __post_init__(self):
    #     print(f"post init is running with {self._target_}")
    #     if self._target_ is None:
    #         self._target_ = DEFAULT_SIMFIRE_SIMULATION_CLASS


@dataclass
class ActionSpaceConf(GetClassConf):
    path: str = MISSING


@dataclass
class DiscreteActionSpaceConf(ActionSpaceConf):
    path: str = "gymnasium.spaces.Discrete"


@dataclass
class MultiDiscreteActionSpaceConf(ActionSpaceConf):
    path: str = "gymnasium.spaces.MultiDiscrete"


@dataclass
class AgentAnalyticsConfig:
    _target_: str = MISSING
    _partial_: bool = True
    movement_types: List[str] = II("....movements")
    interaction_types: List[str] = II("....interactions")
    save_history: Optional[bool] = None


@dataclass
class SimAnalyticsConfig:
    _target_: str = MISSING
    _partial_: bool = True
    agent_analytics_partial: AgentAnalyticsConfig = field(
        default_factory=AgentAnalyticsConfig
    )
    save_history: Optional[bool] = None


# TODO: Need to simplify usage of HarnessAnalytics. Make a default config?
@dataclass
class HarnessAnalyticsConfig:
    _target_: str = MISSING
    _partial_: bool = True
    sim_analytics_partial: SimAnalyticsConfig = field(default_factory=SimAnalyticsConfig)


@dataclass
class RewardClassConfig:
    _target_: str = MISSING
    _partial_: bool = True
    # fixed_reward: Optional[float] = None
    # static_penalty: Optional[float] = None
    # invalid_movement_penalty: Optional[float] = None


@dataclass
class EnvContextConfig:
    """Arguments dict passed to the env creator as an `EnvContext` object.

    The `EnvContext` object is a dict, plus the following properties as keys:
      - num_rollout_workers
      - worker_index
      - vector_index
      - remote

    """

    # FIXME: Which specification of `sim` makes sense here?
    # sim: FireSimulationConfig = MISSING
    sim: FireSimulationConfig = field(default_factory=FireSimulationConfig)

    # TODO: Maybe we define HARNESS_DEFAULTS somewhere with default movements, etc.
    attributes: List[str] = MISSING
    normalized_attributes: List[str] = MISSING
    movements: List[str] = MISSING
    interactions: List[str] = MISSING
    action_space_cls: ActionSpaceConf = field(default_factory=ActionSpaceConf)
    in_evaluation: bool = False  # FIXME: Should default value be `MISSING`?

    benchmark_sim: Optional[FireSimulationConfig] = None
    harness_analytics_partial: HarnessAnalyticsConfig = field(
        default_factory=HarnessAnalyticsConfig
    )
    reward_cls_partial: RewardClassConfig = field(default_factory=RewardClassConfig)

    num_agents: int = 1
    agent_speed: int = MISSING
    agent_initialization_method: str = "automatic"
    # TODO: Make a custom type to annotate this?
    # FIXME: omegaconf raises ValidationError for Tuple[int, int]. Use List[int].
    initial_agent_positions: Optional[List[List[int]]] = None

    def __post_init__(self):
        # TODO: Maybe we validate normalized attributes here?
        pass


@dataclass
class EnvironmentConfig:
    """Config schema used to validate provided RL-environment settings in `rllib`.

    The "final" config is used as the input to `AlgorithmConfig.environment()`.
    For more information on specifying environments in `rllib`, see:
    https://docs.ray.io/en/latest/rllib/rllib-training.html#specifying-environments

    Note that not all available input arguments to `AlgorithmConfig.environment()` are
    specified here. For example, as of Ray 2.8.0, we omit the following arguments:
      - `observation_space` and `action_space`
      - `env_task_fn` (TODO: Enable usage of `TaskSettableEnv` in the future.)
      - `render_env`
      - `clip_actions` (deprecated)

    Arguments:
      env: The environment specifier to use. This can be a tune-registered env or an
        RLlib supported type. For RLlib supported types, it can be a gymnasium env, a
        PyBullet env, a ViZDoomGym env, or a fully qualified classpath to an Env class.
      env_config: The arguments dict passed to the env creator as an `EnvContext` object.
      clip_rewards: Whether to clip rewards during Policy's postprocessing. Options are:
        - None (default): Clip for Atari only (r = sign(r)).
        - True: r = sign(r): Fixed rewards -1.0, 1.0, or 0.0.
        - False: Never clip.
        - [float value]: Clip at -value and +value.

      normalize_actions: If True, RLlib will learn entirely inside a normalized action
        space (0.0 centered with small stddev; only affecting Box components). Actions
        will be unsquashed (and clipped, just in case) to the bounds of the env's action
        space before sending actions back to env.
      disable_env_checking: If True, disable the environment pre-checking module. For
        more details on the common pre-checks, see `ray/rllib/utils/pre_checks/env.py`.
      is_atari: Whether the env is an Atari env or not. If not specified, RLlib will try
        to auto-detect this. When using a SimHarness env, this should be False.
      auto_wrap_old_gym_envs: Whether to automatically wrap the given gym env class with
        the gym-provided compatibility wrapper (`gym.wrappers.EnvCompatibility`). If
        False, RLlib will produce a descriptive error on which steps to perform to
        upgrade to gymnasium (or to switch this flag to True).
      action_mask_key: If observation is a dictionary, expect the value by the key
        `action_mask_key` to contain a valid actions mask (`numpy.int8` array of zeros
        and ones).
    """

    env: str = MISSING
    env_config: EnvContextConfig = field(default_factory=EnvContextConfig)
    clip_rewards: Optional[Union[bool, float]] = None
    normalize_actions: Optional[bool] = True
    disable_env_checking: Optional[bool] = False
    is_atari: Optional[bool] = False
    auto_wrap_old_gym_envs: Optional[bool] = True
    action_mask_key: str = "action_mask"


cs = ConfigStore.instance()
cs.store(
    group="environment",
    name="base_environment",
    node=EnvironmentConfig,
)
cs.store(
    group="environment/env_config",
    name="base_env_config",
    node=EnvContextConfig,
)
# cs.store(
#     group="environment/env_config",
#     name="fire_simulation",
#     node=FireSimulationConfig,
# )
cs.store(
    group="environment/env_config/sim",
    name="fire_simulation",
    node=FireSimulationConfig,
)
cs.store(
    group="environment/env_config/action_space_cls",
    name="discrete",
    node=DiscreteActionSpaceConf,
)
cs.store(
    group="environment/env_config/action_space_cls",
    name="multi_discrete",
    node=MultiDiscreteActionSpaceConf,
)
