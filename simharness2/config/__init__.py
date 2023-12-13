from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

from simharness2.config.config import SimHarnessConfig
from simharness2.config.environment import (
    EnvironmentConfig,
    EnvContextConfig,
    FireSimulationConfig,
    DiscreteActionSpaceConf,
    MultiDiscreteActionSpaceConf,
)

# Register custom resolvers that are used within the config files
OmegaConf.register_new_resolver("operational_screen_size", lambda x: int(x * 39))
OmegaConf.register_new_resolver("calculate_half", lambda x: int(x / 2))
OmegaConf.register_new_resolver("square", lambda x: x**2)


def register_configs() -> None:
    cs = ConfigStore.instance()
    cs.store(name="simharness_base_config", node=SimHarnessConfig)
    # cs.store(name="aim", node=AimConfig)
    # cs.store(
    #     group="environment",
    #     name="base_environment",
    #     node=EnvironmentConfig,
    # )
    # cs.store(
    #     group="environment/env_config",
    #     name="base_env_config",
    #     node=EnvContextConfig,
    # )
    # cs.store(
    #     group="environment/env_config",
    #     name="fire_simulation",
    #     node=FireSimulationConfig,
    # )
    # cs.store(
    #     group="environment/env_config/sim",
    #     name="fire_simulation",
    #     node=FireSimulationConfig,
    # )
    # cs.store(
    #     group="environment/env_config/action_space_cls",
    #     name="discrete",
    #     node=DiscreteActionSpaceConf,
    # )
    # cs.store(
    #     group="environment/env_config/action_space_cls",
    #     name="multi_discrete",
    #     node=MultiDiscreteActionSpaceConf,
    # )
