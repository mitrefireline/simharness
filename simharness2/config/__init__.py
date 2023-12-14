from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

from simharness2.config.config import SimHarnessConfig

# Register custom resolvers that are used within the config files
OmegaConf.register_new_resolver("operational_screen_size", lambda x: int(x * 39))
OmegaConf.register_new_resolver("calculate_half", lambda x: int(x / 2))
OmegaConf.register_new_resolver("square", lambda x: x**2)


def register_configs() -> None:
    cs = ConfigStore.instance()
    cs.store(name="simharness_base_config", node=SimHarnessConfig)
