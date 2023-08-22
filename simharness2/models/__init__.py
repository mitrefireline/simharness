from ray.rllib.models.catalog import ModelCatalog

from .custom_dqn_torch_model import CustomDQNTorchVisionNet

# Register custom model.
ModelCatalog.register_custom_model(
    "metric_reporting_vision_network", CustomDQNTorchVisionNet
)
