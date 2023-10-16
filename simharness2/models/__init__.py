from ray.rllib.models.catalog import ModelCatalog

from .custom_cc_torch_model import YetAnotherTorchCentralizedCriticModel
from .custom_dqn_torch_model import CustomDQNTorchVisionNet

# Register custom model.
ModelCatalog.register_custom_model(
    "metric_reporting_vision_network", CustomDQNTorchVisionNet
)

ModelCatalog.register_custom_model("cc_model", YetAnotherTorchCentralizedCriticModel)
