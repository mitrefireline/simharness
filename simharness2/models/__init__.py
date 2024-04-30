from ray.rllib.models.catalog import ModelCatalog

from .custom_dqn_torch_model import CustomDQNTorchVisionNet
from .custom_multimodal_torch_model import CustomMultimodalTorchModel


# Register custom model.
ModelCatalog.register_custom_model(
    "metric_reporting_vision_network", CustomDQNTorchVisionNet
)

ModelCatalog.register_custom_model("multimodal_network", CustomMultimodalTorchModel)
