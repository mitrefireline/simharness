"""Custom model to report DQN metrics."""
from typing import Dict

from ray.rllib.algorithms.dqn.dqn_torch_policy import QLoss
from ray.rllib.models.torch.visionnet import VisionNetwork
from ray.rllib.utils.framework import TensorType


class CustomDQNTorchVisionNet(VisionNetwork):
    """Vision network that reports metrics (ie. q_loss) during forward pass."""

    def metrics(self) -> Dict[str, TensorType]:
        """Return custom metrics from the model."""
        # Attempt to access the `q_loss` of the model
        if hasattr(self, "tower_stats") and "q_loss" in self.tower_stats:
            q_loss: QLoss = self.tower_stats["q_loss"]
            if q_loss.loss.device == "cpu":
                loss = q_loss.loss.detach().item()
            else:
                loss = q_loss.loss.detach().cpu().item()
            return {"q_loss": loss}
        else:
            return {}
