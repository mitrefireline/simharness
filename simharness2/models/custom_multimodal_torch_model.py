"""Custom model for running on multimodel data"""
from typing import Dict, List
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.typing import ModelConfigDict, TensorType
from ray.rllib.utils.annotations import override
import torch
import gymnasium as gym
from ray.rllib.models.torch.misc import (
    normc_initializer,
    same_padding,
    SlimConv2d,
    SlimFC,
)


class CustomMultimodalTorchModel(TorchModelV2, torch.nn.Module):
    def __init__(
            self,
            obs_space: Dict[str, gym.spaces.Space],
            action_space: gym.spaces.Space,
            num_outputs: int,
            model_config: ModelConfigDict,
            name: str,
    ):
        TorchModelV2.__init__(
                self, obs_space, action_space, num_outputs, model_config, name
        )
        torch.nn.Module.__init__(self)

        filters = self.model_config["conv_filters"]
        (w, h, in_channels) = obs_space["firemap"].shape
        in_size = [w, h]
        for out_channels, kernel, stride in filters:
                padding, out_size = same_padding(in_size, kernel, stride)
                layers.append(
                        SlimConv2d(
                        in_channels,
                        out_channels,
                        kernel,
                        stride,
                        padding,
                        activation_fn=activation,
                        )       
                )
                in_channels = out_channels
                in_size = out_size
        in_size = out_size[0]*out_size[1]*out_channels + obs_space["position"].shape[0]
        layers.append(torch.nn.Flatten())
        fc_layers = []
        for out_size in self.model_config["fcnet_hiddens"]:
                fc_layers.append(torch.nn.Linear(in_size, out_size))
                in_size = out_size
        fc_layers.append(torch.nn.Linear(in_size, action_space.n))
        self._conv_model = torch.nn.Sequential(*layers)
        self._fc_model = torch.nn.Sequential(*fc_layers)


    @override(TorchModelV2)
    def forward(
            self,
            input_dict: Dict[str, TensorType],
            state: List[TensorType],
            seq_lens: TensorType,
    ) -> (TensorType, List[TensorType]):
        image = input_dict["obs"]["firemap"]
        position = input_dict["obs"]["position"]
        conv_out = self._conv_model(image)
        fc_input = torch.cat([conv_out, posiiont], dim=-1)
        out = self.fc_layers(fc_input)
        return out
