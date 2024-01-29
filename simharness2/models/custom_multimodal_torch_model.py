"""Custom model for running on multimodel data"""

from typing import Dict, List

import torch
from gymnasium.spaces import Space
from ray.rllib.models.torch.misc import SlimConv2d, same_padding, SlimFC
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import ModelConfigDict, TensorType


# FIXME: Move these constants to a more appropriate location.
FIRE_MAP_KEY = "fire_map"
POSITION_KEY = "position"


class CustomMultimodalTorchModel(TorchModelV2, torch.nn.Module):
    """Custom model for running on multimodel data"""

    def __init__(
        self,
        obs_space: Dict[str, Space],
        action_space: Space,
        num_outputs: int,
        model_config: ModelConfigDict,
        name: str,
    ):
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        torch.nn.Module.__init__(self)

        filters = self.model_config["conv_filters"]
        activation = self.model_config.get("conv_activation")
        (w, h, in_channels) = obs_space.original_space[FIRE_MAP_KEY].shape
        in_size = [w, h]
        layers = []
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
        in_size = (
            out_size[0] * out_size[1] * out_channels
            + obs_space.original_space[POSITION_KEY].shape[0]
        )
        layers.append(torch.nn.Flatten())
        fc_layers = []
        fc_layers_value = []
        for out_size in self.model_config["fcnet_hiddens"]:
            fc_layers.append(
                SlimFC(
                    in_size,
                    out_size,
                    activation_fn=torch.nn.ReLU,
                )
            )
            fc_layers_value.append(
                SlimFC(
                    in_size,
                    out_size,
                    activation_fn=torch.nn.ReLU,
                )
            )
            in_size = out_size
        fc_layers.append(SlimFC(in_size, num_outputs))
        fc_layers_value.append(SlimFC(in_size, 1))
        self._conv_model = torch.nn.Sequential(*layers)
        self._fc_model = torch.nn.Sequential(*fc_layers)
        self._fc_value = torch.nn.Sequential(*fc_layers_value)

    @override(TorchModelV2)
    def forward(
        self,
        input_dict: Dict[str, TensorType],
        state: List[TensorType],
        seq_lens: TensorType,
    ) -> (TensorType, List[TensorType]):
        if SampleBatch.OBS in input_dict and "obs_flat" in input_dict:
            orig_obs = input_dict[SampleBatch.OBS]
        else:
            orig_obs = restore_original_dimensions(
                input_dict[SampleBatch.OBS], self.obs_space, "torch"
            )
        fire_map = orig_obs[FIRE_MAP_KEY]
        fire_map = torch.transpose(fire_map, 1, -1)
        position = orig_obs[POSITION_KEY]
        conv_out = self._conv_model(fire_map)
        fc_input = torch.cat([conv_out, position], dim=-1)
        self._features = fc_input
        out = self._fc_model(fc_input)
        return out, state
