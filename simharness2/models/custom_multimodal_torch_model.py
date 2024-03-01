"""Custom model for running on multimodel data"""

from typing import Dict, List, Tuple

import torch
from gymnasium.spaces import Space
from ray.rllib.models.modelv2 import restore_original_dimensions
from ray.rllib.models.torch.misc import SlimConv2d, SlimFC, same_padding
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.utils import get_activation_fn, get_filter_config
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import ModelConfigDict, TensorType


# FIXME: Move these constants to a more appropriate location.
FIRE_MAP_KEY = "fire_map"
AGENT_POSITION_KEY = "agent_pos"


class CustomMultimodalTorchModel(TorchModelV2, torch.nn.Module):
    """Custom model for running on multimodel data"""

    def __init__(
        self,
        obs_space: Space,
        action_space: Space,
        num_outputs: int,
        model_config: ModelConfigDict,
        name: str,
    ):
        """TODO: Add docstring."""
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        torch.nn.Module.__init__(self)

        # NOTE: obs_space layout can change if no preprocessor is used. For more details,
        # see the `_disable_preprocessor_api` parameter here:
        # https://docs.ray.io/en/latest/rllib/rllib-training.html#specifying-experimental-features
        original_obs_space: Dict[str, Space]
        if self.model_config.get("_disable_preprocessor_api"):
            original_obs_space = obs_space
        else:
            original_obs_space = obs_space.original_space

        # Validate the provided config settings for conv
        # TODO: Provide a better default config for conv_filters?
        if not self.model_config.get("conv_filters"):
            self.model_config["conv_filters"] = get_filter_config(
                original_obs_space[FIRE_MAP_KEY].shape
            )
        filters = self.model_config["conv_filters"]
        assert len(filters) > 0, "Must provide at least 1 entry in `conv_filters`!"
        conv_activation = get_activation_fn(
            self.model_config.get("conv_activation"), framework="torch"
        )

        # Construct conv model, where fire map is expected input.
        # FIXME: Decide if we want "fire_map" to be 2D or 3D.
        in_size = original_obs_space[FIRE_MAP_KEY].shape[:2]
        in_channels = original_obs_space[FIRE_MAP_KEY].shape[-1]
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
                    activation_fn=conv_activation,
                )
            )
            in_channels = out_channels
            in_size = out_size

        layers.append(torch.nn.Flatten())
        self._conv_model = torch.nn.Sequential(*layers)

        # Construct fc model, where conv model output and position are expected input.
        fcnet_activation = get_activation_fn(
            self.model_config.get("fcnet_activation"), framework="torch"
        )
        in_size = (
            out_size[0] * out_size[1] * out_channels
            + original_obs_space[AGENT_POSITION_KEY].shape[0]
        )
        fc_layers, fc_layers_value = [], []
        for out_size in self.model_config.get("fcnet_hiddens", []):
            fc_layers.append(
                SlimFC(
                    in_size,
                    out_size,
                    activation_fn=fcnet_activation,
                )
            )
            fc_layers_value.append(
                SlimFC(
                    in_size,
                    out_size,
                    activation_fn=fcnet_activation,
                )
            )
            in_size = out_size

        fc_layers.append(SlimFC(in_size, num_outputs))
        fc_layers_value.append(SlimFC(in_size, 1))
        self._fc_model = torch.nn.Sequential(*fc_layers)
        self._fc_value = torch.nn.Sequential(*fc_layers_value)

    @override(TorchModelV2)
    def forward(
        self,
        input_dict: Dict[str, TensorType],
        state: List[TensorType],
        seq_lens: TensorType,
    ) -> Tuple[TensorType, List[TensorType]]:
        # Extract the original observation from the input_dict
        if SampleBatch.OBS in input_dict and "obs_flat" in input_dict:
            orig_obs = input_dict[SampleBatch.OBS]
        else:
            orig_obs = restore_original_dimensions(
                input_dict[SampleBatch.OBS], self.obs_space, "torch"
            )

        # FIXME: If we want the fire_map in channel-major format, we can simply return
        # the correct layout from the harness; should not NEED to transpose here.
        fire_map = torch.transpose(orig_obs[FIRE_MAP_KEY], 1, -1)
        conv_out = self._conv_model(fire_map)
        self._features = torch.cat([conv_out, orig_obs[AGENT_POSITION_KEY]], dim=-1)
        out = self._fc_model(self._features)
        return out, state

    @override(TorchModelV2)
    def value_function(self) -> TensorType:
        return self._fc_value(self._features).squeeze(1)
