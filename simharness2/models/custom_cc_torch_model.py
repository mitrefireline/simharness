"""TODO."""
from ray.rllib.models.torch.complex_input_net import ComplexInputNetwork as TorchCIN
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.framework import try_import_torch

torch, nn = try_import_torch()


class YetAnotherTorchCentralizedCriticModel(TorchModelV2, nn.Module):
    """Multi-agent model that implements a centralized value function.

    It assumes the observation is a dict with 'own_obs' and 'opponent_obs', the
    former of which can be used for computing actions (i.e., decentralized
    execution), and the latter for optimization (i.e., centralized learning).

    This model has two parts:
    - An action model that looks at just 'own_obs' to compute actions
    - A value model that also looks at the 'opponent_obs' / 'opponent_action'
      to compute the value (it does this by using the 'obs_flat' tensor).
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        """TODO.

        Args:
            obs_space (_type_): _description_
            action_space (_type_): _description_
            num_outputs (_type_): _description_
            model_config (_type_): _description_
            name (_type_): _description_
        """
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)

        self.action_model = TorchCIN(
            obs_space,
            action_space,
            num_outputs,
            model_config,
            name + "_action",
        )

        self.value_model = TorchCIN(
            obs_space, action_space, 1, model_config, name + "_vf"
        )
        self._model_in = None

    def forward(self, input_dict, state, seq_lens):
        """TODO.

        Args:
            input_dict (_type_): _description_
            state (_type_): _description_
            seq_lens (_type_): _description_

        Returns:
            _type_: _description_
        """
        # Store model-input for possible `value_function()` call.
        self._model_in = [input_dict["obs"]["own_obs"], state, seq_lens]

        return self.action_model({"obs": input_dict["obs"]["own_obs"]}, state, seq_lens)

    def value_function(self):
        """TODO.

        Returns:
            _type_: _description_
        """
        value_out, _ = self.value_model(
            {"obs": self._model_in[0]}, self._model_in[1], self._model_in[2]
        )
        return torch.reshape(value_out, [-1])
