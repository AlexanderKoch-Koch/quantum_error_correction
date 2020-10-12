import numpy as np
import torch

from rlpyt.utils.tensor import infer_leading_dims, restore_leading_dims
from rlpyt.models.mlp import MlpModel
from imitation_learning.models.transformer_models import DemonstrationTransformerModel, generate_square_subsequent_mask
from rlpyt.models.running_mean_std import RunningMeanStdModel
from imitation_learning.models.gated_transformer import GatedTransformer, SIZES
from rlpyt.utils.collections import namedarraytuple
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.framework import try_import_torch
from torch import nn


class ConvModel(TorchModelV2, torch.nn.Module):
    def __init__(self, observation_space, action_space, num_outputs, model_config, name):
        """Instantiate neural net modules according to inputs."""
        TorchModelV2.__init__(self, observation_space, action_space, num_outputs,
                              model_config, name)
        torch.nn.Module.__init__(self)
        self.shared_features_dim = 256
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=64, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=2, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=2, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(start_dim=1),
            # nn.Dropout(0.2),
            nn.Linear(288, 512),
            nn.ReLU(),
            nn.Linear(512, num_outputs)
        )

    def forward(self, input_dict, state, seq_lens):
        action_dist_params = self.conv_layers(input_dict['obs'].float())
        return action_dist_params, state
