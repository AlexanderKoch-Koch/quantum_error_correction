import numpy as np
import torch

from rlpyt.utils.tensor import infer_leading_dims, restore_leading_dims
from rlpyt.models.conv2d import Conv2dModel
from rlpyt.models.mlp import MlpModel
from rlpyt.models.dqn.dueling import DuelingHeadModel
from rlpyt.models.pg.atari_lstm_model import RnnState
from imitation_learning.vmpo.compressive_transformer import SIZES, State
from imitation_learning.models.compressive_transformer_pytorch import CompressiveTransformerPyTorch, Memory


class RecurrentVmpoQECModel(torch.nn.Module):
    """Standard convolutional network for DQN.  2-D convolution for multiple
    video frames per observation, feeding an MLP for Q-value outputs for
    the action set.
    """

    def __init__(
            self,
            observation_shape,
            action_size,
            fc_sizes=512,
            lstm_size=64,
            lstm_layers=1,
            use_maxpool=False,
            channels=None,  # None uses default.
            kernel_sizes=None,
            strides=None,
            paddings=None,
            linear_value_output=True
    ):
        """Instantiates the neural network according to arguments; network defaults
        stored within this method."""
        super().__init__()
        c, h, w = observation_shape
        self.conv = Conv2dModel(
            in_channels=c,
            channels=channels or [32, 64, 64],
            kernel_sizes=kernel_sizes or [3, 2, 2],
            strides=strides or [2, 1, 1],
            paddings=paddings or [0, 0, 0],
            use_maxpool=use_maxpool,
        )
        conv_out_size = self.conv.conv_out_size(h, w)
        self.lstm = torch.nn.LSTM(conv_out_size, lstm_size,lstm_layers)
        self.pi_head = MlpModel(conv_out_size + lstm_size, [fc_sizes,], action_size)
        self.value_head = MlpModel(conv_out_size + lstm_size, [256,], 1 if linear_value_output else None)

    def forward(self, observation, prev_action, prev_reward, init_rnn_state):
        img = observation.type(torch.float)  # Expect torch.uint8 inputs
        # Infer (presence of) leading dimensions: [T,B], [B], or [].
        lead_dim, T, B, img_shape = infer_leading_dims(img, 3)

        conv_out = self.conv(img.reshape(T * B, *img_shape))  # Fold if T dimension.
        features = conv_out.reshape(T * B, -1)

        init_rnn_state = None if init_rnn_state is None else tuple(init_rnn_state)
        lstm_out, (hidden_state, cell_state) = self.lstm(features.reshape(T, B, -1), init_rnn_state)
        head_input = torch.cat((features, lstm_out.reshape(T * B, -1)), dim=-1)

        pi = self.pi_head(head_input)
        value = self.value_head(head_input).squeeze(-1)

        # Restore leading dimensions: [T,B], [B], or [], as input.
        pi, value = restore_leading_dims((pi, value), lead_dim, T, B)
        state = RnnState(h=hidden_state, c=cell_state)
        return pi, value, state


class SingleActionRecurrentQECModel(RecurrentVmpoQECModel):

    def forward(self, *args, **kwargs):
        pi, value, state = super().forward(*args, **kwargs)
        pi = torch.softmax(pi, dim=-1)
        return pi, value, state


class MultiActionRecurrentQECModel(RecurrentVmpoQECModel):

    def forward(self, *args, **kwargs):
        pi, value, state = super().forward(*args, **kwargs)
        pi = torch.sigmoid(pi - 2)
        return pi, value, state
