import numpy as np
import torch

from rlpyt.utils.tensor import infer_leading_dims, restore_leading_dims
from rlpyt.models.conv2d import Conv2dModel
from rlpyt.models.mlp import MlpModel
from rlpyt.models.dqn.dueling import DuelingHeadModel


class QECModel(torch.nn.Module):
    """Standard convolutional network for DQN.  2-D convolution for multiple
    video frames per observation, feeding an MLP for Q-value outputs for
    the action set.
    """

    def __init__(
            self,
            image_shape,
            output_size,
            fc_sizes=512,
            dueling=False,
            use_maxpool=False,
            channels=None,  # None uses default.
            kernel_sizes=None,
            strides=None,
            paddings=None,
    ):
        """Instantiates the neural network according to arguments; network defaults
        stored within this method."""
        super().__init__()
        self.dueling = dueling
        c, h, w = image_shape
        self.conv = Conv2dModel(
            in_channels=c,
            channels=channels or [32, 64, 64],
            kernel_sizes=kernel_sizes or [8, 4, 3],
            strides=strides or [4, 2, 1],
            paddings=paddings or [0, 1, 1],
            use_maxpool=use_maxpool,
        )
        conv_out_size = self.conv.conv_out_size(h, w)
        self.mlp = MlpModel(input_size=np.prod(image_shape),
                            hidden_sizes=[256,256],
                            output_size=None)
        # conv_out_size = 256
        if dueling:
            self.head = DuelingHeadModel(conv_out_size, fc_sizes, output_size)
        else:
            self.head = MlpModel(conv_out_size, fc_sizes, output_size)

    def forward(self, observation, prev_action, prev_reward):
        """
        Compute action Q-value estimates from input state.
        Infers leading dimensions of input: can be [T,B], [B], or []; provides
        returns with same leading dims.  Convolution layers process as [T*B,
        image_shape[0], image_shape[1],...,image_shape[-1]], with T=1,B=1 when not given.  Expects uint8 images in
        [0,255] and converts them to float32 in [0,1] (to minimize image data
        storage and transfer).  Used in both sampler and in algorithm (both
        via the agent).
        """
        img = observation.type(torch.float)  # Expect torch.uint8 inputs


        # Infer (presence of) leading dimensions: [T,B], [B], or [].
        lead_dim, T, B, img_shape = infer_leading_dims(img, 3)
        # features = self.mlp(img.reshape(T * B, -1))

        conv_out = self.conv(img.view(T * B, *img_shape))  # Fold if T dimension.
        q = self.head(conv_out.view(T * B, -1))
        # q = self.head(features)

        # Restore leading dimensions: [T,B], [B], or [], as input.
        q = restore_leading_dims(q, lead_dim, T, B)
        return q

class VmpoQECModel(torch.nn.Module):
    """Standard convolutional network for DQN.  2-D convolution for multiple
    video frames per observation, feeding an MLP for Q-value outputs for
    the action set.
    """

    def __init__(
            self,
            image_shape,
            action_size,
            fc_sizes=512,
            dueling=False,
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
        self.dueling = dueling
        c, h, w = image_shape
        self.conv = Conv2dModel(
            in_channels=c,
            channels=channels or [32, 64, 64],
            kernel_sizes=kernel_sizes or [3, 2, 2],
            strides=strides or [2, 1, 1],
            paddings=paddings or [0, 0, 0],
            use_maxpool=use_maxpool,
        )
        conv_out_size = self.conv.conv_out_size(h, w)
        self.pi_head = MlpModel(conv_out_size, [256,], action_size)
        self.value_head = MlpModel(conv_out_size, [256,], 1 if linear_value_output else None)

    def forward(self, observation, prev_action, prev_reward, init_rnn_state):
        img = observation.type(torch.float)  # Expect torch.uint8 inputs
        # Infer (presence of) leading dimensions: [T,B], [B], or [].
        lead_dim, T, B, img_shape = infer_leading_dims(img, 3)

        conv_out = self.conv(img.reshape(T * B, *img_shape))  # Fold if T dimension.
        features = conv_out.reshape(T * B, -1)
        pi = self.pi_head(features)
        pi = torch.softmax(pi, dim=-1)
        value = self.value_head(features).squeeze(-1)
        # Restore leading dimensions: [T,B], [B], or [], as input.
        pi, value = restore_leading_dims((pi, value), lead_dim, T, B)
        return pi, value, torch.zeros(1, B, 1)
