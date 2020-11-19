import numpy as np
import torch

from rlpyt.utils.tensor import infer_leading_dims, restore_leading_dims
from rlpyt.models.conv2d import Conv2dModel
from rlpyt.models.mlp import MlpModel
from rlpyt.models.dqn.dueling import DuelingHeadModel
from rlpyt.models.pg.atari_lstm_model import RnnState
from imitation_learning.vmpo.compressive_transformer import SIZES, State
from imitation_learning.models.compressive_transformer_pytorch import CompressiveTransformerPyTorch, Memory


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
                            hidden_sizes=[512,],
                            output_size=None)
        self.dropout = torch.nn.Dropout(0.2)
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
        # q = self.head(self.dropout(conv_out.view(T * B, -1)))
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
            observation_shape,
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
        self.lstm = torch.nn.LSTM(conv_out_size, lstm_size)
        self.pi_head = MlpModel(conv_out_size + lstm_size, [256,], action_size)
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
        pi = torch.softmax(pi, dim=-1)
        value = self.value_head(head_input).squeeze(-1)

        # Restore leading dimensions: [T,B], [B], or [], as input.
        pi, value = restore_leading_dims((pi, value), lead_dim, T, B)
        state = RnnState(h=hidden_state, c=cell_state)
        return pi, value, state



class QECTransformerModel(torch.nn.Module):
    def __init__(self,
                 observation_shape,
                 action_size,
                 linear_value_output=True,
                 sequence_length=50,
                 seperate_value_network=True,
                 size='small',
                 channels=None,  # None uses default.
                 kernel_sizes=None,
                 strides=None,
                 paddings=None,
                 ):
        super().__init__()
        self.action_size = action_size
        c, h, w = observation_shape
        self.conv = Conv2dModel(
            in_channels=c,
            channels=channels or [32, 64, 64],
            kernel_sizes=kernel_sizes or [3, 2, 2],
            strides=strides or [2, 1, 1],
            paddings=paddings or [0, 0, 0],
            use_maxpool=False,
        )
        self.conv_out_size = self.conv.conv_out_size(h, w)
        self.sequence_length = sequence_length
        self.transformer_dim = 32 # SIZES[size]['dim']
        self.depth = SIZES[size]['depth']
        self.cmem_ratio = SIZES[size]['cmem_ratio']
        self.cmem_length = self.sequence_length // self.cmem_ratio
        memory_layers = range(1, self.depth + 1)
        self.transformer = CompressiveTransformerPyTorch(
            num_tokens=20000,
            emb_dim=self.conv_out_size,
            dim=self.transformer_dim,
            heads=SIZES[size]['num_heads'],
            depth=self.depth,
            seq_len=self.sequence_length,
            mem_len=self.sequence_length,  # memory length
            reconstruction_loss_weight=1,  # weight to place on compressed memory reconstruction loss
            gru_gated_residual=True,
            # whether to gate the residual intersection, from 'Stabilizing Transformer for RL' paper
            memory_layers=memory_layers,
        )
        self.transformer.token_emb = torch.nn.Identity()  # don't use token embedding in compressive transforrmer
        self.transformer.to_logits = torch.nn.Identity()
        # self.input_layer_norm = torch.nn.LayerNorm(self.state_size)
        self.input_layer_norm = torch.nn.Identity()
        self.output_layer_norm = torch.nn.LayerNorm(self.transformer_dim)
        # self.output_layer_norm = torch.nn.Identity()
        self.softplus = torch.nn.Softplus()
        self.pi_head = MlpModel(input_size=self.transformer_dim,
                                hidden_sizes=[256, ],
                                output_size=action_size)
        self.value_head = MlpModel(input_size=self.transformer_dim,
                                   hidden_sizes=[256, ],
                                   output_size=1 if linear_value_output else None)
        self.mask = torch.ones((self.sequence_length, self.sequence_length), dtype=torch.int8).triu()

    def forward(self, observation, prev_action, prev_reward, state):
        img = observation.type(torch.float)  # Expect torch.uint8 inputs
        lead_dim, T, B, img_shape = infer_leading_dims(img, 3)
        conv_out = self.conv(img.view(T * B, *img_shape)).reshape(T, B , -1)

        if T == 1:
            transformer_output, state = self.sample_forward(conv_out, state)
            value = torch.zeros(B)
        elif T == self.sequence_length:
            transformer_output, aux_loss = self.optim_forward(conv_out, state)
            value = self.value_head(transformer_output).reshape(T * B, -1)
        else:
            raise NotImplementedError

        pi_output = self.pi_head(transformer_output).view(T * B, -1)
        pi = torch.softmax(pi_output, dim=-1)

        pi, value = restore_leading_dims((pi, value), lead_dim, T, B)
        return pi, value, state

    def sample_forward(self, observation, state):
        lead_dim, T, B, _ = infer_leading_dims(observation, 1)
        observation = self.input_layer_norm(observation)

        device = observation.device
        if state is None:
            observations = torch.zeros((self.sequence_length, B, self.conv_out_size), device=device)
            length = torch.zeros((1, B, 1), device=device, dtype=torch.int64)
            memory = torch.zeros((self.depth, B, self.sequence_length, self.transformer_dim), device=device)
            compressed_memory = torch.zeros((self.depth, B, self.cmem_length, self.transformer_dim), device=device)
        else:
            observations = state.sequence
            length = state.length.clamp_max(self.sequence_length - 1)
            memory = state.memory
            compressed_memory = state.compressed_memory

        # write new observations in tensor with older observations
        observation = observation.view(B, -1)
        indexes = tuple(torch.cat((length[0, :], torch.arange(B, device=device).unsqueeze(-1)), dim=-1).t())
        observations.index_put_(indexes, observation)

        transformer_output, new_memory, _ = self.transformer(observations.transpose(0, 1), Memory(mem=memory, compressed_mem=None))
        transformer_output = self.output_layer_norm(transformer_output).transpose(0,1)
        output = transformer_output[length[0, :, 0], torch.arange(B)]
        length = torch.fmod(length + 1, self.sequence_length)

        reset = (length == 0).int()[0, :, 0].reshape(B, 1, 1, 1).transpose(0, 1).expand_as(memory)
        memory = reset * new_memory.mem + (1 - reset) * memory

        state = State(sequence=observations, length=length, memory=memory, compressed_memory=compressed_memory)
        return output, state

    def optim_forward(self, observation, state):
        observation = self.input_layer_norm(observation)
        output, _, aux_loss = self.transformer(observation.transpose(0, 1), Memory(mem=state.memory, compressed_mem=None))
        output = self.output_layer_norm(output)
        return output.transpose(0, 1), aux_loss
