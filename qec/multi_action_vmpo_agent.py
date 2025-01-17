import torch

from rlpyt.agents.base import (AgentStep, BaseAgent, RecurrentAgentMixin,
                               AlternatingRecurrentAgentMixin)
from rlpyt.agents.pg.categorical import CategoricalPgAgent, RecurrentCategoricalPgAgent
from rlpyt.agents.pg.base import AgentInfo, AgentInfoRnn
from rlpyt.distributions.categorical import Categorical, DistInfo
from rlpyt.utils.buffer import buffer_to, buffer_func, buffer_method
from qec.independent_bernoulli_dist import IndependentBernoulli


class MultiActionVmpoAgent(RecurrentCategoricalPgAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._mode = 'eval'

    def __call__(self, observation, prev_action, prev_reward, init_rnn_state):
        # Assume init_rnn_state already shaped: [N,B,H]
        model_inputs = buffer_to((observation, prev_action, prev_reward,
                                  init_rnn_state), device=self.device)
        pi, value, next_rnn_state = self.model(*model_inputs)
        dist_info, value = buffer_to((DistInfo(prob=pi), value), device="cpu")
        return dist_info, value, next_rnn_state  # Leave rnn_state on device.

    def initialize(self, env_spaces, share_memory=False,
                   global_B=1, env_ranks=None):
        super().initialize(env_spaces, share_memory,
                           global_B=global_B, env_ranks=env_ranks)
        # self.distribution = Categorical(dim=env_spaces.action.n)
        self.distribution = IndependentBernoulli()

    def make_env_to_model_kwargs(self, env_spaces):
        return dict(observation_shape=env_spaces.observation.shape, action_size=env_spaces.action.n)

    @torch.no_grad()
    def step(self, observation, prev_action, prev_reward):
        # prev_action = self.distribution.to_onehot(prev_action)
        agent_inputs = buffer_to((observation, prev_action, prev_reward),
                                 device=self.device)
        probs, value, rnn_state = self.model(*agent_inputs, self.prev_rnn_state)
        dist_info = DistInfo(prob=probs)
        if self._mode == 'sample':
            action = self.distribution.sample(dist_info)
        elif self._mode == 'eval':
            action = self.distribution.sample_deterministically(dist_info)
        # Model handles None, but Buffer does not, make zeros if needed:
        if self.prev_rnn_state is None:
            prev_rnn_state = buffer_func(rnn_state, torch.zeros_like)
        else:
            prev_rnn_state = self.prev_rnn_state
        # Transpose the rnn_state from [N,B,H] --> [B,N,H] for storage.
        # (Special case: model should always leave B dimension in.)
        prev_rnn_state = buffer_method(prev_rnn_state, "transpose", 0, 1)
        agent_info = AgentInfoRnn(dist_info=dist_info, value=value,
                                  prev_rnn_state=prev_rnn_state)
        action, agent_info = buffer_to((action, agent_info), device="cpu")
        self.advance_rnn_state(rnn_state)  # Keep on device.
        return AgentStep(action=action, agent_info=agent_info)
