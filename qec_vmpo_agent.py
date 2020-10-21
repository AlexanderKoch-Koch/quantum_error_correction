from rlpyt.agents.base import (AgentStep, BaseAgent, RecurrentAgentMixin,
                               AlternatingRecurrentAgentMixin)
from torch.distributions.multivariate_normal import MultivariateNormal
from rlpyt.utils.collections import namedarraytuple
from rlpyt.agents.pg.categorical import CategoricalPgAgent, RecurrentCategoricalPgAgent

DistInfo = namedarraytuple("DistInfo", ["mean", 'std'])

class QECVmpoAgent(RecurrentCategoricalPgAgent):
    def make_env_to_model_kwargs(self, env_spaces):
        return dict(image_shape=env_spaces.observation.shape, action_size=env_spaces.action.n)

    # def eval_mode(self, itr):
    #     super().eval_mode(itr)
    #     # print('eval mode #################')
    #     self.distribution.set_std(0)
    #
    # def sample_mode(self, itr):
    #     super().sample_mode(itr)
    #     # print("sample mode ############################")
    #     self.distribution.set_std(None)
    #
