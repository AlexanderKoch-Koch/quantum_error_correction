
from rlpyt.samplers.parallel.cpu.collectors import (CpuResetCollector,
                                                    CpuWaitResetCollector)
from rlpyt.samplers.parallel.gpu.collectors import (GpuResetCollector,
                                                    GpuWaitResetCollector)
import numpy as np
from rlpyt.samplers.async_.collectors import DoubleBufferCollectorMixin

from rlpyt.samplers.collectors import (DecorrelatingStartCollector,
                                       BaseEvalCollector)
from rlpyt.agents.base import AgentInputs
from rlpyt.utils.buffer import (torchify_buffer, numpify_buffer, buffer_from_example,
                                buffer_method)
from keras.models import load_model
import time


class QecDbCpuResetCollector(DoubleBufferCollectorMixin, CpuResetCollector):
    def __init__(self, rank, envs, *args, **kwargs):
        super().__init__(rank, envs, *args, **kwargs)
        if envs[0].error_model == 'X':
            self.static_decoder_path = '/home/alex/DeepQ-Decoding/example_notebooks/referee_decoders/nn_d5_X_p5'
        else:
            self.static_decoder_path = '/home/alex/DeepQ-Decoding/example_notebooks/referee_decoders/nn_d5_DP_p5'
        self.static_decoder = load_model(self.static_decoder_path, compile=True)

    def collect_batch(self, *args, **kwargs):
        """Swap in the called-for double buffer to record samples into."""
        self.samples_np = self.double_buffer[self.sync.db_idx.value]
        return self._collect_batch(*args, **kwargs)

    def _collect_batch(self, agent_inputs, traj_infos, itr):
        # Numpy arrays can be written to from numpy arrays or torch tensors
        # (whereas torch tensors can only be written to from torch tensors).
        agent_buf, env_buf = self.samples_np.agent, self.samples_np.env
        completed_infos = list()
        observation, action, reward = agent_inputs
        obs_pyt, act_pyt, rew_pyt = torchify_buffer(agent_inputs)
        agent_buf.prev_action[0] = action  # Leading prev_action.
        env_buf.prev_reward[0] = reward
        self.agent.sample_mode(itr)
        for t in range(self.batch_T):
            if (t * len(self.envs)) % 400 == 0:
                self.agent.recv_shared_memory()
            env_buf.observation[t] = observation
            # Agent inputs and outputs are torch tensors.
            act_pyt, agent_info = self.agent.step(obs_pyt, act_pyt, rew_pyt)
            action = numpify_buffer(act_pyt)
            static_decoder_inputs = []
            correct_labels = []
            env_infos = []
            done = []
            for b, env in enumerate(self.envs):
                # Environment inputs and outputs are numpy arrays.
                o, r, d, env_info = env.step(action[b])
                done.append(d)
                observation[b] = o
                reward[b] = r
                env_infos.append(env_info)
                static_decoder_inputs.append(env_info.static_decoder_input)
                correct_labels.append(env_info.correct_label)

            static_decoder_inputs = np.stack(static_decoder_inputs)
            correct_labels = np.stack(correct_labels)
            label_prediction = np.argmax(self.static_decoder(static_decoder_inputs), axis=-1).squeeze(axis=1)
            done = label_prediction != correct_labels

            for b, env in enumerate(self.envs):
                traj_infos[b].step(observation[b], action[b], reward[b], done[b], agent_info[b],
                                   env_infos[b])
                if getattr(env_info, "traj_done", done[b]):
                    completed_infos.append(traj_infos[b].terminate(observation[b]))
                    traj_infos[b] = self.TrajInfoCls()
                    observation[b] = env.reset()
                if done[b]:
                    self.agent.reset_one(idx=b)
                env_buf.done[t, b] = done[b]
                if env_info:
                    env_buf.env_info[t, b] = env_infos[b]
            agent_buf.action[t] = action
            env_buf.reward[t] = reward
            if agent_info:
                agent_buf.agent_info[t] = agent_info

        if "bootstrap_value" in agent_buf:
            # agent.value() should not advance rnn state.
            agent_buf.bootstrap_value[:] = self.agent.value(obs_pyt, act_pyt, rew_pyt)

        return AgentInputs(observation, action, reward), traj_infos, completed_infos


class QecCpuEvalCollector(BaseEvalCollector):
    """Offline agent evaluation collector which calls ``agent.step()`` in
    sampling loop.  Immediately resets any environment which finishes a
    trajectory.  Stops when the max time-steps have been reached, or when
    signaled by the master process (i.e. if enough trajectories have
    completed).
    """
    def __init__(self, rank, envs, *args, **kwargs):
        super().__init__(rank, envs, *args, **kwargs)
        if envs[0].error_model == 'X':
            self.static_decoder_path = '/home/alex/DeepQ-Decoding/example_notebooks/referee_decoders/nn_d5_X_p5'
        else:
            self.static_decoder_path = '/home/alex/DeepQ-Decoding/example_notebooks/referee_decoders/nn_d5_DP_p5'
        self.static_decoder = load_model(self.static_decoder_path, compile=True)

    def collect_evaluation(self, itr, max_episodes=1):
        assert len(self.envs) == 1, 'qec eval collector needs max 1 env. Otherwise evaluation will be biased'
        traj_infos = [self.TrajInfoCls() for _ in range(len(self.envs))]
        observations = list()
        for env in self.envs:
            observations.append(env.reset())
        observation = buffer_from_example(observations[0], len(self.envs))
        for b, o in enumerate(observations):
            observation[b] = o
        action = buffer_from_example(self.envs[0].action_space.null_value(),
                                     len(self.envs))
        reward = np.zeros(len(self.envs), dtype="float32")
        obs_pyt, act_pyt, rew_pyt = torchify_buffer((observation, action, reward))
        self.agent.reset()
        self.agent.eval_mode(itr)
        num_completed_episodes = 0
        for t in range(self.max_T):
            act_pyt, agent_info = self.agent.step(obs_pyt, act_pyt, rew_pyt)
            action = numpify_buffer(act_pyt)
            static_decoder_inputs = []
            correct_labels = []
            env_infos = []
            done = []
            for b, env in enumerate(self.envs):
                o, r, d, env_info = env.step(action[b])
                done.append(d)
                observation[b] = o
                reward[b] = r
                env_infos.append(env_info)
                static_decoder_inputs.append(env_info.static_decoder_input)
                correct_labels.append(env_info.correct_label)

            static_decoder_inputs = np.stack(static_decoder_inputs)
            correct_labels = np.stack(correct_labels)
            label_prediction = np.argmax(self.static_decoder(static_decoder_inputs), axis=-1).squeeze(axis=1)
            done = label_prediction != correct_labels

            for b, env in enumerate(self.envs):
                traj_infos[b].step(observation[b], action[b], reward[b], done[b],
                                   agent_info[b], env_infos[b])
                if getattr(env_infos[b], "traj_done", done[b]):
                    self.traj_infos_queue.put(traj_infos[b].terminate(observation[b]))
                    traj_infos[b] = self.TrajInfoCls()
                    observation[b] = env.reset()
                if done[b]:
                    action[b] = 0  # Next prev_action.
                    reward[b] = 0
                    self.agent.reset_one(idx=b)
                    num_completed_episodes += 1
            if num_completed_episodes >= max_episodes:
                print('reached max episodes')
                break
            if self.sync.stop_eval.value:
                print(f'sync stop')
                break
        self.traj_infos_queue.put(None)  # End sentinel.
