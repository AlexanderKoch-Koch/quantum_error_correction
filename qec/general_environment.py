from qec.optimized_environment import OptimizedSurfaceCodeEnvironment
import time
import numpy as np
from qec.Function_Library import *

class GeneralSurfaceCodeEnv(OptimizedSurfaceCodeEnvironment):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, volume_depth=1)


    def step(self, action):
        """
        Given an action, this method executes the logic of the environment.

        :param: action: The action, given as an integer, supplied by some agent.
        :return: self.board_state: The new reset visible state of the environment = syndrome volume + action history volume
        :return: reward: The reward for the action
        :return: self.done: The boolean terminal state indicator
        :return: info: A dictionary via which additional diagnostic information can be provided. Empty here.
        """
        start = time.time()
        done_identity = False
        action = int(action)  # make sure action is integer
        if action == self.identity_index or int(self.completed_actions[action]) == 1:
            done_identity = True
        # 1) Apply the action to the hidden state
        action_lattice = index_to_move(self.d, action, self.error_model, self.use_Y)
        self.hidden_state = obtain_new_error_configuration(self.hidden_state, action_lattice)

        # 2) Calculate the reward
        self.current_true_syndrome = self.generate_surface_code_syndrome_NoFT_efficient(self.hidden_state, self.qubits)
        # print(f'afteer syndrome generation {time.time() - s}')
        current_true_syndrome_vector = np.reshape(self.current_true_syndrome, (self.d + 1) ** 2)
        num_anyons = np.sum(self.current_true_syndrome)
        # print(f'1 {time.time() - s}')

        correct_label = np.argmax(generate_one_hot_labels_surface_code(self.hidden_state, self.error_model))
        static_decoder_input = np.array([current_true_syndrome_vector])
        reward = 0
        if np.argmax(correct_label) == 0 and num_anyons == 0:
            reward = 1.0

        error = generate_error(self.d, self.p_phys, self.error_model)
        if int(np.sum(error) != 0):
            self.hidden_state = obtain_new_error_configuration(self.hidden_state, error)
            self.current_true_syndrome = self.generate_surface_code_syndrome_NoFT_efficient(self.hidden_state,
                                                                                            self.qubits)
        current_faulty_syndrome = generate_faulty_syndrome(self.current_true_syndrome, self.p_meas)
        self.lifetime += 1
        obs = [self.padding_syndrome(current_faulty_syndrome)]
        for k in range(self.n_action_layers):
            obs.append(self.padding_actions(self.completed_actions[k * self.d ** 2:(k + 1) * self.d ** 2]))
        obs = np.stack(obs)
        info = dict(lifetime=self.lifetime,
                    static_decoder_input=static_decoder_input,
                    correct_label=correct_label)
        return obs, reward, self.done, info


if __name__ == '__main__':
    env = GeneralSurfaceCodeEnv(error_model='X', )
    while True:
        done = False
        obs = env.reset()
        while not done:
            action = env.action_space.sample()
            s = time.time()
            obs, reward, done, info = env.step(action)
            print(f'step took {time.time() - s}')