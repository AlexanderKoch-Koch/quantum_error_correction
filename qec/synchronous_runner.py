from rlpyt.runners.minibatch_rl import MinibatchRlEval
from rlpyt.utils.logging import logger
import numpy as np

class QECSynchronousRunner(MinibatchRlEval):
    def train(self):
        """
        Performs startup, evaluates the initial agent, then loops by
        alternating between ``sampler.obtain_samples()`` and
        ``algo.optimize_agent()``.  Pauses to evaluate the agent at the
        specified log interval.
        """
        n_itr = self.startup()
        with logger.prefix(f"itr #0 "):
            eval_traj_infos, eval_time = self.evaluate_agent(0)
            self.log_diagnostics(0, eval_traj_infos, eval_time)
        for itr in range(n_itr):
            logger.set_iteration(self.get_cum_steps(itr))
            with logger.prefix(f"itr #{itr} "):
                self.agent.sample_mode(itr)
                samples, traj_infos = self.sampler.obtain_samples(itr)
                self.agent.train_mode(itr)
                opt_info = self.algo.optimize_agent(itr, samples)
                self.store_diagnostics(itr, traj_infos, opt_info)
                if (itr + 1) % self.log_interval_itrs == 0:
                    eval_traj_infos, eval_time = self.evaluate_agent(itr)
                    p_error = self.sampler.env_kwargs['error_rate']
                    for traj_info in eval_traj_infos:
                        traj_info['p_error'] = p_error
                    self.log_diagnostics(itr, eval_traj_infos, eval_time)
                    avg_lifetime = np.nanmean(np.array([x['lifetime'] for x in eval_traj_infos]))
                    if avg_lifetime > (1/p_error):
                        if p_error + 0.002 <= 0.011001:
                            p_error += 0.002
                            self.sampler.env_kwargs['error_rate'] = p_error
                            self.sampler.eval_env_kwargs['error_rate'] = p_error
                            print(f'new p error is {p_error}', flush=True)
                            self.shutdown()
                            self.startup()
                        else:
                            print(f'didnt change p_error - currently at {p_error}')

                        # for env in self.sampler.collector.envs + self.sampler.eval_collector.envs:
                        #     env.p_phys = new_p_error
                        #     env.p_meas = new_p_error
        print(f'training end due to n_itr', flush=True)
        self.shutdown()
