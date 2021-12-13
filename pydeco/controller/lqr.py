import logging
from time import perf_counter
from queue import Queue

import numpy as np

from pydeco.types import *
from pydeco.constants import PolicyType, TrainMethod, NoiseShape
from pydeco.problem.lq import LQ
from pydeco.controller.agent import Agent


class LQR(Agent):
    def __init__(
            self,
            name: str = 'LQR',
            noise_type: str = NoiseShape.MV_NORMAL,
            verbose: bool = True,
    ):
        # logger
        self._configure_logger(name, verbose)

        # cache for visualization and logging
        self._cache = {}

        # noise
        self._noise_cache = Queue()
        self._noise_type = noise_type
        self._noise_params = None

        # model
        # Value function matrix (n_s, n_s)
        self._P = None
        # Policy matrix (n_s, n_a)
        self._K = None

    @property
    def K(self):
        return self._K

    @property
    def P(self):
        return self._P

    @K.setter
    def K(self, K: Tensor):
        self._K = K

    @P.setter
    def P(self, P: Tensor):
        self._P = P

    def act(
            self,
            state: Tensor,
            policy_type: PolicyType = PolicyType.GREEDY,
            **kwargs
    ) -> Tensor:
        if policy_type == PolicyType.GREEDY:
            return self._K @ state
        elif policy_type == PolicyType.EPS_GREEDY:
            if self._noise_cache.empty():
                self._refill_noise_cache()

            noise = self._noise_cache.get().reshape((-1, 1))
            return self._K @ state + noise
        else:
            raise ValueError(f'Policy {policy_type} not supported.')

    def train(
            self,
            env: LQ,
            method: str,
            gamma: Scalar = 1.,
            eps: Scalar = 1e-8,
            max_iter: int = 100,
            max_policy_evals: int = 80,
            max_policy_improves: int = 20,
            initial_state: Tensor | None = None,
            initial_policy: Tensor | None = None,
            alpha: Scalar = 0.01,
            **kwargs
    ):
        if self._calibrated:
            self._logger.warning('Controller has already been calibrated! Re-running the calculation.')

        self._logger.info(f'Calibrating controller.')

        # find optimal policy
        match method:
            case TrainMethod.ANALYTICAL:
                P, K = self._train_analytical_lqr(
                    env,
                    gamma,
                    eps,
                    max_iter,
                )
            case TrainMethod.QLEARN:
                P, K = self._train_qlearn_lqr(
                    env,
                    initial_state,
                    initial_policy,
                    gamma,
                    alpha=alpha,
                    max_iter=max_iter,
                )
            case TrainMethod.QLEARN_LS:
                P, K = self._train_qlearn_ls_lqr(
                    env,
                    initial_state,
                    initial_policy,
                    gamma,
                    max_policy_evals=max_policy_evals,
                    max_policy_improves=max_policy_improves,
                    eps=eps,
                )
            case _:
                raise ValueError(f'Method {method} not supported.')

        # save calibrated values
        self._P = P
        self._K = K
        self._calibrated = True

    def simulate_trajectory(
            self,
            env: LQ,
            initial_state: Tensor,
            t0: float,
            tn: float,
            n_steps: int,
    ) -> Tuple[Tensor, Tensor, float]:
        if not self._calibrated:
            self._logger.warning('Controller has not been calibrated!')
            return tuple()

        self._logger.info(f'Simulating a trajectory.')

        # starting x0
        env.reset(initial_state)

        # clear previous results
        # self._cache.clear()

        # calculate optimal controls
        time_grid = np.linspace(t0, tn, num=n_steps + 1)
        x_k = env.get_state()
        xs = [x_k]
        us = []
        total_cost = .0
        for k in time_grid[:-1]:
            # optimal control at k
            u_k = self.act(x_k)
            us.append(u_k)

            # update state
            r_k, next_x_k = env.step(u_k)
            xs.append(x_k)

            # increment stage cost
            total_cost += r_k

            # update state
            x_k = next_x_k

        rf = env.terminal_cost(x_k)
        total_cost += rf

        xs = np.squeeze(np.stack(xs))
        us = np.squeeze(np.stack(us))

        return xs, us, total_cost

    def _configure_logger(
            self,
            name: str,
            verbose: bool,
    ):
        self._logger = logging.getLogger(name)

        if self._logger.hasHandlers():
            self._logger.handlers.clear()

        handler = logging.StreamHandler()
        # formatter = logging.Formatter('%(levelname)s:%(name)s:%(message)s')
        formatter = logging.Formatter('%(message)s')
        handler.setFormatter(formatter)
        self._logger.addHandler(handler)
        self._logger.setLevel(logging.INFO if verbose else logging.WARNING)
        self._logger.info(f'Initializing {name} controller.')

    def _save_param(
            self,
            k: int,
            name: str,
            value: Union[float, int, str, Tensor],
    ):
        if k not in self._cache:
            self._cache[k] = {}

        if name in self._cache[k]:
            self._logger.error(f'{name} has already been cached for time step {k}!')

        self._cache[k][name] = value

    def _log_header(self):
        i = 'iter'
        loss = 'loss'
        kkt = 'kkt_viol'
        alpha = 'alpha'
        bt_n = 'bt_N'
        sigma = 'sigma'

        msg = f"|{i:<5}|{loss:<10}|{kkt:<10}|{alpha:<10}|{bt_n:<5}|{'infs':<5}|{sigma:<8}|"
        msg2 = f"{'d_min':<8}|{'d_max':<8}|{'kkt_eq':<9}|{'kkt_ineq':<9}|{'kkt_lgr':<9}|"

        self._logger.info(msg + msg2)

    def _log_step(
            self,
            k: int,
    ):
        sc = self._cache[k]
        loss = sc['loss']
        kkt = sc['penalty']
        alpha = sc['alpha']
        bt_n = sc['bt_n']
        sigma = sc['sigma']

        d_min = sc['min_d']
        d_max = sc['max_d']
        max_c_eq = sc['max_c_eq']
        max_c_ineq = sc['max_c_ineq']
        max_grad_lgr = sc['max_grad_Lagrangian']

        is_infs = 'Y' if sc['is_infeasible'] else 'N'

        msg1 = f'|{k:<5}|{loss:<10.4f}|{kkt:<10.4f}|{alpha:<10.6f}|{bt_n:<5}|{is_infs:<5}|{sigma:<8.4f}|'
        msg2 = f'{d_min:<8.2f}|{d_max:<8.2f}|{max_c_eq:<9.4f}|{max_c_ineq:<9.4f}|{max_grad_lgr:<9.4f}|'

        self._logger.info(msg1 + msg2)

    def _train_analytical_lqr(
            self,
            env: LQ,
            gamma: Scalar,
            eps: Scalar,
            max_iter,
            **kwargs
    ):
        # analytical solution requires access to the LQ model
        A, B, Q, R = env.get_model()

        # apply discounting
        A = np.sqrt(gamma) * A
        B = np.sqrt(gamma) * B

        # calculate P, K by iteratively updating P
        err = np.inf
        iter = 0

        P = Q
        while err > eps:
            if iter > max_iter:
                raise ValueError(f'Max number of iterations reached: {iter}.')

            pa = P @ A
            pb = P @ B
            P_next = Q + A.T @ pa - A.T @ pb @ np.linalg.inv(R + B.T @ pb) @ B.T @ pa

            err = np.max(np.abs(P_next - P))

            P = P_next
            iter += 1

        K = - np.linalg.inv(R + B.T @ P @ B) @ B.T @ P @ A

        self._logger.info(f'Converged after {iter} iterations.')

        return P, K

    def _ls_policy_eval(self):
        pass

    def _ls_policy_improve(self):
        pass

    def _train_qlearn_ls_lqr(
            self,
            env: LQ,
            initial_state: Tensor,
            initial_policy: Tensor,
            gamma: Scalar = 1.,
            max_policy_evals: int = 80,
            max_policy_improves: int = 20,
            eps: Scalar = 1e-3,
            **kwargs
    ):
        # s
        curr_state = env.reset(initial_state)

        # clear previous results
        self._cache.clear()

        # initial stabilizing controller
        n_s = env.n_s
        n_a = env.n_a
        n_q = n_s + n_a
        p = int(n_q * (n_q + 1) / 2)

        self._noise_params = (np.zeros((n_a,)), np.eye(n_a), (1000,))

        self._K = initial_policy
        H_k = None

        beta = 1
        theta = np.full((p, 1), fill_value=.0)

        pi_improve_iter = 0
        pi_improve_converged = False

        while not pi_improve_converged and pi_improve_iter < max_policy_improves:

            pi_eval_iter = 0
            pi_eval_converged = False

            G_k = np.eye(p) * beta

            while not pi_eval_converged and pi_eval_iter < max_policy_evals:
                # a
                curr_action = self.act(curr_state, policy_type=PolicyType.EPS_GREEDY)

                # r, s'
                curr_reward, next_state = env.step(curr_action)

                # max_a'
                next_action = self.act(next_state, policy_type=PolicyType.GREEDY)

                # features from (s,a)
                f_x = self._build_feature_vector(curr_state, curr_action, p)
                f_x_next = self._build_feature_vector(next_state, next_action, p)

                # update params theta w/ RLS
                phi = f_x - gamma * f_x_next

                num = G_k @ phi * (curr_reward - phi.T @ theta)
                den = 1 + phi.T @ G_k @ phi
                theta_adj = num / den
                theta += theta_adj

                num2 = G_k @ phi @ phi.T @ G_k
                G_adj = num2 / den
                G_k -= G_adj

                # update state
                curr_state = next_state

                # update counter
                pi_eval_iter += 1

                # convergence
                theta_adj_diff = np.max(np.abs(theta_adj))
                pi_eval_converged = theta_adj_diff < eps

            # policy improvement
            H_k = self._convert_to_parameter_matrix(theta, n_q)
            H_uk = H_k[n_s:, :n_s]
            H_uu = H_k[n_s:, n_s:]

            # argmax policy
            K_new = -np.linalg.inv(H_uu) @ H_uk

            # update counter
            pi_improve_iter += 1

            # convergence
            # TODO consider stop at |P_new - P|
            pi_improve_converged = np.max(np.abs(K_new - self._K)) < eps
            self._K = K_new

        return H_k[:n_s, :n_s], self._K

    def _train_qlearn_lqr(
            self,
            env: LQ,
            initial_state: Tensor,
            initial_policy: Tensor,
            gamma: Scalar = 1.,
            alpha: Scalar = 0.005,
            max_iter: int = 50,
            eps: Scalar = 1e-8,
            **kwargs
    ):
        # s
        curr_state = env.reset(initial_state)

        # clear previous results
        self._cache.clear()

        # initial stabilizing controller
        n_s = env.n_s
        n_a = env.n_a
        n_q = n_s + n_a
        p = int(n_q * (n_q + 1) / 2)

        self._K = initial_policy
        H_k = None

        theta = np.full((p, 1), fill_value=.0)

        self._noise_params = (np.zeros((n_a,)), np.eye(n_a), (1000,))

        iter = 0
        converged = False

        while not converged and iter < max_iter:
            # a
            curr_action = self.act(curr_state, policy_type=PolicyType.EPS_GREEDY)

            # r, s'
            curr_reward, next_state = env.step(curr_action)

            # max_a'
            next_action = self.act(next_state, policy_type=PolicyType.GREEDY)

            # features from (s,a)
            f_x = self._build_feature_vector(curr_state, curr_action, p)
            f_x_next = self._build_feature_vector(next_state, next_action, p)

            # update params theta w/ Q-learning
            theta_adj = alpha * (curr_reward + self._q_value(f_x_next, theta, gamma)
                                 - self._q_value(f_x, theta)) * f_x
            theta += theta_adj

            # update state
            curr_state = next_state

            # policy improvement
            H_k = self._convert_to_parameter_matrix(theta, n_q)
            H_uk = H_k[n_s:, :n_s]
            H_uu = H_k[n_s:, n_s:]

            # argmax policy
            self._K = -np.linalg.inv(H_uu) @ H_uk

            # update counter
            iter += 1

            # convergence
            theta_adj_diff = np.max(np.abs(theta_adj))
            converged = theta_adj_diff < eps

        # self._H = H_k
        # P = H_k[:n_s, :n_s]
        # self._calibrated = True

        return H_k[:n_s, :n_s], self._K

    def _q_value(
            self,
            x,
            theta,
            gamma=1.,
    ):
        q = gamma * x.T @ theta
        return q.item()

    def _build_feature_vector(
            self,
            s: Tensor,
            a: Tensor,
            p: int,
    ) -> Tensor:
        # s, a -> theta
        sa = np.concatenate((s, a)).reshape((-1,))
        n = sa.shape[0]

        # TODO optimize building quadratic features
        # t1 = perf_counter()
        # y = np.multiply(sa.reshape((-1, 1)), sa.reshape((1, -1)))
        # y = np.outer(sa, sa)
        # y_ind = np.triu_indices(y.shape[0])
        # y_ = y[y_ind].reshape((-1, 1))
        # tf_1 = perf_counter()-t1
        # print(tf_1)

        # t2 = perf_counter()
        x = np.empty((p, 1))
        c = 0
        for i in range(n):
            for j in range(i, n):
                x[c, 0] = sa[i] * sa[j]
                c += 1

        # tf_2 = perf_counter()-t2
        # print(tf_2)

        return x

    def _convert_to_parameter_matrix(
            self,
            theta: Tensor,
            n_q: int,
    ) -> Tensor:
        # thetas -> H_u
        H_u = np.zeros((n_q, n_q))

        H_u[np.triu_indices(n_q)] = theta.reshape((-1,)) * 0.5
        # H_u[np.tril_indices(n_q)] = theta.reshape((-1,))

        H_u += H_u.T

        # di = np.diag_indices(n_q)
        # H_u[di] = H_u[di] * 0.5

        return H_u

    def _convert_to_parameter_vector(
            self,
            s: Tensor,
            a: Tensor,
    ) -> Tensor:
        # H_u -> thetas
        raise NotImplementedError

    def _refill_noise_cache(self):
        noise_mean, noise_cov, noise_size = self._noise_params
        noises = np.random.multivariate_normal(noise_mean, noise_cov, size=noise_size)
        list(map(self._noise_cache.put, noises))
