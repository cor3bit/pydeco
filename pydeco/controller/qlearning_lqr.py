import logging
from time import perf_counter
from queue import Queue

import numpy as np

from pydeco.types import *
from pydeco.constants import PolicyType
from pydeco.controller.lqr import LQR
from pydeco.problem.lq import LQ


class QlearningLQR(LQR):
    def __init__(
            self,
            verbose: bool = True,
    ):
        super().__init__('qlearning_lqr', verbose)

        # Q-function matrix (n_x+n_u, n_x+n_u)
        self._H = None

        self._noise_cache = Queue()

    def act(
            self,
            x: Tensor,
            noise: Tensor = None,
            policy_type: PolicyType = PolicyType.EPS_GREEDY,
    ) -> Tensor:
        if policy_type == PolicyType.EPS_GREEDY:
            assert noise is not None
            return self._K @ x + noise
        elif policy_type == PolicyType.GREEDY:
            return self._K @ x
        else:
            raise ValueError(f'Policy {policy_type} not supported.')

    def act_q(
            self,
            x: Tensor,
            policy_type: PolicyType = PolicyType.EPS_GREEDY,
    ) -> Tuple[Tensor, Tensor]:
        raise NotImplementedError

    def state_value(
            self,
            s: Tensor,
    ) -> float:
        return s.T @ self._P @ s

    def state_action_value(
            self,
            s: Tensor,
            a: Tensor,
    ) -> float:
        sa = np.concatenate([s, a])
        return sa.T @ self._H @ sa

    def train(
            self,
            env: LQ,
            K0: Tensor,
            x0: Tensor,
    ):
        if self._calibrated:
            self._logger.warning('Controller has already been calibrated! Re-running the calculation.')

        self._logger.info(f'Calibrating controller.')

        # s
        x = env.reset(x0)

        # clear previous results
        self._cache.clear()

        # initial stabilizing controller
        self._K = K0

        n_max_pi_iters = 10
        n_policy_evals = 45

        n_x = env.n_x
        n_u = env.n_u
        noise_mean = np.zeros((n_u,))
        noise_cov = np.eye(n_u)

        n_q = n_x + n_u
        p = int(n_q * (n_q + 1) / 2)

        gamma = 1.

        beta = 1
        G_k = np.eye(p) * beta
        theta = np.full((p, 1), fill_value=0.1)

        for k in range(n_max_pi_iters):
            # compute exploration noise in advance
            noises = np.random.multivariate_normal(noise_mean, noise_cov, size=(n_policy_evals * 2,))
            noise_count = 0

            for i in range(n_policy_evals):
                # a
                # TODO optimize reshape step
                noise1 = noises[noise_count, :].reshape((n_x, 1))
                u_k = self.act(x, noise1, policy_type=PolicyType.EPS_GREEDY)

                # r, s'
                r_k, next_x_k = env.step(u_k)

                # max_a'
                # TODO optimize reshape step
                noise2 = noises[noise_count + 1, :].reshape((n_x, 1))
                next_u_k = self.act(next_x_k, noise2, policy_type=PolicyType.GREEDY)

                # features from (s,a)
                f_x = self._build_feature_vector(x, u_k, p)
                f_x_next = self._build_feature_vector(next_x_k, next_u_k, p)

                # update params theta w/ RLS
                phi = f_x - gamma * f_x_next

                num = G_k @ phi * (r_k - phi.T @ theta)
                den = 1 + phi.T @ G_k @ phi
                theta_adj = num / den
                theta += theta_adj

                num2 = G_k @ phi @ phi.T @ G_k
                G_adj = num2 / den
                G_k -= G_adj

                # update state
                x = next_x_k
                noise_count += 2

            # policy improvement
            H_k = self._convert_to_parameter_matrix(theta, n_q)
            H_uk = H_k[n_x:, :n_x]
            H_uu = H_k[n_x:, n_x:]

            # argmax policy
            self._K = -np.linalg.inv(H_uu) @ H_uk

        self._H = H_k
        self._P = H_k[:n_x, :n_x]

        self._calibrated = True


    def train2(
            self,
            env: LQ,
            K0: Tensor,
            x0: Tensor,
    ):
        if self._calibrated:
            self._logger.warning('Controller has already been calibrated! Re-running the calculation.')

        self._logger.info(f'Calibrating controller.')

        # s
        x = env.reset(x0)

        # clear previous results
        self._cache.clear()

        # initial stabilizing controller
        self._K = K0

        n_policy_evals = 2000

        n_x = env.n_x
        n_u = env.n_u
        noise_mean = np.zeros((n_u,))
        noise_cov = np.eye(n_u)

        n_q = n_x + n_u
        p = int(n_q * (n_q + 1) / 2)

        gamma = 1.

        alpha = 0.005

        theta = np.full((p, 1), fill_value=-0.1)

        # compute exploration noise in advance
        noises = np.random.multivariate_normal(noise_mean, noise_cov, size=(n_policy_evals * 2,))
        noise_count = 0

        for i in range(n_policy_evals):
            # a
            # TODO optimize reshape step
            noise1 = noises[noise_count, :].reshape((n_x, 1))
            u_k = self.act(x, noise1, policy_type=PolicyType.EPS_GREEDY)

            # r, s'
            r_k, next_x_k = env.step(u_k)

            # max_a'
            # TODO optimize reshape step
            noise2 = noises[noise_count + 1, :].reshape((n_x, 1))
            next_u_k = self.act(next_x_k, noise2, policy_type=PolicyType.GREEDY)

            # features from (s,a)
            f_x = self._build_feature_vector(x, u_k, p)
            f_x_next = self._build_feature_vector(next_x_k, next_u_k, p)

            # update params theta w/ RLS
            # phi = f_x - gamma * f_x_next
            # sss = self._q_value(f_x_next, theta, gamma)
            theta_adj = alpha * (r_k + self._q_value(f_x_next, theta, gamma) - self._q_value(f_x, theta)) * f_x
            theta += theta_adj

            # update state
            x = next_x_k
            noise_count += 2

            # policy improvement
            H_k = self._convert_to_parameter_matrix(theta, n_q)
            H_uk = H_k[n_x:, :n_x]
            H_uu = H_k[n_x:, n_x:]

            # argmax policy
            self._K = -np.linalg.inv(H_uu) @ H_uk

        self._H = H_k
        self._P = H_k[:n_x, :n_x]

        self._calibrated = True


    def _q_value(self, x, theta, gamma=1.):
        return (gamma*x.T @ theta)[0][0]

    def _build_feature_vector(
            self,
            s: Tensor,
            a: Tensor,
            p: int,
    ) -> Tensor:
        # s, a -> theta
        sa = np.concatenate((s, a)).reshape((-1,))
        n = sa.shape[0]

        x = np.empty((p, 1))

        # TODO optimize building quadratic features
        c = 0
        for i in range(n):
            for j in range(i, n):
                x[c, 0] = sa[i] * sa[j]
                c += 1

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
