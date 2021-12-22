import logging
from time import perf_counter
from queue import Queue

import numpy as np
from scipy.linalg import solve_discrete_are as dare

from pydeco.types import *
from pydeco.constants import *
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

        # optimal K* for logging
        self._K_star = None

        # noise
        self._noise_cache = Queue()
        self._noise_type = noise_type
        self._noise_params = None

        # linear model defines matrices - P, H, K
        # Value function matrix (n_s, n_s)
        self._P = None
        # Q-value function matrix ([n_s, n_a], [n_s, n_a])
        self._H = None
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
            policy_eval: str = PolicyEvaluation.QLEARN_RLS,
            gamma: Scalar = 1.,
            eps: Scalar = 1e-8,
            max_iter: int = 100,
            max_policy_evals: int = 80,
            max_policy_improves: int = 20,
            reset_every_n: int = 100,
            initial_state: Tensor | None = None,
            initial_policy: Tensor | None = None,
            optimal_controller: Tensor | None = None,
            alpha: Scalar = 0.01,
            **kwargs
    ):
        if self._calibrated:
            self._logger.warning('Controller has already been calibrated! Re-running the calculation.')

        self._logger.info(f'Calibrating controller.')

        # optimal policy for logging
        self._K_star = optimal_controller

        # find optimal policy
        match method:
            case TrainMethod.DARE:
                self._train_analytical_dare(
                    env,
                    gamma=gamma,
                    **kwargs
                )
            case TrainMethod.ITERATIVE:
                self._train_analytical_iterative(
                    env,
                    gamma=gamma,
                    eps=eps,
                    max_iter=max_iter,
                    **kwargs
                )
            case TrainMethod.GPI:
                self._train_gpi(
                    env=env,
                    policy_eval=policy_eval,
                    gamma=gamma,
                    eps=eps,
                    max_policy_evals=max_policy_evals,
                    max_policy_improves=max_policy_improves,
                    reset_every_n=reset_every_n,
                    initial_state=initial_state,
                    initial_policy=initial_policy,
                    alpha=alpha,
                    **kwargs
                )
            case _:
                raise ValueError(f'Method {method} not supported.')

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
        for _ in time_grid[:-1]:
            # optimal control at k
            u_k = self.act(x_k)
            us.append(u_k)

            # update state
            r_k, next_x_k = env.step(u_k)
            xs.append(next_x_k)

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
        n_evals = 'n_evals'
        eval_conv = 'e_conv'
        k_max_diff = 'K_iter_diff'
        opt_diff = 'K*_diff'

        msg = f"|{i:^7}|{n_evals:^7}|{eval_conv:^7}|{k_max_diff:^12}|{opt_diff:^12}|"

        self._logger.info(msg)

    def _log_step(
            self,
            iter: int,
    ):
        sc = self._cache[iter]
        n_evals = sc['n_evals']
        eval_conv = 'Y' if sc['eval_conv'] else 'N'
        k_max_diff = sc['K_max_diff']
        opt_diff = sc['opt_diff'] if 'opt_diff' in sc else np.nan

        msg = f'|{iter:<7}|{n_evals:<7}|{eval_conv:<7}|{k_max_diff:<12.8f}|{opt_diff:<12.8f}|'

        self._logger.info(msg)

    def _train_analytical_dare(
            self,
            env: LQ,
            gamma: Scalar,
            **kwargs
    ):
        # analytical solution requires access to the LQ model
        A_, B_, Q, R = env.get_model()

        # adjust A and B for gamma
        A = A_ * np.sqrt(gamma)
        B = B_ * np.sqrt(gamma)

        # numpy solution
        P = dare(
            np.array(A),
            np.array(B),
            np.array(Q),
            np.array(R),
        )

        # save output
        self.P = P
        self.K = - np.linalg.inv(R + B.T @ P @ B) @ B.T @ P @ A
        self._calibrated = True

    def _train_analytical_iterative(
            self,
            env: LQ,
            gamma: Scalar,
            eps: Scalar,
            max_iter: int,
            **kwargs
    ):
        # analytical solution requires access to the LQ model
        A_, B_, Q, R = env.get_model()

        # apply discounting
        A = np.sqrt(gamma) * A_
        B = np.sqrt(gamma) * B_

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

        # save output
        self.P = P
        self.K = K
        self._calibrated = True

    def _init_gpi(
            self,
            env: LQ,
            initial_state: Tensor,
            initial_policy: Tensor,
            noise_cache_size: int = 1000,
    ):
        # dimensions of the controller
        n_s = env.n_s
        n_a = env.n_a
        n_q = n_s + n_a
        p = int(n_q * (n_q + 1) / 2)

        self._n_s = n_s
        self._n_q = n_q
        self._p = p

        # noise for acting with exploration
        self._noise_params = (np.zeros((n_a,)), np.eye(n_a), (noise_cache_size,))

        # s_0
        env.reset(initial_state)

        # K_0
        self.K = initial_policy

        # weights of the FA
        self._weights = np.zeros((p, 1))

    def _train_gpi(
            self,
            env: LQ,
            policy_eval: str,
            gamma: Scalar,
            eps: Scalar,
            max_policy_evals: int,
            max_policy_improves: int,
            reset_every_n: int,
            initial_state: Tensor,
            initial_policy: Tensor,
            alpha: Scalar,
            **kwargs
    ):
        # clear previous results
        self._cache.clear()

        # initialize params for GPI
        self._init_gpi(env, initial_state, initial_policy)

        # fix policy eval function
        policy_eval_fn = None
        match policy_eval:
            case PolicyEvaluation.QLEARN_RLS:
                policy_eval_fn = self._policy_eval_qlearn_rls
            case PolicyEvaluation.QLEARN:
                policy_eval_fn = self._policy_eval_qlearn
            case _:
                raise ValueError(f'Policy Eval {policy_eval} not supported.')

        # logging
        self._log_header()

        # run GPI loop
        pi_improve_iter = 1
        pi_improve_converged = False

        while (not pi_improve_converged) and (not pi_improve_iter > max_policy_improves):
            # policy evaluation - given K, compute H
            policy_eval_fn(
                env,
                gamma=gamma,
                eps=eps,
                max_policy_evals=max_policy_evals,
                reset_every_n=reset_every_n,
                gpi_iter=pi_improve_iter,
                initial_state=initial_state,
                alpha=alpha,
            )

            # policy improvement - given H_K, re-compute K
            H_uk = self._H[self._n_s:, :self._n_s]
            H_uu = self._H[self._n_s:, self._n_s:]

            # argmax policy
            K_new = -np.linalg.inv(H_uu) @ H_uk

            # convergence
            k_diff = np.max(np.abs(K_new - self._K))
            self._save_param(pi_improve_iter, 'K_max_diff', k_diff)
            pi_improve_converged = k_diff < eps
            self._K = K_new

            # diff with K* (if given)
            if self._K_star is not None:
                opt_diff = np.max(np.abs(self._K_star - self._K))
                self._save_param(pi_improve_iter, 'opt_diff', opt_diff)

            # logging
            self._log_step(pi_improve_iter)

            # update counter
            pi_improve_iter += 1

        # save output after training
        # Note: recent P, H are saved during policy eval
        # Note: recent K is saved during policy improve
        self._calibrated = True

    def _policy_eval_qlearn_rls(
            self,
            env: LQ,
            gamma: Scalar,
            eps: Scalar,
            max_policy_evals: int,
            reset_every_n: int,
            gpi_iter: int,
            initial_state: Tensor,
            beta: Scalar = 1.,
            **kwargs
    ):
        # reset covariance matrix
        G_k = np.eye(self._p) * beta

        # s
        curr_state = env.get_state()

        # policy eval loop
        pi_eval_iter = 0
        pi_eval_converged = False

        while not pi_eval_converged and pi_eval_iter < max_policy_evals:
            # a
            curr_action = self.act(curr_state, policy_type=PolicyType.EPS_GREEDY)

            # r, s'
            curr_reward, next_state = env.step(curr_action)

            # max_a'
            next_action = self.act(next_state, policy_type=PolicyType.GREEDY)

            # features from (s,a)
            f_x = self._build_feature_vector(self._p, curr_state, curr_action)
            f_x_next = self._build_feature_vector(self._p, next_state, next_action)

            # update weights w/ RLS
            phi = f_x - gamma * f_x_next

            num = G_k @ phi * (curr_reward - phi.T @ self._weights)
            den = 1 + phi.T @ G_k @ phi
            w_adj = num / den
            self._weights += w_adj

            num2 = G_k @ phi @ phi.T @ G_k
            G_adj = num2 / den
            G_k -= G_adj

            # convergence
            w_adj_diff = np.max(np.abs(w_adj))
            pi_eval_converged = w_adj_diff < eps

            # update current state or reset
            if pi_eval_iter % reset_every_n == 0:
                curr_state = env.reset(initial_state)
            else:
                curr_state = next_state

            # update counter
            pi_eval_iter += 1

        # logging
        self._save_param(gpi_iter, 'n_evals', pi_eval_iter)
        self._save_param(gpi_iter, 'eval_conv', pi_eval_converged)

        # save output of policy eval
        self._H = self._convert_to_parameter_matrix(self._weights, self._n_q)
        self._P = self._H[:self._n_s, :self._n_s]

    def _policy_eval_qlearn(
            self,
            env: LQ,
            gamma: Scalar,
            eps: Scalar,
            max_policy_evals: int,
            reset_every_n: int,
            gpi_iter: int,
            initial_state: Tensor,
            alpha: Scalar = .01,
            beta: Scalar = 1.,
            **kwargs
    ):
        # s
        curr_state = env.get_state()

        # policy eval loop
        pi_eval_iter = 0
        pi_eval_converged = False

        while not pi_eval_converged and pi_eval_iter < max_policy_evals:
            # a
            curr_action = self.act(curr_state, policy_type=PolicyType.EPS_GREEDY)

            # r, s'
            curr_reward, next_state = env.step(curr_action)

            # max_a'
            next_action = self.act(next_state, policy_type=PolicyType.GREEDY)

            # features from (s,a)
            f_x = self._build_feature_vector(self._p, curr_state, curr_action)
            f_x_next = self._build_feature_vector(self._p, next_state, next_action)

            # update weights w/ Q-learning
            w_adj = alpha * (curr_reward + gamma * self._q_value(
                f_x_next, self._weights) - self._q_value(f_x, self._weights)) * f_x
            self._weights += w_adj

            # convergence
            w_adj_diff = np.max(np.abs(w_adj))
            pi_eval_converged = w_adj_diff < eps

            # update current state or reset
            if pi_eval_iter % reset_every_n == 0:
                curr_state = env.reset(initial_state)
            else:
                curr_state = next_state

            # update counter
            pi_eval_iter += 1

        # save output of policy eval
        self._H = self._convert_to_parameter_matrix(self._weights, self._n_q)
        self._P = self._H[:self._n_s, :self._n_s]

    def _q_value(
            self,
            features: Tensor,
            weights: Tensor,
    ):
        q = weights.T @ features
        return q.item()

    def _build_feature_vector(
            self,
            p: int,
            # s: Tensor,
            # a: Tensor,
            *args,
    ) -> Tensor:
        # s, a -> theta
        sa = np.concatenate(args).reshape((-1,))
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

    def _refill_noise_cache(self):
        noise_mean, noise_cov, noise_size = self._noise_params
        noises = np.random.multivariate_normal(noise_mean, noise_cov, size=noise_size)
        list(map(self._noise_cache.put, noises))
