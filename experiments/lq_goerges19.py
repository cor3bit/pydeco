import numpy as np

from pydeco.constants import *
from pydeco.problem.lq import LQ
from pydeco.controller.lqr import LQR

if __name__ == '__main__':
    np.set_printoptions(suppress=True)

    np.random.seed(42)

    # env params
    A = np.array([
        [0.3, 0.7, 0],
        [0.4, 0.5, 0.2],
        [0, 0.2, 0.4],
    ])
    B = np.eye(3)
    Q = -np.eye(3)
    R = -np.eye(3)

    lq = LQ(A, B, Q, R)

    # sim params
    x0 = np.ones(shape=(3,))
    t0, tn, n_steps = 0., 1., 20

    # ------------ Closed-form ------------
    # print('\nRiccati LQR:')
    lqr = LQR()

    lqr.train(
        lq,
        method=TrainMethod.ITERATIVE,
        initial_state=x0,
    )
    # print(f'P: {lqr.P}')
    # print(f'K: {lqr.K}')

    # xs, us, tcost = lqr.simulate_trajectory(lq, x0, t0, tn, n_steps)
    # print(f'Total Cost: {tcost}')

    # ------------ Q-learning RLS ------------
    # print('\nQ-learning LQR:')

    # initial stabilizing controller
    K0 = np.full_like(lqr.K, fill_value=-0.1)

    lqr.train(
        lq,
        method=TrainMethod.GPI,
        policy_eval=PolicyEvaluation.QLEARN_RLS,
        max_policy_evals=5000,
        max_policy_improves=20,
        reset_every_n=5000,
        initial_state=x0,
        initial_policy=K0,
        optimal_controller=lqr.K,
    )
    # print(f'P: {lqr.P}')
    # print(f'K: {lqr.K}')
    #
    # xs, us, tcost = lqr.simulate_trajectory(lq, x0, t0, tn, n_steps)
    # print(f'Total Cost: {tcost}')

    # ------------ Q-learning ------------
    lqr.train(
        lq,
        method=TrainMethod.GPI,
        policy_eval=PolicyEvaluation.QLEARN,
        alpha=0.01,
        max_policy_evals=5000,
        max_policy_improves=30,
        reset_every_n=5000,
        initial_state=x0,
        initial_policy=K0,
        optimal_controller=lqr.K,
    )

    # ------------ Q-learning Gauss-Newton ------------
    lqr.train(
        lq,
        method=TrainMethod.GPI,
        policy_eval=PolicyEvaluation.QLEARN_GN,
        alpha=1.0,
        max_policy_evals=5000,
        max_policy_improves=30,
        reset_every_n=5000,
        initial_state=x0,
        initial_policy=K0,
        optimal_controller=lqr.K,
    )
