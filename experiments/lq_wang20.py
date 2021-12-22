import numpy as np

from pydeco.constants import *
from pydeco.problem.lq import LQ
from pydeco.controller.lqr import LQR

if __name__ == '__main__':
    np.set_printoptions(suppress=True)
    np.random.seed(42)

    # set-up: wang20
    n_x = 4
    n_u = 4

    # env params
    A = np.array([
        [0.2, 0.4, 0.1, 0.01],
        [0.4, 0.2, 0.3, 0.1],
        [0.1, 0.3, 0.3, 0.4],
        [0.2, 0.1, 0.5, 0.3],
    ])

    B = np.eye(n_u)
    Q = -np.eye(n_x)
    R = -np.eye(n_u)

    lq = LQ(A, B, Q, R)

    # sim params
    x0 = np.full(shape=(n_x,), fill_value=0.01)

    t0, tn, n_steps = 0., 1., 20

    # Closed-form
    print('\nRiccati LQR:')
    lqr = LQR()

    lqr.train(lq, method=TrainMethod.ITERATIVE, initial_state=x0)
    print(f'P: {lqr.P}')
    print(f'K: {lqr.K}')

    xs, us, tcost = lqr.simulate_trajectory(lq, x0, t0, tn, n_steps)
    print(f'Total Cost: {tcost}')

    # Q-learning
    print('\nQ-learning LQR:')

    # K0 = np.full_like(alqr._K, fill_value=-0.1)
    K0 = -1. * np.array([
        [1, 1, 0.0004, 2],
        [1, 0.2, 1, 0.1],
        [4, 0.1, 1, 3],
        [0.2, 0.1, 0.3, 0.2],
    ])

    lqr.train(
        lq,
        method=TrainMethod.GPI,
        policy_eval=PolicyEvaluation.QLEARN_RLS,
        max_policy_evals=200,
        max_policy_improves=100,
        reset_every_n=10,
        initial_state=x0,
        initial_policy=K0,
        optimal_controller=lqr.K,
    )

    print(f'P: {lqr.P}')
    print(f'K: {lqr.K}')

    xs, us, tcost = lqr.simulate_trajectory(lq, x0, t0, tn, n_steps)
    print(f'Total Cost: {tcost}')
