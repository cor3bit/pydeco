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

    # Closed-form
    print('\nRiccati LQR:')
    lqr = LQR()

    lqr.train(lq, method=TrainMethod.ANALYTICAL, initial_state=x0)
    print(f'P: {lqr.P}')
    print(f'K: {lqr.K}')

    xs, us, tcost = lqr.simulate_trajectory(lq, x0, t0, tn, n_steps)
    print(f'Total Cost: {tcost}')

    # Q-learning
    print('\nQ-learning LQR:')

    # initial stabilizing controller
    K0 = np.full_like(lqr.K, fill_value=-0.1)

    lqr.train(lq, method=TrainMethod.QLEARN_LS,
              initial_state=x0, initial_policy=K0)
    print(f'P: {lqr.P}')
    print(f'K: {lqr.K}')

    xs, us, tcost = lqr.simulate_trajectory(lq, x0, t0, tn, n_steps)
    print(f'Total Cost: {tcost}')
