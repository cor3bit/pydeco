import numpy as np

from pydeco.problem.lq import LQ
from pydeco.controller.analytic_lqr import AnalyticalLQR
from pydeco.controller.qlearning_lqr import QlearningLQR

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
    Q = np.eye(3)
    R = np.eye(3)

    lq = LQ(A, B, Q, R)

    # sim params
    x0 = np.ones(shape=(3,))
    t0, tn, n_steps = 0., 1., 20

    # Closed-form
    print('\nRiccati LQR:')
    alqr = AnalyticalLQR()

    alqr.train(lq, n_steps=n_steps)
    print(f'P: {alqr._P}')
    print(f'K: {alqr._K}')

    # xs, us, tcost = alqr.simulate_trajectory(lq, x0, t0, tn, n_steps)
    # print(f'Total Cost: {tcost}')

    # Q-learning
    print('\nQ-learning LQR:')
    qlqr = QlearningLQR()

    # TODO initial stabilizing controller
    K0 = np.full_like(alqr._K, fill_value=-0.1)
    # K0 = np.copy(alqr._K)

    qlqr.train2(lq, K0=K0, x0=x0)
    print(f'P: {qlqr._P}')
    print(f'K: {qlqr._K}')

    # xs, us, tcost = qlqr.simulate_trajectory(lq, x0, t0, tn, n_steps)
    # print(f'Total Cost: {tcost}')
