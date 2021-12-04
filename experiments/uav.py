# Experiment based on alemzadeh18
# Web: https://arxiv.org/abs/1809.08745

import numpy as np

from pydeco.problem.lq import LQ
from pydeco.controller.sort.lq_analytic_agent import AnalyticalLQR


def run():
    # set-up: alemzadeh18

    n_x = 5
    n_u = 3

    # UAV params
    A = np.array(
        [
            [0.0000, 0.0000, 1.1320, 0.0000, -1.000],
            [0.0000, -0.0538, -0.1712, 0.0000, 0.0705],
            [0.0000, 0.0000, 0.0000, 1.0000, 0.0000],
            [0.0000, 0.0485, 0.0000, -0.8556, -1.013],
            [0.0000, -0.2909, 0.0000, 1.0532, -0.6859],
        ]
    )

    B = np.array(
        [
            [0.0000, 0.0000, 0.0000],
            [-0.120, 1.0000, 0.0000],
            [0.0000, 0.0000, 0.0000],
            [4.4190, 0.0000, -1.665],
            [1.5750, 0.0000, -0.0732],
        ]
    )

    Q = np.eye(n_x)
    R = np.eye(n_u)

    # sim params
    # x0 = np.full(shape=(n_x,), fill_value=0.01)

    t0, tn, n_steps = 0., 1., 20

    # Closed-form SA
    # print('\nSingle-Agent Riccati LQR:')
    # alqr = AnalyticalLQR()
    # lq = LQ(A, B, Q, R)
    #
    # alqr.train(lq, n_steps=n_steps)
    # # print(f'P: {alqr._P}')
    # print(f'K: {alqr._K}')

    # Closed-form MA
    print('\nMulti-Agent Riccati LQR:')
    alqr = AnalyticalLQR()

    A_, B_, Q_, R_ = convert_to_ma_model(A, B, Q, R, 3, [(1, 2), (2, 3)])
    lq = LQ(A_, B_, Q_, R_)

    alqr.train(lq)
    # print(f'P: {alqr._P}')
    K = alqr._K

    print(f'K: {alqr._K}')


if __name__ == '__main__':
    np.set_printoptions(suppress=True)

    np.random.seed(42)

    run()
