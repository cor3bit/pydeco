import numpy as np

from pydeco.problem.lq import LQ
from pydeco.controller.analytic_lqr import AnalyticalLQR
from pydeco.controller.qlearning_lqr import QlearningLQR


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

    alqr.train(lq, n_steps=n_steps)
    # print(f'P: {alqr._P}')
    K = alqr._K

    print(f'K: {alqr._K}')


def convert_to_ma_model(A, B, Q, R, n_agents, edges):
    # TODO dimension checks

    I_n = np.eye(n_agents)
    A_ = np.kron(I_n, A)
    B_ = np.kron(I_n, B)
    R_ = np.kron(I_n, R)

    # TODO hardcode
    q_m, q_n = Q.shape
    Q_ = np.zeros((n_agents * q_m, n_agents * q_n))
    # diagonal
    Q_[:5, :5] = 2 * Q
    Q_[5:10, 5:10] = 3 * Q
    Q_[10:15, 10:15] = 2 * Q

    # off-diagonal
    Q_[5:10, :5] = -Q
    Q_[5:10, 10:15] = -Q
    Q_[:5, 5:10] = -Q
    Q_[10:15, 5:10] = -Q

    # Q_ = np.kron(I_n, Q)

    return A_, B_, Q_, R_


if __name__ == '__main__':
    np.set_printoptions(suppress=True)

    np.random.seed(42)

    run()
