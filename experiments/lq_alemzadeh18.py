import numpy as np

from pydeco.constants import *
from pydeco.problem.lq import LQ
from pydeco.controller.lqr import LQR


def run_experiment():
    np.set_printoptions(suppress=True)
    np.random.seed(42)

    # env params
    n_s = 5
    n_a = 3
    A, B, Q, R = problem_setup()

    lq = LQ(A, B, Q, R)

    # sim params
    x0 = np.zeros(shape=(n_s,))

    # Closed-form
    print('\nRiccati LQR:')
    lqr = LQR()

    lqr.train(
        lq,
        method=TrainMethod.ANALYTICAL,
        initial_state=x0,
    )
    # print(f'P: {lqr.P}')
    print(f'K: {lqr.K}')

    # Q-learning
    print('\nQ-learning LQR:')

    K0 = np.full_like(lqr.K, fill_value=-0.01)

    lqr.train(
        lq,
        method=TrainMethod.QLEARN_LS,
        initial_state=x0,
        initial_policy=K0,
        max_policy_evals=200,
        max_policy_improves=50,
    )
    # print(f'P: {lqr.P}')
    print(f'K: {lqr.K}')


def problem_setup():
    # UAV example
    n_s = 5
    n_a = 3

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

    Q = -np.eye(n_s)
    R = -np.eye(n_a)

    return A, B, Q, R


if __name__ == '__main__':
    run_experiment()
