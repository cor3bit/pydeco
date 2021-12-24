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
    x0 = np.ones(shape=(n_s,))

    # Closed-form
    # print('\nRiccati LQR:')
    lqr = LQR()

    lqr.train(
        lq,
        method=TrainMethod.ITERATIVE,
        initial_state=x0,
    )

    K_star = np.array(lqr.K)
    # print(f'P: {lqr.P}')
    # print(f'K: {lqr.K}')

    # ------------ Q-learning RLS ------------
    # print('\nQ-learning LQR:')

    K0 = np.full_like(lqr.K, fill_value=-0.1)

    # lqr.train(
    #     lq,
    #     method=TrainMethod.GPI,
    #     policy_eval=PolicyEvaluation.QLEARN_RLS,
    #     max_policy_evals=1000,
    #     max_policy_improves=30,
    #     reset_every_n=10,
    #     initial_state=x0,
    #     initial_policy=K0,
    #     optimal_controller=K_star,
    # )

    # ------------ Q-learning ------------
    # lqr.train(
    #     lq,
    #     method=TrainMethod.GPI,
    #     policy_eval=PolicyEvaluation.QLEARN,
    #     alpha=0.01,
    #     max_policy_evals=5000,
    #     max_policy_improves=30,
    #     reset_every_n=1,
    #     initial_state=x0,
    #     initial_policy=K0,
    #     optimal_controller=K_star,
    # )

    # ------------ Q-learning Gauss-Newton ------------
    lqr.train(
        lq,
        method=TrainMethod.GPI,
        policy_eval=PolicyEvaluation.QLEARN_GN,
        alpha=.05,
        max_policy_evals=1000,
        max_policy_improves=100,
        reset_every_n=5,
        initial_state=x0,
        initial_policy=K0,
        optimal_controller=K_star,
    )


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
