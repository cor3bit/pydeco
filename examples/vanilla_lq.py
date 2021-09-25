import numpy as np
import jax.numpy as jnp
from jax.config import config

from pydeco.controller.analytic_lqr import AnalyticalLQR

PLATFORM = 'cpu'  # cpu gpu

if __name__ == '__main__':
    config.update('jax_platform_name', PLATFORM)
    np.set_printoptions(suppress=True)

    # DS params
    A = jnp.array([
        [0.3, 0.7, 0],
        [0.4, 0.5, 0.2],
        [0, 0.2, 0.4],
    ], jnp.float32)
    B = jnp.eye(3, dtype=jnp.float32)

    # reward params
    Q = jnp.eye(3, dtype=jnp.float32)
    R = jnp.eye(3, dtype=jnp.float32)

    # sim params
    x0 = jnp.array([1, 1, 1], jnp.float32)
    t0, tn, n_steps = 0., 1., 20

    # Closed-form
    print('Riccati LQR:')
    lqr = AnalyticalLQR(A, B, Q, R)
    lqr.fit(n_steps=n_steps)
    xs, us, tcost, info = lqr.simulate_trajectory(x0, t0, tn, n_steps)
    print(f'Total Cost: {tcost}')
    print(f'Control sequence: {us}')

    # DP
    # TODO

    # Q-learning
    # TODO
