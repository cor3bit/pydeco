import jax.numpy as jnp
from jax.config import config

from pydeco.controller.analytic_lqr import AnalyticalLQR

PLATFORM = 'cpu'  # cpu gpu

if __name__ == '__main__':
    config.update('jax_platform_name', PLATFORM)

    # params
    A = jnp.array([[0, 1], [0, 0]], jnp.float32)
    B = jnp.array([[0], [1]], jnp.float32)
    Q = jnp.eye(2, dtype=jnp.float32)
    R = jnp.eye(1, dtype=jnp.float32)

    # problem
    lqr = AnalyticalLQR(A, B, Q, R)

    # solution
    # TODO or terminal episode?
    x0 = jnp.array([1, 0], jnp.float32)
    t0, tn, n_steps = 0., 1., 20
    us, loss, info = lqr.solve(x0, t0, tn, n_steps)

    print(us)
    print(loss)
