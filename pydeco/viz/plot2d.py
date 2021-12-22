import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pydeco.types import *


def plot_evolution(xs: Tensor, ts: Tensor, indx: Sequence[int], name: str):
    # chart data
    for ind in indx:
        x = xs[:, ind]
        sns.lineplot(x=ts, y=x, label=f'$x_{1}$')

    # chart plotting
    # plt.title(f'{scheme}')
    plt.xlabel('$t$')
    plt.ylabel('$s^{i}$ position')
    # plt.legend()
    plt.show()


def plot_errors(errors, scheme):
    e = np.array(errors)
    plt.plot(e[:, 1], 'o', color='red')
    plt.title(f'{scheme}')
    plt.xlabel('h')
    plt.ylabel('E')
    plt.show()

    def plot_shooting(self):
        assert self._solver_caches is not None
        assert len(self._solver_caches) == 1
        cache = self._solver_caches[0]

        # extract meta data from cache
        xf = cache[0]['xf']
        if xf[0] == 2.:
            scenario = 'a'
        elif xf[0] == 1.:
            scenario = 'c'
        else:
            scenario = 'b'
        n_steps = cache[0]['n_steps']

        # 1) plot main time plot k vs. |g(x)|, |dLagrangian|, alpha
        fig, ax = plt.subplots()
        ax.set_yscale('log')

        times = np.fromiter(cache.keys(), dtype=float)

        vals = np.array([(
            step_cache['max_c_eq'],
            step_cache['max_grad_Lagrangian'],
            step_cache['loss'],
            step_cache['alpha'] if 'alpha' in step_cache else None,
        ) for step_cache in cache.values()])

        ax.plot(times, vals[:, 0], '-', label=r'$\left\Vert g\right\Vert _{\infty}$')
        ax.plot(times, vals[:, 1], '-', label=r'$\left\Vert \nabla_{x}\mathcal{L}\right\Vert _{\infty}$')
        ax.plot(times[:-1], vals[:-1, 3], '-', label=r'$\alpha$')
        # ax.plot(times, vals[:, 2], '-', label=r'$Loss$')

        plt.legend()
        plt.xlabel('k')
        plt.show()

        if self._save_dir is not None:
            fig.savefig(self._save_dir + f'{self._solver_type}_{scenario}_n{n_steps}_main.png',
                        bbox_inches='tight')

        # 2) plot w_k every N iterations
        t0, tf = 0., 2.
        u_min, u_max = -20., 20.
        N = self._n
        for k, step_cache in cache.items():
            if k % N == 0:
                fig, ax = plt.subplots()

                w_k = np.array(step_cache['u'])
                w_k_dims = w_k.shape[0]

                times = np.linspace(t0, tf, w_k_dims + 1)
                w_k_steps = np.concatenate((w_k, [w_k[-1]]))

                ax.plot(times, np.array([u_min for i in range(len(times))]), color='red', ls='--')
                ax.plot(times, np.array([u_max for i in range(len(times))]), color='red', ls='--')
                ax.step(times, w_k_steps, where='post')

                plt.title(f'SQP iteration: {k}')
                # plt.legend()
                plt.xlabel('$t$')
                plt.ylabel('$u$')
                plt.show()

                if self._save_dir is not None:
                    fig.savefig(self._save_dir + f'{self._solver_type}_{scenario}_n{n_steps}_i{k}.png',
                                bbox_inches='tight')

        # 3) plot of x_k evolution
        x1 = self._xs[:, 0]
        x2 = self._xs[:, 1]
        x3 = self._xs[:, 2]
        x4 = self._xs[:, 3]

        t0 = 0.
        tf = 2.
        ts = np.linspace(t0, tf, num=x1.shape[0])

        fig, axs = plt.subplots(2, 2, sharex=True)

        fig.set_figwidth(13)
        fig.set_figheight(6)

        axs[0, 0].plot(ts, x1)
        # axs[0, 0].set_xlabel('t')
        axs[0, 0].set_ylabel('w')

        axs[0, 1].plot(ts, x2)
        # axs[0, 1].set_xlabel('t')
        axs[0, 1].set_ylabel(r'$\theta$')

        axs[1, 0].plot(ts, x3)
        axs[1, 0].set_xlabel('t')
        axs[1, 0].set_ylabel('v')

        axs[1, 1].plot(ts, x4)
        axs[1, 1].set_xlabel('t')
        axs[1, 1].set_ylabel(r'$\omega$')

        # plt.subplots_adjust(wspace=0.35, hspace=0.1)
        plt.show()

        if self._save_dir is not None:
            fig.savefig(self._save_dir + f'{self._solver_type}_{scenario}_n{n_steps}_states.png', bbox_inches='tight')

        # 4) create gif of w_k evolution
        # TODO
