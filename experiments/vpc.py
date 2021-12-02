import numpy as np

from pydeco.problem.platoon import VehiclePlatoon
from pydeco.controller.lq_qlearn_agent import QlearningLQR


def run():
    # params
    n_vehicles = 5


    vp = VehiclePlatoon(n_vehicles)


if __name__ == '__main__':
    np.set_printoptions(suppress=True)

    np.random.seed(42)

    run()
