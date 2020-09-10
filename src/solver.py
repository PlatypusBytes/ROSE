import numpy as np
from scipy.sparse.linalg import spsolve, inv
import os
import pickle
from tqdm import tqdm

from src.exceptions import *
import logging


def init(m_global, c_global, k_global, force_ini, u, v):
    r"""
    Calculation of the initial conditions - acceleration for the first time-step.

    :param m_global: Global mass matrix
    :param c_global: Global damping matrix
    :param k_global: Global stiffness matrix
    :param force_ini: Initial force
    :param u: Initial conditions - displacement
    :param v: Initial conditions - velocity

    :return a: Initial acceleration
    """

    k_part = k_global.dot(u)
    c_part = c_global.dot(v)

    # initial acceleration
    a = inv(m_global).dot(force_ini - c_part - k_part)
    return a


class Solver:
    def __init__(self):
        # define initial conditions
        self.u0 = []
        self.v0 = []

        # define variables
        self.u = []
        self.v = []
        self.a = []
        self.time = []

        return

    def initialise(self, number_equations, time):
        self.u0 = np.zeros(number_equations)
        self.v0 = np.zeros(number_equations)

        self.time = np.array(time)

        self.u = np.zeros([len(time), number_equations])
        self.v = np.zeros([len(time), number_equations])
        self.a = np.zeros([len(time), number_equations])

    def update(self, t_start_idx):
        self.u0 = self.u[t_start_idx, :]
        self.v0 = self.v[t_start_idx, :]

    def validate_input(self, F, t_start_idx, t_end_idx):
        if len(self.time) != np.shape(F)[1]:
            logging.error("Solver error: Solver time is not equal to force vector time")
            raise TimeException("Solver time is not equal to force vector time")

        diff = np.diff(self.time[t_start_idx:t_end_idx])
        if not np.all(np.isclose(diff, diff[0])):
            logging.error("Solver error: Time steps differ in current stage")
            raise TimeException("Time steps differ in current stage")


class NewmarkSolver(Solver):
    def __init__(self):
        super(NewmarkSolver, self).__init__()
        self.beta = 0.25
        self.gamma = 0.5

    def calculate(self, M, C, K, F, t_start_idx, t_end_idx):
        """
        Newmark integration scheme.
        Incremental formulation.

        :param settings: dictionary with the integration settings
        :param M: Mass matrix
        :param C: Damping matrix
        :param K: Stiffness matrix
        :param F: External force matrix
        :param t_start_idx: time index of starting time for the analysis
        :param t_end_idx: time index of end time for the analysis
        :return:
        """

        self.validate_input(F, t_start_idx, t_end_idx)
        # todo correct t_step, as it is not correct, but tests succeed
        t_step = (self.time[t_end_idx] - self.time[t_start_idx]) / (
            (t_end_idx - t_start_idx) + 1
        )

        # constants for the Newmark integration
        beta = self.beta
        gamma = self.gamma

        # initial force conditions: for computation of initial acceleration
        d_force = F[:, t_start_idx].toarray()  # add index of timestep
        d_force = d_force[:, 0]

        # initial conditions u, v, a
        u = self.u0
        v = self.v0
        a = init(M, C, K, d_force, u, v)
        # add to results initial conditions

        self.u[t_start_idx, :] = u
        self.v[t_start_idx, :] = v
        self.a[t_start_idx, :] = a

        # combined stiffness matrix
        K_till = K + C.dot(gamma / (beta * t_step)) + M.dot(1 / (beta * t_step ** 2))

        # define progress bar
        pbar = tqdm(
            total=(t_end_idx - t_start_idx),
            unit_scale=True,
            unit_divisor=1000,
            unit="steps",
        )

        # iterate for each time step
        for t in range(t_start_idx + 1, t_end_idx + 1):
            # update progress bar
            pbar.update(1)

            # updated mass
            m_part = v.dot(1 / (beta * t_step)) + a.dot(1 / (2 * beta))
            m_part = M.dot(m_part)
            # updated damping
            c_part = v.dot(gamma / beta) + a.dot(t_step * (gamma / (2 * beta) - 1))
            c_part = C.dot(c_part)

            # update external force
            d_force = F[:, t] - F[:, t - 1]

            # external force
            force_ext = d_force.toarray()[:, 0] + m_part + c_part

            # solve
            du = spsolve(K_till, force_ext)

            # velocity calculated through Newmark relation
            dv = (
                du.dot(gamma / (beta * t_step))
                - v.dot(gamma / beta)
                + a.dot(t_step * (1 - gamma / (2 * beta)))
            )

            # acceleration calculated through Newmark relation
            da = (
                du.dot(1 / (beta * t_step ** 2))
                - v.dot(1 / (beta * t_step))
                - a.dot(1 / (2 * beta))
            )

            # update variables
            u = u + du
            v = v + dv
            a = a + da

            # add to results
            self.u[t, :] = u
            self.v[t, :] = v
            self.a[t, :] = a

        # close the progress bar
        pbar.close()
        return


class StaticSolver(Solver):
    def __init__(self):
        super(StaticSolver, self).__init__()

    def calculate(self, K, F, t_start_idx, t_end_idx):
        """
        Static integration scheme.
        Incremental formulation.

        :param K: Stiffness matrix
        :param F: External force matrix
        :param t_start_idx: time index of starting time for the analysis
        :param t_end_idx: time index of end time for the analysis
        :return:
        """

        # initial conditions u
        u = self.u0
        # add to results initial conditions
        self.u[t_start_idx, :] = u
        # initial differential force
        d_force = F[:, t_start_idx]

        # validate input
        self.validate_input(F, t_start_idx, t_end_idx)

        # define progress bar
        pbar = tqdm(
            total=(t_end_idx - t_start_idx),
            unit_scale=True,
            unit_divisor=1000,
            unit="steps",
        )

        for t in range(t_start_idx + 1, t_end_idx + 1):
            # update progress bar
            pbar.update(1)

            # solve
            uu = spsolve(K, d_force)

            # update displacement
            u = np.array(u + uu)

            # update external force
            d_force = F[:, t] - F[:, t - 1]

            # add to results
            self.u[t, :] = u

        # close the progress bar
        pbar.close()
        return

    def save_data(self):
        """
        Saves the data into a binary pickle file
        """

        # construct dic structure
        data = {
            "displacement": self.u,
            "velocity": self.v,
            "acceleration": self.a,
            "time": self.time,
        }

        # dump data
        with open(os.path.join(self.output_folder, "data.pickle"), "wb") as f:
            pickle.dump(data, f)

        return
