import numpy as np
from scipy.sparse.linalg import spsolve, inv
import os
import pickle
from tqdm import tqdm
from enum import Enum


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

        self.time = time


    def init(self, m_global, c_global, k_global, force_ini, u, v):
        r"""
        Calculation of the initial conditions - acceleration for the first time-step.

        :param m_global: Global mass matrix.
        :type m_global: np.ndarray
        :param c_global: Global damping matrix.
        :type c_global: np.ndarray
        :param k_global: Global stiffness matrix beam.
        :type k_global: np.ndarray
        :param force_ini: Initial force.
        :type force_ini: np.ndarray
        :param u: Initial conditions - displacement.
        :type u: np.ndarray
        :param v: Initial conditions - velocity.
        :type v: np.ndarray

        :return a: Initial acceleration.

        :raises ValueError:
        :raises TypeError:
        """

        k_part = k_global.dot(u)
        c_part = c_global.dot(v)

        if m_global.size == 1:
            a = (force_ini - c_part - k_part) / m_global
        else:
            a = inv(m_global).dot(force_ini - c_part - k_part)
        return a


class NewmarkSolver(Solver):
    def __init__(self):
        super(NewmarkSolver, self).__init__()
        self.beta = 0.25
        self.gamma = 0.5

    def calculate(self, M, C, K, F, t_step, t_end, t_start=0):
        """
        Newmark integration scheme.
        Incremental formulation.

        :param settings: dictionary with the integration settings
        :param M: Mass matrix
        :param C: Damping matrix
        :param K: Stiffness matrix
        :param F: External force matrix
        :param t_step: time step for the analysis
        :param t_end: total time for the analysis
        :param t_start: start time for the analysis (optional, default is zero)
        :return:
        """

        # constants for the Newmark integration
        beta = self.beta
        gamma = self.gamma

        # initial force conditions: for computation of initial acceleration
        d_force = F[:, 0].toarray()
        d_force = d_force[:, 0]

        # initial conditions u, v, a
        u = self.u0
        v = self.v0
        a = self.init(M, C, K, d_force, u, v)
        # add to results initial conditions
        self.u.append(u)
        self.v.append(v)
        self.a.append(a)

        # time
        self.time = np.linspace(t_start, t_end, int(np.ceil((t_end - t_start) / t_step)))

        # combined stiffness matrix
        K_till = K + C.dot(gamma / (beta * t_step)) + M.dot(1 / (beta * t_step ** 2))

        # define progress bar
        pbar = tqdm(total=int(len(self.time) - 1), unit_scale=True, unit_divisor=1000, unit="steps")

        # iterate for each time step
        for t in range(1, len(self.time)):
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
            dv = du.dot(gamma / (beta * t_step)) - v.dot(gamma / beta) + a.dot(t_step * (1 - gamma / (2 * beta)))
            # acceleration calculated through Newmark relation
            da = du.dot(1 / (beta * t_step ** 2)) - v.dot(1 / (beta * t_step)) - a.dot(1 / (2 * beta))

            # update variables
            u = u + du
            v = v + dv
            a = a + da

            # add to results
            self.u.append(u)
            self.v.append(v)
            self.a.append(a)

        # convert to numpy arrays
        self.u = np.array(self.u)
        self.v = np.array(self.v)
        self.a = np.array(self.a)

        # close the progress bar
        pbar.close()
        return

class StaticSolver(Solver):

    def __init__(self, number_equations):
        super(StaticSolver, self).__init__(number_equations)

    def calculate(self, K, F, t_step, t_end, t_start=0):
        """
        Static integration scheme.
        Incremental formulation.

        :param K: Stiffness matrix
        :param F: External force matrix
        :param t_step: time step for the analysis
        :param t_end: total time for the analysis
        :param t_start: start time for the analysis (optional, default is zero)
        :return:
        """

        # initial conditions u
        u = self.u0
        # add to results initial conditions
        self.u.append(u)
        # initial differential force
        d_force = F[:, 0]

        # time
        self.time = np.linspace(t_start, t_end, int(np.ceil((t_end - t_start) / t_step)))

        # define progress bar
        pbar = tqdm(total=len(self.time), unit_scale=True, unit_divisor=1000, unit="steps")

        for t in range(1, len(self.time)):
            # update progress bar
            pbar.update(1)

            # solve
            uu = spsolve(K, d_force)

            # update displacement
            u = np.array(u + uu)

            # update external force
            d_force = F[:, t] - F[:, t - 1]

            # add to results
            self.u.append(np.array(u))

        # convert to numpy arrays
        self.u = np.array(self.u)

        # close the progress bar
        pbar.close()
        return

    def save_data(self):
        """
        Saves the data into a binary pickle file
        """

        # construct dic structure
        data = {"displacement": self.u,
                "velocity": self.v,
                "acceleration": self.a,
                "time": self.time}

        # dump data
        with open(os.path.join(self.output_folder, "data.pickle"), "wb") as f:
            pickle.dump(data, f)

        return
