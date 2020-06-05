import numpy as np
from scipy.linalg import solve, inv
import os
import pickle
from tqdm import tqdm


def const(beta, gamma, h):
    r"""
    Constants for the Newmark-Beta solver.

    :param beta: Parameter :math:`\beta` that weights the contribution of the initial and final acceleration to the
        change of displacement.
    :type beta: float
    :param gamma: Parameter :math:`\gamma` that weights the contribution of the initial and final acceleration to the
        change of velocity.
    :type gamma: float
    :param h: Time step.
    :type h: float

    :return a1: Parameter :math:`\alpha_1`.
    :return a2: Parameter :math:`\alpha_2`.
    :return a3: Parameter :math:`\alpha_3`.
    :return a4: Parameter :math:`\alpha_4`.
    :return a5: Parameter :math:`\alpha_5`.
    :return a6: Parameter :math:`\alpha_6`.

    :raises ValueError:
    :raises TypeError:
    """

    a1 = 1 / (beta * h ** 2)
    a2 = 1 / (beta * h)
    a3 = 1 / (2 * beta) - 1
    a4 = gamma / (beta * h)
    a5 = (gamma / beta) - 1
    a6 = (gamma / beta - 2) * h / 2

    return a1, a2, a3, a4, a5, a6


def init(m_global, c_global, k_global, force_ini, u, v):
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


class Solver:
    def __init__(self, number_equations):
        # define initial conditions
        self.u0 = np.zeros(number_equations)
        self.v0 = np.zeros(number_equations)

        # define variables
        self.u = []
        self.v = []
        self.a = []
        self.time = []

        return

    def newmark(self, settings, M, C, K, F, t_step, t_end, t_start=0, alpha=0):

        # constants for the Newmark
        a1, a2, a3, a4, a5, a6 = const(settings["beta"], settings["gamma"], t_step)

        # initial force conditions
        F_ini = F[:, 0]

        # initial conditions u, v, a
        u = self.u0
        v = self.v0
        a = init(M, C, K, F_ini, u, v)
        # add to results initial conditions
        self.u.append(u)
        self.v.append(v)
        self.a.append(a)

        # time
        self.time = np.linspace(t_start, t_end, int(np.ceil((t_end - t_start) / t_step)))

        # combined stiffness matrix
        K_till = K.dot(1 + alpha) + C.dot(a4).dot(1 + alpha) + M.dot(a1)

        # define progress bar
        pbar = tqdm(total=len(self.time), unit_scale=True, unit_divisor=1000, unit="steps")

        # iterate for each time step
        for t in range(1, len(self.time)):
            # update progress bar
            pbar.update(1)

            # updated mass
            m_part = u.dot(a1) + v.dot(a2) + a.dot(a3)
            m_part = M.dot(m_part)
            # updated damping
            c_part = u.dot(a4) + v.dot(a5) + a.dot(a6)
            c_part = C.dot(c_part).dot(1 + alpha)

            # external force
            force = F[:, t]
            force_ext = force + m_part + c_part

            # solve
            uu = solve(K_till, force_ext)

            # velocity calculated through Newmark relation
            vv = (uu - u).dot(a4) - v.dot(a5) - a.dot(a6)
            # acceleration calculated through Newmark relation
            aa = (uu - u).dot(a1) - v.dot(a2) - a.dot(a3)

            # add to results
            self.u.append(uu)
            self.v.append(vv)
            self.a.append(aa)

            # update variables
            u = uu
            a = aa
            v = vv

        # convert to numpy arrays
        self.u = np.array(self.u)
        self.v = np.array(self.v)
        self.a = np.array(self.a)

        # close the progress bar
        pbar.close()
        return

    def static(self, K, F, t_step, t_end, t_start=0):

        # initial conditions u
        u = self.u0
        # add to results initial conditions
        self.u.append(u)

        # time
        self.time = np.linspace(t_start, t_end, int(np.ceil((t_end - t_start) / t_step)))

        # define progress bar
        pbar = tqdm(total=len(self.time), unit_scale=True, unit_divisor=1000, unit="steps")

        for t in range(len(self.time)):
            # update progress bar
            pbar.update(1)

            # external force
            force = F[:, t]

            # solve
            uu = solve(K, force)

            # add to results
            self.u.append(uu)

        # convert to numpy arrays
        self.u = np.array(self.u)

        # close the progress bar
        pbar.close()
        return

    def save_data(self):

        # construct dic structure
        data = {"displacement": self.u,
                "velocity": self.v,
                "acceleration": self.a,
                "time": self.time}

        # dump data
        with open(os.path.join(self.output_folder, "data.pickle"), "wb") as f:
            pickle.dump(data, f)
        return
