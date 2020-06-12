import numpy as np
from scipy.linalg import solve, inv
import os
import pickle
from tqdm import tqdm


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

    def newmark(self, settings, M, C, K, F, t_step, t_end, t_start=0):
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
        beta = settings["beta"]
        gamma = settings["gamma"]

        # initial force conditions: for computation of initial acceleration
        d_force = F[:, 0]

        # initial conditions u, v, a
        u = du = self.u0
        v = self.v0
        a = init(M, C, K, d_force, u, v)
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
            force_ext = d_force + m_part + c_part

            # solve
            du = solve(K_till, force_ext)

            # velocity calculated through Newmark relation
            dv = du.dot(gamma / (beta * t_step)) - v.dot(gamma / beta) + a.dot(t_step * (1 - gamma / (2 * beta)))
            # acceleration calculated through Newmark relation
            da = du.dot(1 / (beta * t_step ** 2)) - v.dot(1 / (beta * t_step)) - a.dot(1 / (2 * beta))

            # update variables
            u = np.array(u + du)
            v = np.array(v + dv)
            a = np.array(a + da)

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

    def static(self, K, F, t_step, t_end, t_start=0):
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
            uu = solve(K, d_force)

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
