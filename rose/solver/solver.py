import numpy as np
from scipy.sparse.linalg import spsolve, inv
from scipy.sparse import csc_matrix, diags, issparse
import os
import pickle
from tqdm import tqdm

from rose.base.exceptions import *
import logging
import copy


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
    # a=np.zeros((len(k_part)))
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

        # load function
        self.load_func = None
        self.stiffness_func = None
        self.mass_func = None
        self.damping_func = None

        return

    def initialise(self, number_equations, time):
        self.u0 = np.zeros((number_equations))
        self.v0 = np.zeros((number_equations))

        self.time = np.array(time)

        self.u = np.zeros((len(time), number_equations))
        self.v = np.zeros((len(time), number_equations))
        self.a = np.zeros((len(time), number_equations))

    def update(self, t_start_idx):
        self.u0 = self.u[t_start_idx, :]
        self.v0 = self.v[t_start_idx, :]

    def validate_input(self, F, t_start_idx, t_end_idx):
        if len(self.time) != np.shape(F)[1]:
            logging.error("Solver error: Solver time is not equal to force vector time")
            raise TimeException("Solver time is not equal to force vector time")

        diff = np.diff(self.time[t_start_idx:t_end_idx])
        if diff.size >0:
            if not np.all(np.isclose(diff, diff[0])):
                logging.error("Solver error: Time steps differ in current stage")
                raise TimeException("Time steps differ in current stage")


class ZhaiSolver(Solver):
    def __init__(self):
        super(ZhaiSolver, self).__init__()

        self.psi = 0.5
        self.phi = 0.5
        self.beta = 1/4
        self.gamma = 1/2

        self.number_equations = None

    def calculate_initial_values(self, M, C, K, F, u0, v0):
        """
        Calculate inverse mass matrix and initial acceleration

        :param M: global mass matrix
        :param C: global damping matrix
        :param K: global stiffness matrix
        :param F: global force vector at current time step
        :param u0: initial displacement
        :param v0:  initial velocity
        :return:
        """
        inv_M = inv(M)
        a0 = self.evaluate_acceleration(inv_M, C, K, F, u0, v0)
        return inv_M, a0


    def calculate_force(self, u, F, t):
        """
        Calculate external force if a load function is given. If no load function is given, force is taken from current
        load vector
        :param u:
        :param F:
        :param t:
        :return:
        """
        if self.load_func is not None:
            force = self.load_func(u, t)
            if issparse(force):
                force = force.toarray()[:,0]
        else:
            force = F[:, t]
            force = force.toarray()[:, 0]
        return force

    def prediction(self, u, v, a, a_old, dt, is_initial):
        """
        Perform prediction for displacement and acceleration

        :param u: displacement
        :param v: velocity
        :param a: acceleration
        :param a_old: acceleration at previous time step
        :param dt: delta time
        :param is_initial: bool to indicate current iteration is the initial iteration
        :return:
        """

        # set Zhai factors
        if is_initial:
            psi = phi = 0
        else:
            psi = self.psi
            phi = self.phi

        # predict displacement and velocity
        u_new = u + v * dt + (1/2 + psi) * a * dt ** 2 - psi * a_old * dt**2
        v_new = v + (1 + phi) * a * dt - phi * a_old * dt
        return u_new, v_new

    def evaluate_acceleration(self, inv_M, C, K, F, u, v):
        """
        Calculate acceleration

        :param inv_M: inverse global mass matrix
        :param C: global damping matrix
        :param K: Global stiffness matrix
        :param F: Force vector at current time step
        :param u: displacement
        :param v: velocity
        :return:
        """
        a_new = inv_M.dot(F - K.dot(u) - C.dot(v))
        return a_new

    def newmark_iteration(self, u, v, a, a_new, dt):
        """
        Perform Wewmark iteration as corrector for displacement and velocity

        :param u: displacement
        :param v: velocity
        :param a: acceleration
        :param a_new: predicted acceleration
        :param dt: delta time
        :return:
        """
        u_new = u + v * dt + (1/2 - self.beta) * a * dt ** 2 + self.beta * a_new * dt ** 2
        v_new = v + (1-self.gamma) * a * dt + self.gamma * a_new * dt

        return u_new, v_new

    def calculate(self, M, C, K, F, t_start_idx, t_end_idx):
        """
        Perform calculation with Zhai solver [Zhai 1996]

        :param M: Mass matrix
        :param C: Damping matrix
        :param K: Stiffness matrix
        :param F: External force matrix
        :param t_start_idx: time index of starting time for the analysis
        :param t_end_idx: time index of end time for the analysis
        :return:
        """

        self.validate_input(F, t_start_idx, t_end_idx)

        t_step = (self.time[t_end_idx] - self.time[t_start_idx]) / (
            (t_end_idx - t_start_idx))

        # initial force conditions: for computation of initial acceleration
        # force = F[:, t_start_idx].toarray()[0]
        force = F[:, t_start_idx].toarray()
        force = force[:,0]


        u = self.u0
        v = self.v0
        inv_M, a = self.calculate_initial_values(M, C, K, force, u, v)

        self.u[t_start_idx, :] = u
        self.v[t_start_idx, :] = v
        self.a[t_start_idx, :] = a

        a_old = np.zeros(self.number_equations)

        # define progress bar
        pbar = tqdm(
            total=(t_end_idx - t_start_idx),
            unit_scale=True,
            unit_divisor=1000,
            unit="steps",
        )

        is_initial = True
        for t in range(t_start_idx + 1, t_end_idx + 1):

            # check if current timestep is the initial timestep
            if t > 1:
                is_initial = False

            # Predict displacement and velocity
            u_new, v_new = self.prediction(u, v, a, a_old, t_step, is_initial)

            # Calculate predicted external force vector
            force = self.calculate_force(u_new, F, t)

            # Calculate predicted acceleration
            a_new = self.evaluate_acceleration(inv_M, C, K, force, u_new, v_new)

            # Correct displacement and velocity
            u_new, v_new = self.newmark_iteration(u, v, a, a_new, t_step)

            # Calculate corrected force vector
            force = self.calculate_force(u_new, F, t)

            # Calculate corrected acceleration
            a_new = self.evaluate_acceleration(inv_M, C, K, force, u_new, v_new)

            # add to results
            self.u[t, :] = u_new
            self.v[t, :] = v_new
            self.a[t, :] = a_new

            # set vectors for next time step
            u = copy.deepcopy(u_new)
            v = copy.deepcopy(v_new)

            a_old = copy.deepcopy(a)
            a = copy.deepcopy(a_new)

        # close the progress bar
        pbar.close()
        return


class NewmarkSolver(Solver):
    def __init__(self):
        super(NewmarkSolver, self).__init__()
        self.beta = 0.25
        self.gamma = 0.5

    def update_force(self, u, F_previous, t):
        force = self.load_func(u, t)
        if issparse(force):
            force = force.toarray()[:, 0]
        d_force = force - F_previous
        F_total = copy.deepcopy(force)

        return d_force, F_total

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
            (t_end_idx - t_start_idx))

        # constants for the Newmark integration
        beta = self.beta
        gamma = self.gamma

        # initial force conditions: for computation of initial acceleration
        d_force = F[:, t_start_idx].toarray()
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

        # initialise Force from load function
        if self.load_func is not None:
            F_previous = self.load_func(u, t_start_idx)
            if issparse(F_previous):
                F_previous = F_previous.toarray()[:, 0]
        else:
            F_previous = F[:, t_start_idx]

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
            if self.load_func is not None:
                d_force, F_previous = self.update_force(u, F_previous, t)
            else:
                d_force = F[:, t] - F_previous
                d_force = d_force.toarray()[:, 0]
                F_previous = F[:, t]

            # external force
            force_ext = d_force + m_part + c_part

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
            u = u + uu

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
