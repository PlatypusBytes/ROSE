import numpy as np
from scipy.sparse.linalg import spsolve, inv
from scipy.sparse import issparse
import os
import pickle
from tqdm import tqdm

from rose.model.exceptions import *
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
    """
    Solver class. This class forms the base for each solver.

    :Attributes:

        - :self.u0:                 initial displacement vector
        - :self.v0:                 initial velocity vector
        - :self.u:                  full displacement matrix [ndof, number of time steps]
        - :self.v:                  full velocity matrix [ndof, number of time steps]
        - :self.a:                  full acceleration matrix [ndof, number of time steps]
        - :self.f:                  full force matrix [ndof, number of time steps]
        - :self.time:               time discretisation
        - :self.load_func:          optional custom load function to alter external force during calculation
        - :self.stiffness_func:     optional custom stiffness function to alter stiffness matrix during calculation
        - :self.mass_func:          optional custom mass function to alter mass matrix during calculation
        - :self.damping_func:       optional custom damping function to alter damping matrix during calculation
        - :self.output_interval:    number of time steps interval in which output results are stored
        - :self.u_out:              output displacement stored at self.output_interval
        - :self.v_out:              output velocities stored at self.output_interval
        - :self.a_out:              output accelerations stored at self.output_interval
        - :self.time_out:           output time discretisation stored at self.output_interval
        - :self.number_equations:   number of equations to be solved
    """

    def __init__(self):
        # define initial conditions
        self.u0 = []
        self.v0 = []

        # define variables
        self.u = []
        self.v = []
        self.a = []
        self.f = []
        self.time = []

        # load function
        self.load_func = None
        self.stiffness_func = None
        self.mass_func = None
        self.damping_func = None

        self.output_interval = 10
        self.u_out = []
        self.v_out = []
        self.a_out = []
        self.time_out = []

        self.number_equations = None

    def initialise(self, number_equations, time):
        """
        Initialises the solver before the calculation starts

        :param number_equations: number of equations to be solved
        :param time: time discretisation
        :return:
        """
        print("Initialising solver")
        self.u0 = np.zeros(number_equations)
        self.v0 = np.zeros(number_equations)

        self.time = np.array(time)

        self.u = np.zeros((len(time), number_equations))
        self.v = np.zeros((len(time), number_equations))
        self.a = np.zeros((len(time), number_equations))
        self.f = np.zeros((len(time), number_equations))

        self.number_equations = number_equations

    def update(self, t_start_idx):
        """
        Updates the solver on a certain stage. Initial conditions are retrieved from previously calculated values for
        displacements and velocities.

        :param t_start_idx: start time index of current stage
        :return:
        """
        self.u0 = self.u[t_start_idx, :]
        self.v0 = self.v[t_start_idx, :]

    def finalise(self):
        """
        Finalises the solver. Displacements, velocities, accelerations and time are stored at a certain interval.
        :return:
        """
        self.u_out = self.u[0::self.output_interval,:]
        self.v_out = self.v[0::self.output_interval,:]
        self.a_out = self.a[0::self.output_interval,:]

        self.time_out = self.time[0::self.output_interval]

    def validate_input(self, F, t_start_idx, t_end_idx):
        """
        Validates solver input at current stage. It is checked if the external force vector shape corresponds with the
        time discretisation. Furthermore, it is checked if all time steps in the current stage are equal.

        :param F:           External force vector.
        :param t_start_idx: first time index of current stage
        :param t_end_idx:   last time index of current stage
        :return:
        """

        # validate shape external force vector
        if len(self.time) != np.shape(F)[1]:
            logging.error("Solver error: Solver time is not equal to force vector time")
            raise TimeException("Solver time is not equal to force vector time")

        # validate time step size
        diff = np.diff(self.time[t_start_idx:t_end_idx])
        if diff.size >0:
            if not np.all(np.isclose(diff, diff[0])):
                logging.error("Solver error: Time steps differ in current stage")
                raise TimeException("Time steps differ in current stage")


class ZhaiSolver(Solver):
    """
    Zhai Solver class. This class contains the explicit solver according to [Zhai 1996]. This class bases from
    :class:`~rose.model.solver.Solver`.

    :Attributes:

       - :self.psi:      Zhai numerical stability parameter
       - :self.phi:      Zhai numerical stability parameter
       - :self.beta:     Newmark numerical stability parameter
       - :self.gamma:    Newmark numerical stability parameter
    """
    def __init__(self):
        super(ZhaiSolver, self).__init__()

        self.psi = 0.5
        self.phi = 0.5
        self.beta = 1/4
        self.gamma = 1/2

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
        inv_M = inv(M).tocsr()

        a0 = self.evaluate_acceleration(inv_M, C, K, F, u0, v0)
        return inv_M, a0

    def calculate_force(self, u, F, t):
        """
        Calculate external force if a load function is given. If no load function is given, force is taken from current
        load vector

        :param u: displacement at time t
        :param F: External force matrix
        :param t: current time step
        :return:
        """
        if self.load_func is not None:
            force = self.load_func(u, t)
            if issparse(force):
                force = force.toarray()[:, 0]
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
        :param dt: time step size
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

    @staticmethod
    def evaluate_acceleration(inv_M, C, K, F, u, v):
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
        Perform Newmark iteration as corrector for displacement and velocity

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
        Perform calculation with the explicit Zhai solver [Zhai 1996]

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
        force = F[:, t_start_idx].toarray()
        force = force[:,0]

        # get initial displacement, velocity, acceleration and inverse mass matrix
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
            # update progress bar
            pbar.update(1)

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
            u = np.copy(u_new)
            v = np.copy(v_new)

            a_old = np.copy(a)
            a = np.copy(a_new)

        # close the progress bar
        pbar.close()


class NewmarkSolver(Solver):
    """
    Newmark Solver class. This class contains the implicit incremental Newmark solver. This class bases from
    :class:`~rose.model.solver.Solver`.

    :Attributes:

       - :self.beta:     Newmark numerical stability parameter
       - :self.gamma:    Newmark numerical stability parameter
    """

    def __init__(self):
        super(NewmarkSolver, self).__init__()
        self.beta = 0.25
        self.gamma = 0.5

    def update_force(self, u, F_previous, t):
        """
        Updates the external force vector at time t

        :param u: displacement vector at time t
        :param F_previous: Force vector at previous time step
        :param t:  current time step index
        :return:
        """

        # calculates force with custom load function
        force = self.load_func(u, t)

        # Convert force vector to a 1d numpy array
        if issparse(force):
            force = force.toarray()[:, 0]

        # calculate force increment with respect to the previous time step
        d_force = force - F_previous

        # copy force vector such that force vector data at each time step is maintained
        F_total = np.copy(force)

        return d_force, F_total

    def calculate(self, M, C, K, F, t_start_idx, t_end_idx):
        """
        Newmark integration scheme.
        Incremental formulation.

        :param M: Mass matrix
        :param C: Damping matrix
        :param K: Stiffness matrix
        :param F: External force matrix
        :param t_start_idx: time index of starting time for the stage analysis
        :param t_end_idx: time index of end time for the stage analysis
        :return:
        """

        # validate solver index
        self.validate_input(F, t_start_idx, t_end_idx)

        # calculate time step size
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
        self.f[t_start_idx, :] = d_force

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
            F_previous = F[:, t_start_idx].toarray()[:, 0]
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

        # calculate nodal force
        self.f[:, :] = np.transpose(K.dot(np.transpose(self.u)))
        # close the progress bar
        pbar.close()


class StaticSolver(Solver):
    """
    Static Solver class. This class contains the static incremental solver. This class bases from
    :class:`~rose.model.solver.Solver`.

    """

    def __init__(self):
        super(StaticSolver, self).__init__()

    def calculate(self, K, F, t_start_idx, t_end_idx):
        """
        Static integration scheme.
        Incremental formulation.

        :param K: Stiffness matrix
        :param F: External force matrix
        :param t_start_idx: time index of starting time for the stage analysis
        :param t_end_idx: time index of end time for the stage analysis
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
