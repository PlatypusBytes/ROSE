import numpy as np
import matplotlib.pylab as plt


class TwoDofVehicle:
    """
    Based on analytical solution of Fryba (1972) (chapter 8)
    """
    def __init__(self):

        # vehicle
        self.m1 = []  # mass unsprung mass
        self.m2 = []  # mass sprung mass
        self.P1 = []  # weight unsprung mass
        self.P2 = []  # weight sprung mass
        self.P = []  # total vehicle weight
        self.speed = []  # vehicle speed
        self.k_contact = []  # contact stiffness
        self.k_vehicle = []  # vehicle stiffness
        self.circumference = []  # wheel circumference
        self.force = []  # additional harmonic static force

        # beam
        self.length = []  # length of beam
        self.mass = []   # mass per unit length
        self.EI = []  # bending stiffness of the beam

        # dimensionless
        self.alpha = []  # speed parameter
        self.chi = []  # ratio between eights of vehicle and beam
        self.chi_0 = []  # ratio between unsprung and sprung vehicle
        self.gamma_1 = []  # frequency parameter of unsprung mass
        self.gamma_1_dash = []  # frequency parameter of unsprung mass
        self.gamma_2 = []  # frequency parameter of sprung mass
        self.gamma_2_dash = []  # frequency parameter of sprung mass
        self.a = 0  # relation between stiffness of elastic layer of beam
        self.b = 2  # function of beam span
        self.Q = []  # harmonic force
        self.a1 = []  # amplitude harmonic force
        self.b1 = []  # frequency harmonic force
        self.a2 = 0  # track uneveness
        self.b2 = 2  # lenght track uneveness
        qsi = []  # decrement of beam damping
        qsi_2 = []  # decrement of vehicle damping


        # constants
        self.g = 9.81
        return

    def vehicle(self, m1, m2, speed, k1):
        """
        Define vehicle properties

        :param m1: mass of unsprung vehicle part
        :param m2: mass of sprung vehicle part
        :param speed: vehicle speed
        :param k1: contact stiffness
        """

        self.m1 = m1
        self.m2 = m2
        self.P1 = m1 * self.g
        self.P2 = m2 * self.g
        self.P = self.P1 + self.P2
        self.speed = speed
        self.k1 = k1
        return

    def beam(self, E, I,  rho, A, L):

        self.EI = E * I
        self.mass = rho * A
        self.length = L

        # determine first eigen frequency
        self.omega_1 = self.eigen_freq(1)

        # bridge stiffness
        self.C0 = 1
        return

    def eigen_freq(self, n):
        """
        Computes eigen frequency for a simple supported beam

        :param n: n mode
        :return eigen frequency for *n* mode
        """
        self.eig = n ** 2 * np.pi ** 2 * np.sqrt(self.EI / (self.mass * self.length ** 4))
        return self.eig



    def dimensionless_parameters(self):

        f_b_1 = self.eigen_freq(1) / (2 * np.pi)  # first natural frequency of beam in Hz
        self.alpha = self.speed / 2 * f_b_1 * self.length  # speed parameter
        self.chi = self.P / (self.mass * self.g * self.length)  # ratio between eights of vehicle and beam
        self.chi_0 = self.P1 / self.P2   # ratio between unsprung and sprung vehicle

        f_m_1 = 1 / (2 * np.pi) * (self.k_contact1 / self.m1) ** 0.5  # natural frequency unsprung mass
        self.gamma_1 = f_m_1 / f_b_1  # frequency parameter of unsprung mass
        self.gamma_1_dash = self.gamma_1 * (1 + self.chi_0) / (2 * self.chi * self.chi_0)

        f_m_2 = 1 / (2 * np.pi) * (self.k_vehicle / self.m2) ** 0.5  # natural frequency sprung mass
        self.gamma_2 = f_m_2 / f_b_1  # frequency parameter of sprung mass
        self.gamma_2_dash = self.gamma_2 * (1 + self.chi_0) / (2 * self.chi)

        omega = 2 * np.pi * self.speed / self.circumference
        self.Q = self.force * np.sin(omega * self.time)
        self.a1 = self.force / self.P
        self.b1 = 2 * self.length / self.circumference

        self.qsi = / f_b_1

        self.tau0 =
        self.q_0_dot
        self.a2 =
        self.vs =
        return


if __name__ == "__main__":
    ss = TwoDofVehicle()
    ss.vehicle(100, 20, 40, 1e6)
    ss.beam(20e6, 1e-4, 2000, 0.01, 20)