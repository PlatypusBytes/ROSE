import numpy as np
import sys
import json


class MovingLoad:
    def __init__(self, nb_terms=100, a=0.05):
        """
        Initialise

        :param nb_terms: number of terms for Fourier solution
        :param a: small value for the integration around the pole
        """
        self.EI = []
        self.speed = []
        self.rho = []
        self.stiffness = []  # stiffness
        self.time = []  # time
        self.force = []  # force
        self.tau = []  # dimensionless time
        self.qsi = []  # dimensionless position
        self.period = []  # period
        self.u = []  # displacement for x < 0
        self.uu = []  # displacement for x >= 0
        self.displacement = []  # total displacement
        self.position = []  # position
        self.k_ratio = []  # ratio stiffness

        self.nb_terms = int(nb_terms)
        self.a = a

        self.result = {}
        return

    def parameters(self, x, speed, E, I, rho, k, force):

        self.position = x
        self.speed = speed
        self.EI = E * I
        self.rho = rho
        # check if stiffness size is two
        if len(k) != 2:
            sys.exit("Error: stiffness len must be two")
        self.stiffness = k

        # check if speed smaller than critical speed
        c_crit = (4 * min(self.stiffness) * self.EI / self.rho ** 2) ** (1 / 4)
        print(c_crit*0.95)
        if speed > c_crit:
            sys.exit(f"Error: travelling speed higher than critical speed: {round(c_crit, 2)} m/s")

        self.time = self.position / self.speed
        self.force = force

        # constants
        self.beta = np.sqrt(self.EI / self.rho)
        self.h = np.sqrt(self.stiffness[0] / self.rho)
        self.alpha = self.speed / np.sqrt(self.beta * self.h)
        self.F = self.force / (self.rho * self.h ** 1.5 * self.beta ** 0.5)

        # dimensionless time
        self.tau = self.h * self.time
        # dimensionless position
        self.qsi = np.sqrt(self.h / self.beta) * self.position

        # period
        self.period = self.tau[-1] * 2

        # create displacement variable
        self.u = np.zeros((len(self.qsi), len(self.tau)))
        self.uu = np.zeros((len(self.qsi), len(self.tau)))

        # constants
        self.qa = -np.sqrt(1 / 2 * (-self.alpha ** 2 + 1j * np.sqrt(4 - self.alpha ** 4)))
        self.qb = -np.sqrt(1 / 2 * (-self.alpha ** 2 - 1j * np.sqrt(4 - self.alpha ** 4)))

        self.A = self.F / (2 * 1j * self.qa * np.sqrt(4 - self.alpha ** 4))
        self.B = - self.qa / self.qb * self.A

        # ratios k
        self.k_ratio = self.stiffness[1] / self.stiffness[0]

        return

    def solve(self):

        # static variables
        self.static_coefficients()

        # solve for time < 0
        self.time_low()

        # solve for time >=0
        self.time_high()

        # combine solution
        self.displacement = self.u + self.uu
        return

    def time_low(self):

        for i, t in enumerate(self.tau[self.tau < [0]]):

            # initialize aux variables
            u1 = np.zeros(len(self.qsi))
            u2 = np.zeros(len(self.qsi))

            # eta
            eta = self.qsi - self.alpha * t

            # homogeneous solution
            u1h, u2h = self.coefficients_low(t)
            # particular solution
            u1p = self.A * np.exp(self.qa * np.abs(eta)) + self.B * np.exp(self.qb * np.abs(eta))

            u1[self.qsi <= [0]] = np.real(u1p[self.qsi <= [0]]) + np.real(u1h[self.qsi <= [0]])
            u2[self.qsi > [0]] = np.real(u2h[self.qsi > [0]])
            self.u[:, i] = u1 + u2

        return

    def time_high(self):

        # initialize aux variables
        F1 = np.zeros((len(self.qsi), len(self.tau)))
        F2 = np.zeros((len(self.qsi), len(self.tau)))

        # calculation F1, F2, F
        for t in self.tau[self.tau >= [0]]:
            i = np.where(self.tau == t)[0][0]  # index of time

            # aux variable
            ubar = np.zeros(len(self.tau))

            ub1, ub2 = self.coefficients_bar(self.qsi, self.a)

            ubar[self.qsi <= [0]] = np.real(ub1)[self.qsi <= [0]]
            ubar[self.qsi > [0]] = np.real(ub2)[self.qsi > [0]]

            F1[:, i] += 1 / 2 * ubar

        # calculation F2
        for t in self.tau[self.tau >= [0]]:
            i = np.where(self.tau == t)[0][0]  # index of time
            for k in range(1, self.nb_terms):
                s = self.a + k * np.pi * 1j / self.period
                ub1, ub2 = self.coefficients_bar(self.qsi, s)

                # aux variable
                ubar = np.zeros(len(self.tau))
                ubar[self.qsi <= [0]] = np.real(ub1)[self.qsi <= [0]]
                ubar[self.qsi > [0]] = np.real(ub2)[self.qsi > [0]]
                F2[:, i] += np.real(ubar) * np.cos(k * np.pi * t / self.period)

            self.uu[:, i] = 2 * np.exp(self.a * t) / self.period * (F1[:, i] + F2[:, i])

        return

    def coefficients_low(self, t):

        W1a = self.C1a * np.exp(self.r1a1 * self.qsi) + self.D1a * np.exp(self.r1a2 * self.qsi)
        W1b = self.C1b * np.exp(self.r1b1 * self.qsi) + self.D1b * np.exp(self.r1b2 * self.qsi)
        W2a = self.C2a * np.exp(self.r2a1 * self.qsi) + self.D2a * np.exp(self.r2a2 * self.qsi)
        W2b = self.C2b * np.exp(self.r2b1 * self.qsi) + self.D2b * np.exp(self.r2b2 * self.qsi)

        u1h = W1a * np.exp(-self.qa * self.alpha * t) + W1b * np.exp(-self.qb * self.alpha * t)
        u2h = W2a * np.exp(-self.qa * self.alpha * t) + W2b * np.exp(-self.qb * self.alpha * t)

        return u1h, u2h

    def coefficients_bar(self, x, s):

        self.s1 = np.sqrt(1 + s ** 2)
        self.s2 = np.sqrt(self.k_ratio + s ** 2)

        self.rs11 = np.sqrt(1j * self.s1)
        self.rs12 = np.sqrt(-1j * self.s1)
        self.rs21 = -np.sqrt(1j * self.s2)
        self.rs22 = -np.sqrt(-1j * self.s2)

        self.ZZ1 = self.C1a * (s - self.alpha * self.qa) / (self.s1 ** 2 + self.r1a1 ** 4) + self.D1a * (s - self.alpha * self.qa) / (self.s1 ** 2 + self.r1a2 ** 4) + self.C1b * (s - self.alpha * self.qb) / (self.s1 ** 2 - self.r1b1 ** 4) + \
                   self.D1b * (s - self.alpha * self.qb) / (self.s1 ** 2 + self.r1b2 ** 4) + self.A * (s + self.alpha * self.qa) / (self.s1 ** 2 + self.qa ** 4) + self.B * (s + self.alpha * self.qb) / (self.s1 ** 2 + self.qb ** 4) - \
                   self.C2a * (s - self.alpha * self.qa) / (self.s2 ** 2 + self.r2a1 ** 4) - self.D2a * (s - self.alpha * self.qa) / (self.s2 ** 2 + self.r2a2 ** 4) - self.C2b * (s - self.alpha * self.qb) / (self.s2 ** 2 + self.r2b1 ** 4) - \
                   self.D2b * (s - self.alpha * self.qb) / (self.s2 ** 2 + self.r2b2 ** 4) - self.F / (self.alpha * (self.s2 ** 2 + (s ** 4) / (self.alpha ** 4)))

        self.ZZ2 = self.r1a1 * self.C1a * (s - self.alpha * self.qa) / (self.s1 ** 2 + self.r1a1 ** 4) + self.r1a2 * self.D1a * (s - self.alpha * self.qa) / (self.s1 ** 2 + self.r1a2 ** 4) + \
                   self.r1b1 * self.C1b * (s - self.alpha * self.qb) / (self.s1 ** 2 + self.r1b1 ** 4) + self.r1b2 * self.D1b * (s - self.alpha * self.qb) / (self.s1 ** 2 + self.r1b2 ** 4) - \
                   self.qa * self.A * (s + self.alpha * self.qa) / (self.s1 ** 2 + self.qa ** 4) - self.qb * self.B * (s + self.alpha * self.qb) / (self.s1 ** 2 + self.qb ** 4) - self.r2a1 * self.C2a * (s - self.alpha * self.qa) / (self.s2 ** 2 + self.r2a1 ** 4) - \
                   self.r2a2 * self.D2a * (s - self.alpha * self.qa) / (self.s2 ** 2 + self.r2a2 ** 4) - self.r2b1 * self.C2b * (s - self.alpha * self.qb) / (self.s2 ** 2 + self.r2b1 ** 4) - self.r2b2 * self.D2b * (s - self.alpha * self.qb) / (self.s2 ** 2 + self.r2b2 ** 4) + \
                   s * self.F / (self.alpha ** 2 * (self.s2 ** 2 + s ** 4 / self.alpha ** 4))

        self.ZZ3 = self.r1a1 ** 2 * self.C1a * (s - self.alpha * self.qa) / (self.s1 ** 2 + self.r1a1 ** 4) + self.r1a2 ** 2 * self.D1a * (s - self.alpha * self.qa) / (self.s1 ** 2 + self.r1a2 ** 4) + \
                   self.r1b1 ** 2 * self.C1b * (s - self.alpha * self.qb) / (self.s1 ** 2 + self.r1b1 ** 4) + self.r1b2 ** 2 * self.D1b * (s - self.alpha * self.qb) / (self.s1 ** 2 + self.r1b2 ** 4) + \
                   self.qa ** 2 * self.A * (s + self.alpha * self.qa) / (self.s1 ** 2 + self.qa ** 4) + self.qb ** 2 * self.B * (s + self.alpha * self.qb) / (self.s1 ** 2 + self.qb ** 4) - \
                   self.r2a1 ** 2 * self.C2a * (s - self.alpha * self.qa) / (self.s2 ** 2 + self.r2a1 ** 4) - self.r2a2 ** 2 * self.D2a * (s - self.alpha * self.qa) / (self.s2 ** 2 + self.r2a2 ** 4) - \
                   self.r2b1 ** 2 * self.C2b * (s - self.alpha * self.qb) / (self.s2 ** 2 + self.r2b1 ** 4) - self.r2b2 ** 2 * self.D2b * (s - self.alpha * self.qb) / (self.s2 ** 2 + self.r2b2 ** 4) - \
                   s ** 2 * self.F / (self.alpha ** 3 * (self.s2 ** 2 + s ** 4 / self.alpha ** 4))

        self.ZZ4 = self.r1a1 ** 3 * self.C1a * (s - self.alpha * self.qa) / (self.s1 ** 2 + self.r1a1 ** 4) + self.r1a2 ** 3 * self.D1a * (s - self.alpha * self.qa) / (self.s1 ** 2 + self.r1a2 ** 4) + \
                   self.r1b1 ** 3 * self.C1b * (s - self.alpha * self.qb) / (self.s1 ** 2 + self.r1b1 ** 4) + self.r1b2 ** 3 * self.D1b * (s - self.alpha * self.qb) / (self.s1 ** 2 + self.r1b2 ** 4) - \
                   self.qa ** 3 * self.A * (s + self.alpha * self.qa) / (self.s1 ** 2 + self.qa ** 4) - self.qb ** 3 * self.B * (s + self.alpha * self.qb) / (self.s1 ** 2 + self.qb ** 4) - \
                   self.r2a1 ** 3 * self.C2a * (s - self.alpha * self.qa) / (self.s2 ** 2 + self.r2a1 ** 4) - self.r2a2 ** 3 * self.D2a * (s - self.alpha * self.qa) / (self.s2 ** 2 + self.r2a2 ** 4) - \
                   self.r2b1 ** 3 * self.C2b * (s - self.alpha * self.qb) / (self.s2 ** 2 + self.r2b1 ** 4) - self.r2b2 ** 3 * self.D2b * (s - self.alpha * self.qb) / (self.s2 ** 2 + self.r2b2 ** 4) + \
                   s ** 3 * self.F / (self.alpha ** 4 * (self.s2 ** 2 + s ** 4 / self.alpha ** 4))

        self.E1 = (self.rs12 * self.rs21 * self.rs22 * self.ZZ1 - self.rs12 * self.rs21 * self.ZZ2 - self.rs12 * self.rs22 * self.ZZ2 - \
                   self.rs21 * self.rs22 * self.ZZ2 + self.rs12 * self.ZZ3 + self.rs21 * self.ZZ3 + self.rs22 * self.ZZ3 - self.ZZ4) / \
                  ((self.rs11 - self.rs12) * (self.rs11 ** 2 - self.rs11 * self.rs22 - self.rs11 * self.rs21 + self.rs21 * self.rs22))
        self.G1 = (self.rs11 * self.rs21 * self.rs22 * self.ZZ1 - self.rs11 * self.rs21 * self.ZZ2 - self.rs11 * self.rs22 * self.ZZ2 - \
                   self.rs21 * self.rs22 * self.ZZ2 + self.rs11 * self.ZZ3 + self.rs21 * self.ZZ3 + self.rs22 * self.ZZ3 - self.ZZ4) / \
                  ((self.rs12 - self.rs11) * (self.rs12 - self.rs21) * (self.rs12 - self.rs22))
        self.E2 = (self.rs11 * self.rs12 * self.rs22 * self.ZZ1 - self.rs11 * self.rs12 * self.ZZ2 - self.rs11 * self.rs22 * self.ZZ2 - \
                   self.rs12 * self.rs22 * self.ZZ2 + self.rs11 * self.ZZ3 + self.rs12 * self.ZZ3 + self.rs22 * self.ZZ3 - self.ZZ4) / \
                  ((self.rs22 - self.rs21) * (self.rs21 ** 2 - self.rs11 * self.rs21 - self.rs12 * self.rs21 + self.rs11 * self.rs12))
        self.G2 = (self.rs11 * self.rs12 * self.rs21 * self.ZZ1 - self.rs11 * self.rs12 * self.ZZ2 - self.rs11 * self.rs21 * self.ZZ2 - \
                   self.rs12 * self.rs21 * self.ZZ2 + self.rs11 * self.ZZ3 + self.rs12 * self.ZZ3 + self.rs21 * self.ZZ3 - self.ZZ4) / \
                  ((self.rs11 - self.rs22) * (self.rs12 - self.rs22) * (self.rs21 - self.rs22))

        ub1 = self.C1a * (s - self.alpha * self.qa) / (self.s1 ** 2 + self.r1a1 ** 4) * np.exp(self.r1a1 * x) + \
              self.D1a * (s - self.alpha * self.qa) / (self.s1 ** 2 + self.r1a2 ** 4) * np.exp(self.r1a2 * x) + \
              self.C1b * (s - self.alpha * self.qb) / (self.s1 ** 2 + self.r1b1 ** 4) * np.exp(self.r1b1 * x) + \
              self.D1b * (s - self.alpha * self.qb) / (self.s1 ** 2 + self.r1b2 ** 4) * np.exp(self.r1b2 * x) + \
              self.A * (s + self.alpha * self.qa) / (self.s1 ** 2 + self.qa ** 4) * np.exp(-self.qa * x) + \
              self.B * (s + self.alpha * self.qb) / (self.s1 ** 2 + self.qb ** 4) * np.exp(-self.qb * x) + \
              self.E1 * np.exp(self.rs11 * x) + self.G1 * np.exp(self.rs12 * x)

        ub2 = self.C2a * (s - self.alpha * self.qa) / (self.s2 ** 2 + self.r2a1 ** 4) * np.exp(self.r2a1 * x) + \
              self.D2a * (s - self.alpha * self.qa) / (self.s2 ** 2 + self.r2a2 ** 4) * np.exp(self.r2a2 * x) + \
              self.C2b * (s - self.alpha * self.qb) / (self.s2 ** 2 + self.r2b1 ** 4) * np.exp(self.r2b1 * x) + \
              self.D2b * (s - self.alpha * self.qb) / (self.s2 ** 2 + self.r2b2 ** 4) * np.exp(self.r2b2 * x) + \
              self.F / (self.alpha * (self.s2 ** 2 + s ** 4 / self.alpha ** 4)) * np.exp(-s / self.alpha * x) + \
              self.E2 * np.exp(self.rs21 * x) + self.G2 * np.exp(self.rs22 * x)

        return ub1, ub2

    def static_coefficients(self):
        # for low
        g1a = np.sqrt(1 + self.qa ** 2 * self.alpha ** 2)
        g1b = np.sqrt(1 + self.qb ** 2 * self.alpha ** 2)
        g2a = np.sqrt(self.k_ratio + self.qa ** 2 * self.alpha ** 2)
        g2b = np.sqrt(self.k_ratio + self.qb ** 2 * self.alpha ** 2)

        self.r1a1 = np.sqrt(1j * g1a)
        self.r1a2 = np.sqrt(-1j * g1a)
        self.r1b1 = np.sqrt(1j * g1b)
        self.r1b2 = np.sqrt(-1j * g1b)
        self.r2a1 = -np.sqrt(1j * g2a)
        self.r2a2 = -np.sqrt(-1j * g2a)
        self.r2b1 = -np.sqrt(1j * g2b)
        self.r2b2 = -np.sqrt(-1j * g2b)

        self.C1a = -((self.qa - self.r1a2) * (self.qa - self.r2a2) * (self.qa - self.r2a1)) * self.A / ((self.r1a1 - self.r1a2) * (self.r2a2 - self.r1a1) * (self.r2a1 - self.r1a1))
        self.C2a = ((self.qa - self.r1a2) * (self.qa - self.r1a1) * (self.qa - self.r2a2)) * self.A / ((self.r2a1 - self.r2a2) * (self.r2a1 - self.r1a2) * (self.r2a1 - self.r1a1))
        self.D1a = ((self.qa - self.r1a1) * (self.qa - self.r2a2) * (self.qa - self.r2a1)) * self.A / ((self.r2a2 - self.r1a2) * (self.r2a1 - self.r1a2) * (self.r1a1 - self.r1a2))
        self.D2a = -((self.qa - self.r1a2) * (self.qa - self.r1a1) * (self.qa - self.r2a1)) * self.A / ((self.r2a2 - self.r1a2) * (self.r2a2 - self.r1a1) * (self.r2a1 - self.r2a2))
        self.C1b = -((self.qb - self.r1b2) * (self.qb - self.r2b2) * (self.qb - self.r2b1)) * self.B / ((self.r1b1 - self.r1b2) * (self.r1b1 - self.r2b2) * (self.r1b1 - self.r2b1))
        self.C2b = -((self.qb - self.r1b2) * (self.qb - self.r2b2) * (self.qb - self.r1b1)) * self.B / ((self.r2b1 - self.r1b2) * (self.r1b1 - self.r2b1) * (self.r2b1 - self.r2b2))
        self.D1b = ((self.qb - self.r2b2) * (self.qb - self.r2b1) * (self.qb - self.r1b1)) * self.B / ((self.r2b2 - self.r1b2) * (self.r2b1 - self.r1b2) * (self.r1b1 - self.r1b2))
        self.D2b = -((self.qb - self.r1b1) * (self.qb - self.r1b2) * (self.qb - self.r2b1)) * self.B / ((self.r1b1 - self.r2b2) * (self.r1b2 - self.r2b2) * (self.r2b1 - self.r2b2))
        return

    def write_results(self, output="./results.json"):
        """
        Writes and saves output in a json file

        :param output: path to write json file (default "./results.json")
        """

        # create dictionary for results
        self.result = {"time": self.time.tolist(),
                       "coordinates": self.position.tolist(),
                       "displacement": self.displacement.tolist(),
                       }
        # dump results
        with open(output, "w") as f:
            json.dump(self.result, f, indent=2)

        return


if __name__ == "__main__":
    import time
    t_ini = time.time()

    stiffness_spring = 4.27e5
    stiffness_spring_2 = stiffness_spring * 2
    distance_springs = 1
    winkler_mod_1 = stiffness_spring / distance_springs
    winkler_mod_2 = stiffness_spring_2 / distance_springs
    youngs_mod_beam = 1.28e7
    inertia_beam = 1
    rho = 119.7
    y_load = -1e6
    speed = 100

    p = MovingLoad()
    p.parameters(np.linspace(-100, 100, 401), speed, youngs_mod_beam, inertia_beam, rho, [winkler_mod_1, winkler_mod_2], y_load)
    p.solve()
    p.write_results(output="./results.json")
    print(f"Time: {time.time() - t_ini}")

    import matplotlib.pylab as plt
    plt.plot(p.qsi, p.displacement[:, 250], color="b")
    plt.plot(p.qsi, p.displacement[:, 200], color="k", marker='x')
    plt.plot(p.qsi, p.displacement[:, 210], color="k")
    plt.show()
