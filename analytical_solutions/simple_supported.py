import numpy as np
import json
from scipy.optimize import root


class SimpleSupportEulerStatic:
    def __init__(self, ele_size=0.1):
        """
        Initialise the object

        :param ele_size: element size (default 0.1m)
        """
        self.element_size = ele_size  # element size
        self.EI = []  # bending stiffness of the beam
        self.length = []  # length of the beam
        self.force = []  # force
        self.x = []  # discretisation of the beam
        self.x_load = []  # coordinate of the point load

        self.u = []  # vertical displacement

        self.result = {}  # dictionary for json dump
        return

    def properties(self, E, I, L, F, x_F):
        """
        Assigns properties

        :param E: Young modulus
        :param I: Inertia
        :param L: Length
        :param F: Force
        :param x_F: # coordinate of the point load
        """
        self.EI = E * I
        self.length = L
        self.force = F
        self.x_load = x_F

        self.x = np.linspace(0, self.length, int(np.ceil(self.length / self.element_size) + 1))
        self.u = np.zeros((len(self.x)))

        return

    def compute(self):
        """
        Computes the displacement solution
        """

        a = self.x_load
        b = self.length - a

        for idx, coord in enumerate(self.x):
            if 0 <= coord <= self.x_load:
                self.u[idx] = self.force * b * coord / (6*self.length * self.EI) * (self.length**2 - b**2 - coord**2)
            elif coord <= self.length:
                self.u[idx] = self.force * a * (self.length - coord) / (6 * self.length * self.EI) * (
                            self.length ** 2 - a ** 2 - (self.length - coord) ** 2)

        return

    def write_results(self, output="./results.json"):
        """
        Writes and saves output in a json file

        :param output: path to write json file (default "./results.json")
        """
        # create dictionary for results
        self.result = {"coordinates": self.x.tolist(),
                       f"deflection ": self.u.tolist()}

        # dump results
        with open(output, "w") as f:
            json.dump(self.result, f, indent=2)

        return


class SimpleSupportEulerNoDamping:
    """
    Analytical solution for a pulse load in the middle of a simple supported beam (Euler beam).
    No Damping.
    """
    def __init__(self, n=100, ele_size=0.1):
        """
        Initialise the object

        :param n: number of modes (default 100)
        :param ele_size: element size (default 0.1m)
        """
        self.element_size = ele_size  # element size
        self.n = n  # number of modes
        self.EI = []  # bending stiffness of the beam
        self.mass = []  # unit mass of the beam
        self.length = []  # length of the beam
        self.force = []  # force
        self.time = []  # time
        self.x = []  # discretisation of the beam

        self.eig = []  # eigen frequency n mode
        self.eig_shape = []  # eigen shape n mode

        self.u = []  # vertical displacement

        self.result = {}  # dictionary for json dump
        return

    def properties(self, E, I, rho, A, L, F, time):
        """
        Assigns properties

        :param E: Young modulus
        :param I: Inertia
        :param rho: Density
        :param A: Area
        :param L: Length
        :param F: Force
        :param time: np.array vector
        """
        self.EI = E * I
        self.mass = rho * A
        self.length = L
        self.force = F
        self.time = time

        self.x = np.linspace(0, self.length, int(np.ceil(self.length / self.element_size) + 1))

        self.u = np.zeros((len(self.x), len(self.time)))

        return

    def eigen_freq(self, n):
        """
        Computes eigen frequency for a simple supported beam

        :param n: n mode
        """
        self.eig = n ** 2 * np.pi ** 2 * np.sqrt(self.EI / (self.mass * self.length ** 4))
        return

    def mode_shape(self, n):
        """
        Computes eigen *n* shape for a simple supported beam

        :param n: n mode
        """
        self.eig_shape = np.sin(n * np.pi / self.length * self.x)
        return

    def compute(self):
        """
        Computes the displacement solution
        """
        # for each time step
        for idx, t in enumerate(self.time):
            aux = np.zeros(len(self.x))
            # for the desired number of modes
            for n in range(1, self.n):
                # compute eigen frequency
                self.eigen_freq(n)
                # eigen mode
                self.mode_shape(n)

                # compute force: at middle span
                force = self.eig_shape[int((len(self.x)-1) / 2)] * self.force

                aux += force / n**4 * (1 - np.cos(self.eig * t)) * self.eig_shape
            # add to displacement
            self.u[:, idx] = 2 * self.length**3 / (np.pi**4 * self.EI) * aux

        return

    def write_results(self, output="./results.json"):
        """
        Writes and saves output in a json file

        :param output: path to write json file (default "./results.json")
        """
        # create dictionary for results
        self.result = {"time": self.time.tolist(),
                       "coordinates": self.x.tolist()}
        # for each node add results
        for i, x in enumerate(self.x):
            self.result.update({f"x-coordinate {x.round(2)}": self.u[i, :].tolist()})

        # dump results
        with open(output, "w") as f:
            json.dump(self.result, f, indent=2)

        return


class SimpleSupportEulerWithDamping:
    """
    Analytical solution for a pulse load in the middle of a simple supported beam (Euler beam).
    With Damping.
    """
    def __init__(self, n=100, ele_size=0.1):
        """
        Initialise the object

        :param n: number of modes (default 100)
        :param ele_size: element size (default 0.1m)
        """
        self.element_size = ele_size  # element size
        self.n = n  # number of modes
        self.EI = []  # bending stiffness of the beam
        self.E = []  # Young modulus
        self.G = []  # Shear modulus
        self.r = []  # radius of gyration
        self.k = []  # Timoshenko coefficient
        self.mass = []  # unit mass of the beam
        self.length = []  # length of the beam
        self.force = []  # force
        self.time = []  # time
        self.x = []  # discretisation of the beam

        self.eig = []  # eigen frequency n mode
        self.eig_d = []  # eigen frequency n mode with damping
        self.eig_shape = []  # eigen shape n mode
        self.damp_coef = []  # damping coefficients
        self.qsi = []  # damping

        self.u = []  # vertical displacement

        self.result = {}  # dictionary for json dump
        return

    def properties(self, E, I, rho, A, L, F, damp_coef, time):
        """
        Assigns properties

        :param E: Young modulus
        :param I: Inertia
        :param rho: Density
        :param A: Area
        :param L: Length
        :param F: Force
        :param damp_coef: Damping coefficients
        :param time: np.array vector
        """
        self.EI = E * I
        self.mass = rho * A
        self.length = L
        self.force = F
        self.time = time
        self.damp_coef = damp_coef

        self.x = np.linspace(0, self.length, int(np.ceil(self.length / self.element_size) + 1))

        self.u = np.zeros((len(self.x), len(self.time)))

        return

    def eigen_freq(self, n):
        """
        Computes eigen frequency and daping ratio for a simple supported beam, n mode

        :param n: n mode
        """
        self.eig = n ** 2 * np.pi ** 2 * np.sqrt(self.EI / (self.mass * self.length ** 4))
        self.qsi = self.damp_coef[0] / (2 * self.eig) + self.damp_coef[1] * self.eig / 2
        self.eig_d = self.eig * np.sqrt(1 - self.qsi ** 2)
        return

    def mode_shape(self, n):
        """
        Computes eigen *n* shape for a simple supported beam

        :param n: n mode
        """
        self.eig_shape = np.sin(n * np.pi / self.length * self.x)
        return

    def compute(self):
        """
        Computes the displacement solution
        """
        # for each time step
        for idx, t in enumerate(self.time):
            aux = np.zeros(len(self.x))
            # for the desired number of modes
            for n in range(1, self.n):
                # compute eigen frequency and damping ratio
                self.eigen_freq(n)
                # eigen mode
                self.mode_shape(n)
                # mass
                mass = self.mass * self.length / 2
                # stiffness
                stiff = self.eig ** 2 * mass

                # compute force: at middle span
                force = self.eig_shape[int((len(self.x)-1) / 2)] * self.force

                # displacement n mode
                aux += force / stiff * (1 - np.exp(-self.qsi * self.eig * t) * (np.cos(self.eig_d * t) + self.qsi / (np.sqrt(1 - self.qsi**2)) * np.sin(self.eig_d * t))) * self.eig_shape

            # add to displacement
            self.u[:, idx] = aux

        return

    def write_results(self, output="./results.json"):
        """
        Writes and saves output in a json file

        :param output: path to write json file (default "./results.json")
        """
        # create dictionary for results
        self.result = {"time": self.time.tolist(),
                       "coordinates": self.x.tolist()}
        # for each node add results
        for i, x in enumerate(self.x):
            self.result.update({f"x-coordinate {x.round(2)}": self.u[i, :].tolist()})

        # dump results
        with open(output, "w") as f:
            json.dump(self.result, f, indent=2)

        return


class SimpleSupportTimoshenkoNoDamping:
    """
    Analytical solution for a pulse load in the middle of a simple supported beam (Timoshenko beam).
    No Damping.
    """
    def __init__(self, n=100, ele_size=0.1):
        """
        Initialise the object

        :param n: number of modes (default 100)
        :param ele_size: element size (default 0.1m)
        """
        self.element_size = ele_size  # element size
        self.n = n  # number of modes
        self.EI = []  # bending stiffness of the beam
        self.mass = []  # unit mass of the beam
        self.length = []  # length of the beam
        self.force = []  # force
        self.time = []  # time
        self.x = []  # discretisation of the beam

        self.eig = []  # eigen frequency n mode
        self.eig_shape = []  # eigen shape n mode

        self.u = []  # vertical displacement

        self.result = {}  # dictionary for json dump
        return

    def properties(self, E, G, k, I, rho, A, L, F, time):
        """
        Assigns properties

        :param E: Young modulus
        :param G: Shear modulus
        :param k: Timoshenko coefficient
        :param I: Inertia
        :param rho: Density
        :param A: Area
        :param L: Length
        :param F: Force
        :param time: np.array vector
        """
        self.EI = E * I
        self.E = E
        self.G = G
        self.k = k
        self.rho = rho
        self.r = np.sqrt(I / A)
        self.mass = rho * A
        self.length = L
        self.phi = (12 / L ** 2) * (E * I / (k * G * A))
        self.force = F
        self.time = time

        self.x = np.linspace(0, self.length, int(np.ceil(self.length / self.element_size) + 1))

        self.u = np.zeros((len(self.x), len(self.time)))

        return

    def eigen_freq(self, n):
        """
        Computes eigen frequency for a simple supported beam

        :param n: n mode
        """

        # initial guess
        omega = 1 / np.sqrt(1 + (n * np.pi * self.r / self.length)**2 * (1 + self.E / (self.k * self.G)))
        # find coordinates that minimize function
        res = root(self.find_omega, x0=omega, args=n, tol=1e-16)
        omega = res.x[0]

        self.eig = omega * (n ** 2 * np.pi ** 2 * np.sqrt(self.EI / (self.mass * self.length ** 4)))
        return

    def mode_shape(self, n):
        """
        Computes eigen *n* shape for a simple supported beam

        :param n: n mode
        """
        self.eig_shape = np.sin(n * np.pi / self.length * self.x)
        return

    def compute(self):
        """
        Computes the displacement solution
        """
        # for each time step
        for idx, t in enumerate(self.time):
            aux = np.zeros(len(self.x))
            # for the desired number of modes
            for n in range(1, self.n):
                # compute eigen frequency
                self.eigen_freq(n)
                # eigen mode
                self.mode_shape(n)

                # compute force: at middle span
                force = self.eig_shape[int((len(self.x)-1) / 2)] * self.force

                aux += force / n**4 * (1 - np.cos(self.eig * t)) * self.eig_shape
            # add to displacement
            self.u[:, idx] = 2 * self.length**3 * (1+self.phi) / (np.pi**4 * self.EI) * aux

        return

    def write_results(self, output="./results.json"):
        """
        Writes and saves output in a json file

        :param output: path to write json file (default "./results.json")
        """
        # create dictionary for results
        self.result = {"time": self.time.tolist(),
                       "coordinates": self.x.tolist()}
        # for each node add results
        for i, x in enumerate(self.x):
            self.result.update({f"x-coordinate {x.round(2)}": self.u[i, :].tolist()})

        # dump results
        with open(output, "w") as f:
            json.dump(self.result, f, indent=2)

        return

    def find_omega(self, omega, n):
        aux = (1 - omega ** 2) - omega**2 * (n * np.pi * self.r / self.length)**2 * (1 + self.E / (self.k * self.G)) +\
              omega ** 4 * (n * np.pi * self.r / self.length)**4 * self.E / (self.k * self.G)
        return aux

if __name__ == "__main__":

    sss = SimpleSupportEulerNoDamping()
    sss.properties(20e6, 1e-4, 2000, 0.01, 20, -1000, np.linspace(0, 200, 1000))
    sss.compute()
    sss.write_results()

    import matplotlib.pylab as plt
    plt.plot(sss.time, sss.u[50, :], color='b', label="Euler no damping")

    # damping parameters:
    f1 = 0.1
    d1 = 0.025
    f2 = 500
    d2 = 0.025
    # damping matrix
    damp_mat = 1 / 2 * np.array([[1 / (2 * np.pi * f1), 2 * np.pi * f1],
                                 [1 / (2 * np.pi * f2), 2 * np.pi * f2]])
    damp_qsi = np.array([d1, d2])
    # solution
    coefs = np.linalg.solve(damp_mat, damp_qsi)

    sss = SimpleSupportEulerWithDamping()
    sss.properties(20e6, 1e-4, 2000, 0.01, 20, -1000, coefs, np.linspace(0, 200, 1000))
    sss.compute()
    sss.write_results()

    plt.plot(sss.time, sss.u[50, :], color='k', label="Euler damping")

    sss = SimpleSupportTimoshenkoNoDamping()
    sss.properties(20e6, 10e5, 1/6, 1e-4, 2000, 0.01, 20, -1000, np.linspace(0, 200, 1000))
    sss.compute()
    sss.write_results()
    plt.plot(sss.time, sss.u[50, :], color='r', label="Timoshenko no damping")
    plt.grid()
    plt.legend()
    plt.xlabel("time [s]")
    plt.ylabel("displacement [m]")
    plt.show()
