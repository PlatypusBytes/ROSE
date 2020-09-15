import numpy as np
import json
from scipy.integrate import trapz


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
                force = self.eig_shape[int(len(self.x) / 2)] * self.force

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
        self.mass = []  # unit mass of the beam
        self.length = []  # length of the beam
        self.force = []  # force
        self.time = []  # time
        self.x = []  # discretisation of the beam

        self.alpha_n = []  # alpha value for the load

        self.u = []  # vertical displacement

        self.result = {}  # dictionary for json dump
        return

    def properties(self, E, I, rho, A, L, F, damping, time):
        """
        Assigns properties

        :param E: Young modulus
        :param I: Inertia
        :param rho: Density
        :param A: Area
        :param L: Length
        :param F: Force
        :param qsi: damping
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
        :return: eigen frequency
        """
        return n**2 * np.pi**2 * np.sqrt(self.EI / (self.mass * self.length**4))

    def alpha(self):
        """
        Computes alpha for the *n* mode
        """

        self.alpha_n = np.zeros(self.n)
        # alpha is 1 for n = 1, 5, 9, ...
        for i in range(1, self.n, 4):
            self.alpha_n[i] = 1
        # alpha is -1 for n = 3, 7, 11, ...
        for i in range(3, self.n, 4):
            self.alpha_n[i] = -1
        return

    def compute(self):
        """
        Computes the displacement solution
        """
        # computes alpha
        self.alpha()

        # for each time step
        for idx, t in enumerate(self.time):
            aux = np.zeros(len(self.x))
            # for the desired number of modes
            for n in range(1, self.n):
                aux += self.alpha_n[n] / n**4 * (1 - np.cos(self.eigen_freq(n) * t)) * np.sin(n * np.pi * self.x / self.length)
            # add to displacement
            self.u[:, idx] = 2 * self.force * self.length**3 / (np.pi**4 * self.EI) * aux

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


if __name__ == "__main__":

    sss = SimpleSupportEulerNoDamping()
    sss.properties(20e6, 1, 3, 1, 10, -1000, np.linspace(0, 1, 1000))
    sss.compute()
    sss.write_results()

    import matplotlib.pylab as plt
    plt.plot(sss.time, sss.u[50, :], marker='o')
    plt.show()
