import numpy as np
from scipy.optimize import root
from scipy.integrate import trapz
import json


class PulseLoadNoDamping:
    """
    Analytical solution for a pulse load in the top of a cantilever beam (Euler beam).
    No Damping.

    """
    def __init__(self, n=100, ele_size=0.1, int_steps=100):
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

        self.a = []  # parameter a for the n mode
        self.eig = []  # eigen frequency n mode
        self.eig_shape = []  # eigen shape n mode

        self.int_steps = int_steps  # steps for numerical integration of the mode shapes

        self.alpha_n = []  # alpha value for the load

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
        Computes eigen *n* frequency for a cantilever beam

        :param n: n mode
        :return: eigen frequency
        """

        # initial guess
        a_L = (2 * n - 1) * np.pi / 2

        # find coordinates that minimize boundary function
        res = root(self.find_eigen, x0=a_L, tol=1e-16)
        self.a = res.x[0] / self.length
        self.eig = (self.a * self.length)**2 * np.sqrt(self.EI / (self.mass * self.length**4))
        return self.eig

    def mode_shape(self):
        """
        Computes eigen *n* shape for a cantilever beam
        """
        self.eig_shape = np.cos(self.a * self.x) - np.cosh(self.a * self.x) - \
                         (np.cos(self.a * self.length) + np.cosh(self.a * self.length)) / (np.sin(self.a * self.length) + np.sinh(self.a * self.length)) * \
                         (np.sin(self.a * self.x) - np.sinh(self.a * self.x))
        return

    def mass_modal(self):
        """
        Computes modal mass for n mode
        """
        x = np.linspace(0, self.length, int(self.int_steps))
        aux = np.cos(self.a * x) - np.cosh(self.a * x) - \
              (np.cos(self.a * self.length) + np.cosh(self.a * self.length)) / (np.sin(self.a * self.length) + np.sinh(self.a * self.length)) * \
              (np.sin(self.a * x) - np.sinh(self.a * x))

        return self.mass * trapz(aux ** 2, x)

    def compute(self):
        """
        Computes the displacement solution
        """
        self.alpha_n = 0
        # for each time step
        for idx, t in enumerate(self.time):
            aux = np.zeros(len(self.x))
            # for the desired number of modes
            for n in range(1, self.n):
                # eigen frequency
                self.eigen_freq(n)
                # eigen mode
                self.mode_shape()
                # modal mass
                mass = self.mass_modal()
                # modal stiffness
                stiff = self.eig ** 2 * mass

                # compute force
                force = self.eig_shape[-1] * self.force

                # compute summation of solution
                aux += force / stiff * (1 - np.cos(self.eig * t)) * self.eig_shape

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

    @staticmethod
    def find_eigen(a_L):
        return 1 + np.cos(a_L) * np.cosh(a_L)


if __name__ == "__main__":
    ss = PulseLoadNoDamping(ele_size=0.01)
    ss.properties(20e6, 1e-4, 2000, 0.01, 1, -1000, np.linspace(0, 5, 2001))
    ss.compute()
    ss.write_results()

    import matplotlib.pylab as plt
    plt.plot(ss.time, ss.u[-1, :], marker="o")
    plt.show()
