import numpy as np
import matplotlib.pylab as plt
# from SignalProcessing import signal_tools


class WheelFlat:
    """
   Class containing a wheel flat.

   :Attributes:
       - :self.irregularities:     np array of the irregularities at each position
   """
    def __init__(self, x: np.ndarray, wheel_diameter: float, flatness_length: float, start_position=None, seed=14):
        """
        Creates an array with a wheel flat. where every circumference of the wheel, the wheel is indented.
        Adapted from :cite:`Uzzal_2013`.

        :param x: position of the node [m]
        :param wheel_diameter: diameter of the wheel [m]
        :param flatness_length: total flat length of the wheel [m]
        :param start_position: optional starting position of where a wheel flat is known
        :param seed: (default 14) seed for random generator
        """

        # random generator
        random_generator = np.random.default_rng(seed)

        # calculate wheel circumference
        wheel_circumference = np.pi * wheel_diameter

        # calculate half of the angle between the flat part and the new wheel radius at wheel flat
        half_angle_wheel_flat = np.arcsin(flatness_length / wheel_diameter)

        # calculate new wheel radius at wheel flat
        new_radius = (wheel_diameter / 2) * np.cos(half_angle_wheel_flat)

        # calculate height of the wheel flat, with respect to the bottom
        irregularity_height = wheel_diameter / 2 - new_radius

        # Apply a random start distance if no start position is chosen, else the wheel flat starts at the start
        # position, note that also prior wheel rotations are taken into account
        if start_position is None:
            random_start = random_generator.random(1)[0] * wheel_circumference
            x = x + random_start
        else:
            x = x - start_position

        # an array is made which contains a value if the wheel has made a round, else the array contains zeros
        n_circumferences = x / wheel_circumference

        # create array which indicates the amount of rotations of the wheel as an integer
        tmp_array = np.ceil(n_circumferences).astype(int)

        # wheel flat index is located at every new wheel rotation
        wheel_flat_indices = []
        for i in range(1, len(tmp_array) - 1):
            if (tmp_array[i] > tmp_array[i - 1]):
                wheel_flat_indices.append(i)
        wheel_flat_indices = np.array(wheel_flat_indices)

        # generate irregularities array
        self.irregularities = np.zeros_like(x)
        self.irregularities[wheel_flat_indices] = -irregularity_height


class RailIrregularities:
    """
    Class containing rail unevenness following :cite:`zhang_2001`.

    :Attributes:
        - :self.Av:                 Vertical track irregularity parameter
        - :self.omega_c:            critical wave angular frequency
        - :self.omega:              wave angular frequency
        - :self.irregularities:     np array of the irregularities at each position
    """
    def __init__(self, x: np.ndarray,
                 f_min: float = 2, f_max: float = 500, N: int = 2000, Av: float = 0.00002095, omega_c: float = 0.8242, seed=99):
        """
        Creates rail unevenness following :cite:`zhang_2001`.

        A summary of default values can be found in :cite:`Podworna_2015`.

        Parameters
        ----------
        :param x: position of the node
        :param f_min: (default 2 Hz) minimum frequency for the PSD of the unevenness
        :param f_max: (default 500 Hz) maximum frequency for the PSD of the unevenness
        :param N: (default 2000) number of frequency increments
        :param Av: (default 0.00002095 m2 rad/m) vertical track irregularity parameters
        :param omega_c: (default 0.8242 rad/m) critical wave number
        :param seed: (default 99) seed for random generator
        """

        # default parameters
        self.Av = Av
        self.omega_c = omega_c
        # random generator
        random_generator = np.random.default_rng(seed)

        # define omega range
        omega_max = 2 * np.pi * f_max
        omega_min = 2 * np.pi * f_min
        delta_omega = (omega_max - omega_min) / N
        self.omega = np.linspace(omega_min, omega_max, N)

        # irregularities
        self.irregularities = np.zeros(len(x))

        # for each frequency increment
        for n in range(N):
            omega_n = omega_min + delta_omega * n
            phi = random_generator.uniform(0, 2 * np.pi)
            self.irregularities += np.sqrt(4 * self.spectral(omega_n) * delta_omega) * np.cos(omega_n * x - phi)

        # # compute spectrum
        # sig = signal_tools.Signal(x, self.irregularities)
        # sig.psd()
        # self.frequency_spectrum = sig.frequency_Pxx
        # self.spectrum = sig.Pxx

        return

    def spectral(self, omega):
        """
        Computes spectral unevenness

        :param omega: angular frequency rad/s
        :return:
        """
        spectral_unevenness = 2 * np.pi * self.Av * self.omega_c ** 2 / ((omega ** 2 + self.omega_c ** 2) * omega ** 2)
        return spectral_unevenness


if __name__ == "__main__":
    distance = np.linspace(0, 50, 50)
    for i in range(10):
        r = RailIrregularities(distance, seed=i)
        plt.plot(distance, r.irregularities)
    plt.show()
