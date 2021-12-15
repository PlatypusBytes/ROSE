import numpy as np
import matplotlib.pylab as plt
# from SignalProcessing import signal_tools


class RailIrregularities:
    def __init__(self, x: np.ndarray,
                 f_min: float = 2, f_max: float = 500, N: int = 2000, Av: float = 0.000003365, omega_c: float = 0.8242):
        """
        Creates rail unevenness following :cite: `zhang_2001`.

        A summary of default values can be found in : cite: `Podworna_2015`.

        Parameters
        ----------
        @param x: position of the node
        @param f_min: (default 2 Hz) minimum frequency for the PSD of the unevenness
        @param f_max: (default 500 Hz) maximum frequency for the PSD of the unevenness
        @param N: (default 2000) number of frequency increments
        @param Av: (default 0.003365 m rad/m) vertical track irregularity parameters
        @param omega_c: (default 0.8242 rad/m) critical wave number
        """

        # default parameters
        self.Av = Av
        self.omega_c = omega_c

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
            phi = np.random.uniform(0, 2 * np.pi)
            self.irregularities += np.sqrt(4 * self.spectral(omega_n) * delta_omega) * np.cos(omega_n * x - phi)

        # # compute spectrum
        # sig = signal_tools.Signal(x, self.irregularities)
        # sig.psd()
        # self.frequency_spectrum = sig.frequency_Pxx
        # self.spectrum = sig.Pxx

        return

    def spectral(self, omega):
        spectral_unevenness = 2 * np.pi * self.Av * self.omega_c ** 2 / ((omega ** 2 + self.omega_c ** 2) * omega ** 2)
        return spectral_unevenness


if __name__ == "__main__":
    distance = np.linspace(0, 50, 50)
    r = RailIrregularities(distance)
    plt.plot(distance, r.irregularities)
    plt.show()
