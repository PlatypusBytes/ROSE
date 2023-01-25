import numpy as np
from scipy.signal import welch, butter, filtfilt


BANDS = {"one-third": [[.08, .10],
                       [.10, .126],
                       [.126, .16],
                       [.16, .20],
                       [.20, .253],
                       [.253, .32],
                       [.32, .40],
                       [.40, .50],
                       [.50, .63]],
         }


class HrsmMax:
    def __init__(self, signal: np.ndarray, dx: float, convert_m2mm: bool = True) -> object:
        """
        Computes the H rms and H max according to description of Level Accoustics report


        @param signal: signal to be processed
        @param dx: sampling frequency
        @param convert_m2mm: (optional: default True) converts the signal from m to mm
        @rtype: object
        """

        self.signal = signal  # signal
        self.dx = dx  # sampling frequency

        # setting for the processing
        self.DXmaxFast = 1  #
        self.nb_fft_min = 256  # minimum number of samples for the power spectral density
        self.derivative = [0, 0, 0, 2, 2, 2, 2, 2, 2]  # number of times that each frequency band is derived

        self.Hmax = []  # maximum
        self.Hrms = []  # rms of the maximum
        self.Pxx = []  # power spectral density
        self.frequency = []  # frequency power spectral density
        # RMS of the square root of the power spectral density
        self.rms_bands = np.zeros(len(BANDS["one-third"]))
        # maximum effective value over the entire signal
        self.max_fast = np.zeros(len(BANDS["one-third"]))
        # maximum effective value over the length Dx
        self.max_fast_Dx = np.zeros(len(BANDS["one-third"]))
        # frequency bands
        self.frequency_bands = np.mean([np.array(BANDS["one-third"])[:, 0]*(2 ** (1 / 6)),
                                        np.array(BANDS["one-third"])[:, 1]/(2 ** (1 / 6))], axis=0)

        # convert signal from m to mm
        if convert_m2mm:
            self.signal *= 1000

        # compute power spectral density
        self.power_spectral_density()
        # compute rsm psd
        self.rms_effective()
        #
        self.effective_values()

    def power_spectral_density(self):
        """
        Computes power spectral density following Welch's overlapped segment averaging estimator
        """

        # minimum number of samples for FFT
        n_fft = int(np.max([2 ** (np.ceil(np.log2(len(self.signal)))), self.nb_fft_min]))

        # Pwelch
        self.frequency, self.Pxx = welch(self.signal, fs=1/self.dx,
                                         window="hamming", nperseg=len(self.signal), nfft=n_fft, scaling='density', detrend=False)

    def rms_effective(self):
        """
        Computes RMS square root of power spectral density

        """
        # frequency step
        delta_f = self.frequency[1] - self.frequency[0]

        # compute the rms value at each frequency band
        for i, band in enumerate(BANDS["one-third"]):
            # find indexes where the bands exist
            idx = np.where((self.frequency >= band[0]) & (self.frequency < band[1]))[0]
            self.Pxx[idx] = (2 * np.pi * self.frequency[idx]) ** (2 * self.derivative[i]) * self.Pxx[idx]
            self.rms_bands[i] = np.sqrt(np.sum(self.Pxx[idx] * delta_f))

    def effective_values(self, n: int = 4, tau: int = 2, order: int = 3):
        """

        @param n:(optional, default = 4) number of time constants
        @param tau: (optional, default = 2) time constant
        @param order: (optional, default = 3) Butterworth filter order
        """
        fout = 1 / (1 - np.exp(-n))

        for i, band in enumerate(BANDS["one-third"]):
            derivative_value = self.derivative[i]
            b, a = butter(order, np.array(band) * 2 * self.dx, output="ba", btype="bandpass")
            new_signal = filtfilt(b, a, self.signal, padtype="odd", padlen=3*(max(len(b),len(a))-1))

            while derivative_value != 0:
                new_signal = np.diff(new_signal) / self.dx
                derivative_value -= 1

            ksi = np.linspace(0, n * tau, int(n * tau / self.dx + 1))
            g = fout * np.exp(-ksi / tau)

            convoluted_signal = np.sqrt(np.convolve(new_signal**2, g) * self.dx / tau)
            self.max_fast[i] = np.max(convoluted_signal)
            idx = np.floor((len(self.signal) - np.floor(self.DXmaxFast / self.dx)) / 2) + \
                  np.linspace(0, np.floor(self.DXmaxFast / self.dx)-1, int(np.floor(self.DXmaxFast / self.dx)))

            self.max_fast_Dx[i] = np.max(convoluted_signal[idx.astype(int)])
