# Signal processing functions.
"""
DOC for signal processing functions signal_proc.py
to call it:
    signal_proc.__doc__
    available functions:
        fft_sig - computes the Fast Fourier Transform
        int_sig - performs numerical integration
        filter_sig - filters signal
        spectral_subtraction - performs spectral subtraction of noise
"""


def fft_sig(sig, FS, nb_points=None):
    """
    DOC for the subroutine fft_sig
    DOC: 
    signal_proc.fft_sig.__doc__
    to call it:
        import signal_fcts
        signal_fcts.fft_sig(sig,Fs,npoints)
        sig     - signal to be processed
        Fs      - acquisition frequency
        npoints - nb points for the FFT (not compulsory)
                  if not specified nb_points=len(sig)
    """

    # import packages
    import numpy as np

    if not nb_points:  # if nb_points is empty
        nb_points = sig.shape[0]

    if nb_points % 2 == 0:  # if number is even
        nfft = nb_points
    else:
        nfft = nb_points + 1
        sig = np.append(sig, 0.)

    spec = np.fft.fft(sig, nfft) / len(sig[sig != 0.])
    ampl = np.abs(spec)
    phas = np.unwrap(np.angle(spec))

    freq = np.linspace(0, 1, nfft) * FS

    return freq, ampl, phas, spec


def int_sig(sig, tim, rule="trap", baseline=False, mov=False, hp=False, ini_cond=None, fpass=0.5, n=6):
    """
    DOC for the subroutine int_sig inside signal_fcts.py
    DOC: 
    signal_fcts.int_sig.__doc__
    to call it:
    import signal_fcts 
    signal_proc.fft_sig(sig, tim, Fs, rule, baseline, mov, hp)
        sig      - signal to be processed
        tim      - time
        rule     - integration rule ("trap"=trapezoidal) Idea is to extend with other methods
                   default is "trap"
        baseline - perform 2nd degree baseline correction
                   default is False
        mov      - mean average correction
                   default is False
        hp       - high pass filter at 0.5Hz
                   default is False
        ini_cond - initial conditions. if exists the initial that it is passed is added as initial
                   default is None
        fpass    - cut off frequency.
                   default is 0.5
        n        - order of the filter
                   default is 6
    """

    # import packages
    import sys
    import numpy as np
    from scipy import integrate

    # mean average correction
    if mov:
        sig = sig - np.mean(sig)

    # integration rule
    if rule == "trap":  # trapezoidal rule
        sigI = integrate.cumtrapz(sig, tim, initial=ini_cond)
    else:
        sys.exit('Integration rule not defined')

    # baseline correction
    if baseline:
        fit = np.polyfit(tim, sigI, 2)
        fit_int = np.polyval(fit, tim)
        sigI = sigI - fit_int

    # high pass filter
    if hp:
        Fs = np.ceil(1. / np.mean(np.diff(tim)))
        sigI = filter_sig(sigI, Fs, fpass, n, type="highpass")

    return sigI


def filter_sig(sig, Fs, Fpass, N, type="lowpass", rp=0.01, rs=60):
    """
    DOC for the subroutine highpass_sig inside signal_fcts.py
    DOC: 
    signal_fcts.high_pass.__doc__
    to call it:
    import signal_fcts 
    signal_proc.highpass_sig(sig, Fpass, n)
        sig      - signal to be processed
        Fs       - acquisition frequency
        Fpass    - cut off frequency [Hz]
        N        - order of the filter
        type     - type of the filter (lowpass or highpass)
                   default is lowpass
        rp       - maximum ripple allowed below unity gain in the passband. Specified in decibels, as a positive number
                   default is 0.01
        rs       - minimum attenuation required in the stop band. Specified in decibels, as a positive number
                   default is 60
    """

    import numpy as np
    from scipy import signal  # import subpackages

    z, p, k = signal.ellip(N, rp, rs, Fpass / (Fs / 2), btype=type, output='zpk')
    sos = signal.zpk2sos(z, p, k)

    sig = signal.sosfilt(sos, sig)
    sig = sig[::-1]
    sig = signal.sosfilt(sos, sig)
    sig = sig[::-1]

    return sig


def spectral_subtraction(sig, noise, Fs, M=128):
    """
    DOC for the subroutine spectral subtraction signal_fcts.py
    DOC:
    signal_fcts.spectral_subtraction.__doc__

    to call it:
    import signal_fcts
    signal_proc.spectral_subtraction(sig, Fpass, n)
        sig      - signal to be processed
        noise    - noise
        Fs       - acquisition frequency
        M        - number of samples of window
    """
    import numpy as np
    from scipy import signal

    # Hamming window
    window = signal.hann(M, False)

    # create average noise spectrum
    # number of blocks in noise
    number_cycles = np.trunc(len(noise) / M)
    # noise spectrum initialisation
    noise_spectrum = []

    # considering half overlap
    for i in range(int(number_cycles * 2 - 1)):
        idx_i = i * M / 2
        # omega[int(idx_i):int(idx_i + M)] += window
        noise_w = window * noise[int(idx_i):int(idx_i + M)]
        noise_f, noise_a, _, _ = fft_sig(noise_w, Fs)
        noise_spectrum.append(noise_a)

    # average noise spectrum
    noise_amplitude = np.mean(noise_spectrum, axis=0)

    # remove noise from signal
    # number of blocks in signal
    number_cycles = np.ceil(len(sig) / M)
    # signal initialisation
    sig_corr = np.zeros(int((np.ceil(len(sig) / M) * M)))
    # auxiliar signal with the same size as the windowed signal
    sig_aux = np.zeros(int((np.ceil(len(sig) / M) * M)))
    sig_aux[:len(sig)] = sig

    # round off for end of the file. otherwise window goes over the end
    if int(number_cycles) % 2 == 0:
        round_off = 1
    else:
        round_off = 2

    # considering half overlap
    for i in range(int(number_cycles * 2 - 1) - round_off):
        idx_i = i * M / 2
        # window signal
        signal_w = window * sig_aux[int(idx_i): int(idx_i + M)]
        # fft signal
        sig_f, sig_a, sig_p, _ = fft_sig(signal_w, Fs)
        # spectral subtraction
        sig_a_corr = sig_a - noise_amplitude
        # corrections according to Boll
        # if signal amplitude < 0 or signal amplitude < mean noise amplitude -> signal amplitude = 0
        sig_a_corr[sig_a_corr < 0] = 0
        sig_a_corr[sig_a_corr < np.mean(noise_amplitude)] = 0

        spectrum = np.fft.ifft(sig_a_corr * np.exp(1j * sig_p), M)
        sig_corr[int(idx_i):int(idx_i + M)] += np.real(spectrum) * M

    sig_corr = sig_corr[:len(sig)]

    return sig_corr
