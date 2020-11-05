import sys
import os
import matplotlib.pylab as plt
import numpy as np
np.seterr(all="ignore")


class Layers:
    """ Based on Foundation vibration analysis: A strength of materials approach
        Wolf & Deeks
        """

    def __init__(self, data):
        # create variables
        self.name = []
        self.G = []
        self.thickness = []
        self.qsi = []
        self.rho = []
        self.nu = []
        self.c = []
        self.cp = []
        self.cs = []
        self.z0_r = []
        self.delta_M = []  # trapped mass
        self.P = []
        self.u0 = 1.  # initial displacement
        self.amplitude = []
        self.Gn = []
        self.K_dyn = []
        self.static_stiff = []

        # assign data
        for idx, layer in enumerate(data):
            if layer[0] != 'Force':
                # read data for soil layers
                self.name.append(layer[0])
                self.G.append(float(layer[1]))
                self.nu.append(float(layer[2]))
                self.rho.append(float(layer[3]))
                self.qsi.append(float(layer[4]))
                self.amplitude.append([])
                try:
                    self.thickness.append(float(layer[5]))
                except ValueError:
                    self.thickness.append("halfspace")
            elif layer[0] == 'Force':
                # read force layer
                self.radius = float(layer[6])
                self.direction = str(layer[7])
                self.load_idx = idx + 1

                self.name.append(layer[0])
                self.thickness.append(np.nan)
                self.G.append(np.nan)
                self.nu.append(np.nan)
                self.rho.append(np.nan)
                self.qsi.append(np.nan)
                self.amplitude.append([])
        return

    def assign_properties(self):

        for i in range(len(self.name)):
            G_star = self.G[i] * (1. + 2. * 1j * self.qsi[i])
            Ec = (1. - self.nu[i]) / (1. - 2. * self.nu[i]) * 2 * G_star

            self.cp.append(np.sqrt(Ec / self.rho[i]))
            self.cs.append(np.sqrt(G_star / self.rho[i]))

            if self.direction == "V":
                self.c.append(self.cp[-1])
            elif self.direction == "H":
                self.c.append(self.cs[-1])
            else:
                sys.exit("direction is not valid. must be V or H")

        return

    def dynamic_stiffness(self, omega):

        # generate variable amplitude
        for i in range(len(self.amplitude)):
            self.amplitude[i] = np.zeros(len(omega)).astype(np.complex128)

        self.amplitude[self.load_idx - 1] = np.ones(len(omega)) * (self.u0 + 1j * 0.)

        # wave moving downwards and not halfspace
        if self.load_idx <= len(self.name):
            dir = 1  # direction down
            self.transmit(dir, self.load_idx, self.radius, self.u0, omega)

        # wave moving upwards and not halfspace
        if self.load_idx > 1:
            dir = -1  # direction up
            self.transmit(dir, self.load_idx - 1, self.radius, self.u0, omega)

        # force applied to the disk
        self.stiff_cone(omega)

        # Green's function at the disk
        self.Gn = self.amplitude[self.load_idx - 1] / self.P[self.load_idx]

        # Dynamic stiffness
        self.K_dyn = 1. / self.Gn

        return

    def transmit(self, direction, index, radius, u0, omega):

        if direction == -1:  # up
            idx_layer_A = index + 1
            idx_layer_B = index
        else:  # down
            idx_layer_A = index
            idx_layer_B = index + 1

        radius_new = radius + self.thickness[idx_layer_A] / self.z0_r[idx_layer_A]

        # function f
        # pg 56 Eq: 4.5
        f = radius / radius_new * np.exp(-1j * omega / self.c[idx_layer_A] * self.thickness[idx_layer_A]) * u0

        if np.linalg.norm(f) >= 0.0001:
            # function g
            # pg 59 Eq: 4.19
            g = -self.alpha(idx_layer_A, idx_layer_B, radius_new, omega) * f
            # function h
            # pg 57 Eq: 4.10
            h = f + g
            # amplitude of displacement is the summation of h
            # pg 57 Eq: 4.10
            self.amplitude[index] += h

            # reflected wave - changes direction
            self.transmit(-direction, index - direction, radius_new, g, omega)
            if self.thickness[idx_layer_B] != np.inf:
                self.transmit(direction, index + direction, radius_new, h, omega)

        return

    def alpha(self, idx_A, idx_B, rad, omega):

        # alpha
        # pg 58 Eq: 4.16-4.18
        beta_A = self.rho[idx_A] * self.c[idx_A]**2 * (1. / (self.z0_r[idx_A] * rad) +
                                                       1j * omega / self.c[idx_A])

        beta_B = self.rho[idx_B] * self.c[idx_B]**2 * (1. / (self.z0_r[idx_B] * rad) +
                                                       1j * omega / self.c[idx_B])

        if self.name[idx_A] == 'Force':
            beta_A = len(omega) * [0.]
        elif self.name[idx_B] == 'Force':
            beta_B = len(omega) * [0.]

        alpha = - (beta_A - beta_B) / (beta_A + beta_B)

        return alpha

    def static_cone(self):

        # compute static cone apex
        for i in range(len(self.name)):
            if self.direction == "V":
                # compute static cone apex - vertical
                # pg. 33 Eq: 3.28
                z0_r0 = np.pi / 4. * (1. - self.nu[i]) * (self.cp[i] / self.cs[i]) ** 2
            elif self.direction == "H":
                # compute static cone apex - horizontal
                # pg. 34 Eq: 3.31
                z0_r0 = np.pi / 8 * (2. - self.nu[i])

            self.z0_r.append(z0_r0)
        return

    def correction_incompressible(self):

        # correction for incompressible solids
        # the compression wave, as poisson gets to 0.5 tends to infinite
        # pg. 44 section: 3.4
        # only for vertical direction
        for i in range(len(self.name)):
            if float(self.nu[i]) >= 1 / 3. and self.direction == "V":
                # maximum compression speed = 2 * shear speed
                self.cp[i] = 2. * self.cs[i]
                self.c[i] = 2. * self.cs[i]

                # trapped mass
                # pg. 44 Eq: 3.84
                self.delta_M.append(2.4 * (self.nu[i] - 1. / 3.) * self.rho[i] * np.pi * self.radius ** 3.)
            else:
                self.delta_M.append(0.)
        return

    def stiff_cone(self, omega):

        # dynamic load
        # pg 32 Eq: 3.16 (trapped mass pg. 45 Eq: 3.87)
        for i in range(len(self.name)):
            self.P.append(self.rho[i] * self.c[i] ** 2 * np.pi * self.radius ** 2 / (self.z0_r[i] * self.radius) +
                          1j * omega * self.rho[i] * self.c[i] * np.pi * self.radius ** 2 +
                          - omega ** 2 * self.delta_M[i])

            self.static_stiff.append(self.rho[i] * self.c[i] ** 2 * np.pi * self.radius ** 2 / (self.z0_r[i] * self.radius))
        return


def read_file(file_name):

    if not os.path.isfile(file_name):
        sys.exit("Layers file name does not exist.")

    with open(file_name, 'r') as f:
        data = f.read().splitlines()

    lines = []
    for i in data[1:]:
        lines.append([j for j in i.split(';')])

    return lines


def write_output(path_results, name, data, omega, freq):

    if not os.path.isdir(path_results):
        os.makedirs(path_results)

    # tmp variables for plot
    stiff_tmp = []
    damp_tmp = []

    # write table => frequency ; complex stiffness (real; imaginary)
    with open(os.path.join(path_results, "Kdyn_" + str(name) + ".csv"), "w") as f:
        f.write("omega [rad/s];Complex dynamic stiffness [N/m2];Stiffness [N/m2];Damping [Ns/m2]\n")
        for i in range(len(omega)):
            f.write(str(omega[i]) + ";" +
                    str(data.K_dyn[i]) + ";" +
                    str(np.real(data.K_dyn[i])) + ";" +
                    str(np.imag(data.K_dyn[i]) / (omega[i] * data.radius / np.real(data.cs[1]))) + "\n")
            # add to tmp variables
            stiff_tmp.append(np.real(data.K_dyn[i]))
            damp_tmp.append(np.imag(data.K_dyn[i]) / (omega[i] * data.radius / np.real(data.cs[1])))

    # make plots
    if freq:
        # if frequency
        create_plot(omega / 2. / np.pi, stiff_tmp, damp_tmp, "Frequency [Hz]", path_results, name)
    else:
        # if angular frequency
        create_plot(omega, stiff_tmp, damp_tmp, r"$\omega$ [rad/s]", path_results, name)

    return


def create_plot(frequency, stiff, damp, label_x, path_results, name):

    # create figure
    fig, ax = plt.subplots(1, 2, figsize=(8.5, 3.5))
    ax[0].set_position([0.13, 0.13, 0.35, 0.80])
    ax[1].set_position([0.60, 0.13, 0.35, 0.80])
    plt.rcParams.update({'font.size': 10})

    # plot stiffness
    ax[0].grid()
    ax[0].plot(frequency, stiff)
    ax[0].set_xlabel(label_x)
    ax[0].set_ylabel(r'K$_{dyn}$ [N/m]')
    ax[0].set_xlim((frequency[0], frequency[-1]))
    # ax[0].set_ylim(bottom=0)

    # plot damping
    ax[1].grid()
    ax[1].plot(frequency, damp)
    ax[1].set_xlabel(label_x)
    ax[1].set_ylabel(r'Damping [Ns/m]')
    ax[1].set_xlim((frequency[0], frequency[-1]))
    # ax[1].set_ylim(bottom=0)

    # save fig
    plt.savefig(os.path.join(path_results, str(name) + ".png"))
    plt.savefig(os.path.join(path_results, str(name) + ".pdf"))
    plt.close()
    return

# Validation with Wolf & Deeks pg: 108
# parameters:
# Force      -       -            -     -     -    1  V/H
# Layer1     1000.0  0.25         1800  0.05  1    -  -
# Layer2     500.0   0.3          1800  0.05  0.5  -  -
# halfspace  200.0   0.333333333  1602  0.05  inf  -  -
