from rose.utils import LayeredHalfSpace
import numpy as np


class Macro:
    """
    """
    def __init__(self, base, V0, H0, M0, V_cyc, omega, layers_file, total_nb_cycles, **parameters):

        self.base = base  # length of foundation

        # variables
        self.V0 = V0
        self.H0 = H0
        self.M0 = M0
        self.V_cyc = V_cyc
        self.omega = np.array([omega])  # frequency
        self.layers_file = layers_file  # file with soil layers

        # normalised variables
        self.Qv = []
        self.Qh = []
        self.Qm = []
        self.Qcyc = []

        # parameters
        self.V_max = []  # Maximum bearing capacity
        self.Qh_max = []
        self.Qm_max = []

        # variables
        self.nb_cycles = total_nb_cycles
        self.force = []
        self.force_norm = []
        self.elastic_stiffness_matrix = []
        self.plastic_stiffness_matrix = []
        self.u = np.zeros((int(self.nb_cycles), 3))

        # functions
        self.f = []  # yield function

        # numerical parameters
        self.sub_step = 0.01  # sub-step for discretisation of the stress increment

        # model parameters
        self.alpha = parameters["alpha"]
        self.beta = parameters["beta"]
        self.zeta = parameters["zeta"]
        self.iota = parameters["iota"]
        return

    def normalise(self):
        self.Qv = self.V0 / self.V_max
        self.Qh = self.H0 / self.V_max
        self.Qm = self.M0 / (self.base * self.V_max)
        self.Qcyc = self.V_cyc / self.V_max

        self.force = np.array([self.V0, self.H0, self.M0])
        self.force_norm = np.array([self.Qv, self.Qh, self.Qm])

    def force_max(self):
        """
        Bearing capacity according to Brich Hansen
        :return:
        """

        # ToDo: Aron
        # following DFoundation (EC7)
        phi_mean = 30 # weighted mean friction angle in effective zone below foundation
        c_mean = 2 # weighted mean cohesion in effective zone below foundation
        q = 0 # effective pressure of the soil at the same height as the foundation base
        gamma_mean = 20 # weighted mean volumetric weight in effective zone below foundation

        N_q = np.exp(np.pi * np.tan(np.radians(phi_mean))) * np.tan(np.radians(45 + 0.5*phi_mean))**2
        N_gamma = 2 * (N_q - 1)*np.tan(np.radians(phi_mean))
        N_c = (N_q-1)/np.tan(np.radians(phi_mean))

        length_sleeper = 2.6 # length sleeper  [m]

        sigma_max = c_mean * N_c + q * N_q + 0.5 * gamma_mean * self.base * N_gamma
        self.V_max = sigma_max * self.base # * length_sleeper
        # self.V_max = 1000

        # A MACROELEMENT FORMULATION FOR SHALLOW FOUNDATIONS 911
        # CHATZIGOGOS ET AL. (2011)
        # table 3
        self.Qh_max = 0.125
        self.Qm_max = 0.119

        return

    def yield_function(self, Qv, Qh, Qm):
        # CHATZIGOGOS ET AL. (2011)
        self.f = Qv ** 2 + (Qm / self.Qm_max) ** 2 + (Qh / self.Qh_max) ** 2 - 1
        if self.f < 0:
            return True
        return False

    def elastic_matrix(self):
        # wolf

        # vertical stiffness
        data = LayeredHalfSpace.Layers(self.layers_file)
        data.assign_properties()
        data.correction_incompressible()
        data.static_cone()
        data.dynamic_stiffness(self.omega)
        Kv = np.real(data.K_dyn)[0]

        # horizontal stiffness
        self.layers_file[0][-1] = "H"
        data = LayeredHalfSpace.Layers(self.layers_file)
        data.assign_properties()
        data.correction_incompressible()
        data.static_cone()
        data.dynamic_stiffness(self.omega)
        Kh = np.real(data.K_dyn)[0]

        # rotational stiffness
        ktheta = 1 # ToDo: extend wolf

        self.elastic_stiffness_matrix = np.array([[Kv, 0, 0],
                                                  [0, Kh, 0],
                                                  [0, 0, ktheta]])

        return

    def plastic_matrix(self):

        # derived from Maxima: correct elastoplastic matrix
        Kv = self.elastic_stiffness_matrix[0, 0]
        Kh = self.elastic_stiffness_matrix[1, 1]
        Kt = self.elastic_stiffness_matrix[2, 2]

        self.plastic_stiffness_matrix = np.array([[(4 * Kv ** 2 * self.Qv ** 2) / (4 * Kv * self.Qv ** 2 + (4 * Kt * self.Qm ** 2) / self.Qm_max ** 4 + (4 * Kh * self.Qh ** 2) / self.Qh_max ** 4),
                                                   (4 * Kh * Kv * self.Qh * self.Qv) / (self.Qh_max ** 2 * (4 * Kv * self.Qv ** 2 + (4 * Kt * self.Qm ** 2) / self.Qm_max ** 4 + (4 * Kh * self.Qh ** 2) / self.Qh_max ** 4)),
                                                   (4 * Kt * Kv * self.Qm * self.Qv) / (self.Qm_max ** 2 * (4 * Kv * self.Qv ** 2 + (4 * Kt * self.Qm ** 2) / self.Qm_max ** 4 + (4 * Kh * self.Qh ** 2) / self.Qh_max ** 4))],
                                                  [(4 * Kh * Kv * self.Qh * self.Qv) / (self.Qh_max ** 2 * (4 * Kv * self.Qv ** 2 + (4 * Kt * self.Qm ** 2) / self.Qm_max ** 4 + (4 * Kh * self.Qh ** 2) / self.Qh_max ** 4)),
                                                   (4 * Kh ** 2 * self.Qh ** 2) / (self.Qh_max ** 4 * (4 * Kv * self.Qv ** 2 + (4 * Kt * self.Qm ** 2) / self.Qm_max ** 4 + (4 * Kh * self.Qh ** 2) / self.Qh_max ** 4)),
                                                   (4 * Kh * Kt * self.Qh * self.Qm) / (self.Qh_max ** 2 * self.Qm_max ** 2 * (4 * Kv * self.Qv ** 2 + (4 * Kt * self.Qm ** 2) / self.Qm_max ** 4 + (4 * Kh * self.Qh ** 2) / self.Qh_max ** 4))
                                                  ],
                                                  [(4 * Kt * Kv * self.Qm * self.Qv) / (self.Qm_max ** 2 * (4 * Kv * self.Qv ** 2 + (4 * Kt * self.Qm ** 2) / self.Qm_max ** 4 + (4 * Kh * self.Qh ** 2) / self.Qh_max ** 4)),
                                                   (4 * Kh * Kt * self.Qh * self.Qm) / (self.Qh_max ** 2 * self.Qm_max ** 2 * (4 * Kv * self.Qv ** 2 + (4 * Kt * self.Qm ** 2) / self.Qm_max ** 4 + (4 * Kh * self.Qh ** 2) / self.Qh_max ** 4)),
                                                   (4 * Kt ** 2 * self.Qm ** 2) / (self.Qm_max ** 4 * (4 * Kv * self.Qv ** 2 + (4 * Kt * self.Qm ** 2) / self.Qm_max ** 4 + (4 * Kh * self.Qh ** 2) / self.Qh_max ** 4))]
                                                  ])

        return

    def elastic_plastic(self):

        # number of steps for viscoplastic-plastic
        nb_steps = int(np.ceil(self.Qcyc / self.sub_step))
        Qv_cyc_inc = self.Qcyc / nb_steps # increment of V_cyc

        # force for compaction
        force_comp = (self.force + np.array([self.V_cyc, 0, 0]))

        for n in range(1, self.nb_cycles):

            # elastic
            u_e = np.dot(np.linalg.inv(self.elastic_stiffness_matrix), force_comp)

            # compaction
            u_comp = force_comp / self.alpha * (1 / n) ** self.beta

            # viscoplastic
            norm_v_cyc = 0
            for i in range(nb_steps):
                norm_v_cyc = i * Qv_cyc_inc

                # check if elastic / plastic
                elastic = self.yield_function(self.Qv + norm_v_cyc, self.Qh, self.Qm)
                if elastic:
                    u_vp = np.zeros(3)
                else:
                    u_vp = np.dot(np.linalg.inv(self.elastic_stiffness_matrix - self.plastic_stiffness_matrix), force_comp)

            # plastic multiplier creep
            # ToDo: improve distance
            distance = self.V_max - self.V0
            if distance <= 0:
                print("Warning point starts in plasticity")
            plastic_mult = self.zeta * (self.V_cyc / distance) ** self.iota

            self.u[n, :] = u_comp + (u_e + u_vp) * plastic_mult

        return

    def main(self):

        # compute maximum
        self.force_max()
        # normalise variables
        self.normalise()
        # compute elastic stiffness matrix
        self.elastic_matrix()
        # compute plastic stiffness matrix
        self.plastic_matrix()
        # elastic-plastic deformation
        self.elastic_plastic()

        return


if __name__ == "__main__":
    from run_rose import run_wolf

    E = 100e6
    v = 0.2
    emb = ["embankment", E / (2 * (1 + v)), v, 2000, 0.05, 1]
    layers = run_wolf.read_file(r"../run_rose/SOS/SOS.json", emb)
    lay = layers[2][1]


    b = 0.25  # width sleeper
    omega = (20 / 140 / 3.6) * 2 * np.pi
    V0 = 50
    H0 = 0
    M0 = 0
    V_cyc = 200
    nb_cycles = 2000
    # model parameters
    param = {"alpha": 2500,
             "beta": -0.05,
             "zeta": 100,
             "iota": 1e-4}
    import time
    t_ini = time.time()
    m = Macro(b, V0, H0, M0, V_cyc,  omega, lay, nb_cycles, **param)
    m.main()
    print(f"time: {time.time() - t_ini} s")
    import matplotlib.pylab as plt
    plt.plot(range(m.nb_cycles), m.u[:, 0])
    # plt.plot(range(m.nb_cycles), m.u[:, 1])
    # plt.plot(range(m.nb_cycles), m.u[:, 2])
    plt.grid()
    plt.xlabel("Number of cycles [-]")
    plt.ylabel("Vertical displacement [m]")
    plt.xlim(left=0)
    plt.show()
