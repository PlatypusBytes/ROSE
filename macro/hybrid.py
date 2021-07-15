from rose.utils import LayeredHalfSpace
import numpy as np


class Macro:
    """
    """
    def __init__(self, base, V0, H0, M0, V_cyc, omega, layers_file, total_nb_cycles, **parameters):

        self.base = base  # length of foundation

        # variables
        self.M0 = M0
        self.V0 = V0
        self.H0 = H0
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
        self.forces = []
        self.elastic_stiffness_matrix = []
        self.plastic_stiffness_matrix = []
        self.u = np.zeros(int(self.nb_cycles))

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

        self.forces = np.array([self.Qv, self.Qh, self.Qm])

    def force_max(self):

        # ToDo: Aron
        # following DFoundation (EC7)
        # self.Vmax = c * Nc + q * Nq + 0.5 * gamma * self.base * Ngamma
        self.V_max = 500

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
        Kv = np.real(data.K_dyn)

        # horizontal stiffness
        self.layers_file[0][-1] = "H"
        data = LayeredHalfSpace.Layers(self.layers_file)
        data.assign_properties()
        data.correction_incompressible()
        data.static_cone()
        data.dynamic_stiffness(self.omega)
        Kh = np.real(data.K_dyn)

        # rotational stiffness
        ktheta = 1 # ToDo: extend wolf

        self.elastic_stiffness_matrix = np.array([[Kv, 0, 0],
                                                  [0, Kh, 0],
                                                  [0, 0, ktheta]])

        return

    def plastic_matrix(self):

        # derived from Maxima: correct elastoplastic matrix
        Kv = self.elastic_stiffness_matrix[0, 0][0]
        Kh = self.elastic_stiffness_matrix[1, 1][0]
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
        force_comp = self.V0 + self.V_cyc

        for n in range(1, self.nb_cycles):

            # elastic
            u_e = 1 / self.elastic_stiffness_matrix[0, 0] * (self.V0 + self.V_cyc)

            # compaction
            u_comp = force_comp / self.alpha * (1 / n) ** self.beta

            # viscoplastic
            norm_v_cyc = 0
            for i in range(nb_steps):
                norm_v_cyc += i * Qv_cyc_inc

                # check if elastic / plastic
                elastic = self.yield_function(self.Qv + norm_v_cyc, self.Qh, self.Qm)
                if elastic:
                    u_vp = 0
                else:
                    u_vp = (self.elastic_stiffness_matrix[0, 0] - self.plastic_stiffness_matrix[0, 0]) ** -1 * (self.V0 + self.V_cyc)

            # plastic multiplier creep
            # ToDo: improve distance
            distance = self.V_max - self.V0
            if distance <= 0:
                print("Warning point starts in plasticity")
            plastic_mult = self.zeta * (self.V_cyc / distance) ** self.iota

            self.u[n] = u_comp + (u_e + u_vp) * plastic_mult

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
    lay = layers[0][1]

    V0 = 50
    H0 = 10
    M0 = 30
    V_cyc = 200
    # model parameters
    param = {"alpha": 250000,
             "beta": -0.15,
             "zeta": 100,
             "iota": 1e-4}
    m = Macro(1, V0, H0, M0, V_cyc,  2 * np.pi, lay, 200, **param)
    m.main()

    import matplotlib.pylab as plt
    plt.plot(range(m.nb_cycles), m.u)
    plt.grid()
    plt.xlabel("Number of cycles [-]")
    plt.xlabel("Vertical displacement [m]")
    plt.xlim(left=0)
    plt.show()
