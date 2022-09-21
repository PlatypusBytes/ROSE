import sys
import numpy as np
from scipy.integrate import trapz
import os
import pickle
from tqdm import tqdm


def train_info(data, trains, time_days):
    nb_nodes = []
    # determine number of loading cycles
    for t in trains:
        data.trains.append(t)
        # compute number of cycles
        aux = trains[t]["nb-per-hour"] * trains[t]["nb-hours"] * trains[t]["nb-axles"]
        # number axles a day
        data.nb_cycles_day.append(aux)
        # number cycles of train
        data.number_cycles.append(aux * time_days)
        # force for train t
        data.force.append(trains[t]["forces"])
        # nb of nodes
        nb_nodes.append(len(trains[t]["forces"]))

    # number of trains
    data.number_trains = len(data.trains)
    # define cumulative time
    data.cumulative_time = np.linspace(0, time_days - 1, int(np.max(data.number_cycles)))
    # define cumulative nb cycles
    data.cumulative_nb_cycles = np.linspace(0, int(np.max(data.number_cycles)) - 1, int(np.max(data.number_cycles)))
    # index for distributed loading
    for nb in data.number_cycles:
        data.index_cumulative_distributed.append(
            np.linspace(0, nb - 1, int(np.max(data.number_cycles))) * (np.max(data.number_cycles) / nb))
    data.index_cumulative_distributed = np.array(data.index_cumulative_distributed).astype(int)

    # check if number of nodes is the same for all the trains
    if all(nb_nodes[0] == x for x in nb_nodes):
        data.nb_nodes = nb_nodes[0]
    else:
        sys.exit("Error: number of nodes is not the same for the different trains")

    return data


class Results:
    def __init__(self):
        self.time = []
        self.displacement = []
        self.nodes = []


class BaseModel:
    def __init__(self):
        self.trains = []
        self.number_trains = []  # number of trains
        self.trains = []  # name of train types
        self.number_cycles = []  # number of total loading cycles
        self.nb_cycles_day = []  # number of loading cycles / day
        self.force = []
        self.cumulative_time = []
        self.cumulative_nb_cycles = []
        self.index_cumulative_distributed = []
        self.nodes = []
        self.results = Results()


class Varandas(BaseModel):
    def __init__(self, alpha: float = 0.6, beta: float = 0.82, gamma: float = 10, N0: float = 1e6, F0: float = 50):
        """
        Initialisation of the accumulation model of Varandas :cite:`varandas_2014`

        Parameters
        ----------
        :param alpha: (optional, default 0.6) dependency of settlement with loading amplitude
        :param beta: (optional, default 0.82) controls progression of settlement with number of load cycles
        :param gamma: (optional, default 10) accumulated settlement in reference test (with F0, N0)
        :param N0: (optional, default 1e6) reference number of cycles
        :param F0: (optional, default 50) reference load amplitude
        """
        super().__init__()

        # material parameters
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        # model parameters
        self.N0 = N0
        self.F0 = F0

        # M alpha beta
        summation = [(1 / n) ** self.beta for n in range(1, int(self.N0))]
        self.M_alpha_beta = self.F0 ** (self.alpha + 1) / (self.alpha + 1) * np.sum(summation)

        # variables
        self.displacement = []
        self.nb_nodes = []  # number of nodes
        self.histogram_force = []  # force histogram
        self.force_max = []  # maximum force / train

        # numerical parameters
        self.nb_int_step = 100  # number of integration steps
        self.force_scl_fct = 1000  # N -> kN
        self.disp_scl_fct = 1000  # mm -> m
        return

    def read_traffic(self, trains: dict, time_days: int):
        """
        Reads the train traffic information

        Parameters
        ----------
        :param trains: Dictionary with train information
        :param time_days: Time in days of the analysis
        """
        # read train info
        train_info(self, trains, time_days)

    def settlement(self, idx: list = None):
        """
        Computes cumulative settlement following the methodology proposed by Varandas :cite:`varandas_2014`.

        The settlement :math:`S` of sleeper :math:`N` follows:

        .. math::
            S_{N} = \sum_{n=1}^{N} u_{p, n}

        where :math:`u_{p, n}` is:

        .. math::
            u_{p, n} = \frac{\gamma}{M_{\alpha \beta}} \int_{0}^{\bar{F_{n}}} F^{\alpha} \left( \frac{1}{h(F) + 1} \right)^{\beta} dF

        and :math:`M_{\alpha\beta}`:

        .. math::
            M_{\alpha\beta} = \frac{F_{0}^{\alpha + 1}}{\alpha + 1} \sum_{n=1}^{N_{0}} \left(\frac{1}{n}\right)^{\beta}

        :math:`h(F)` corresponds to the load histogram.


        Parameters
        ----------
        :param idx: (optional, default None) node to compute the calculations.
                    if None computes the calculations for all nodes
        """

        # if index is None compute for all nodes
        if not idx:
            idx = range(int(self.nb_nodes))

        # assign nodes
        self.nodes = idx

        # cumulative displacement
        self.displacement = np.zeros((int(len(idx)), int(np.max(self.number_cycles))))
        # displacement due to cycle n
        disp = np.zeros((int(len(idx)), int(np.max(self.number_cycles))))

        # compute maximum force
        for j in range(self.number_trains):
            # histogram.extend(self.number_cycles[j] * [np.max(np.abs(self.force[j]), axis=1) / self.force_scl_fct])
            self.force_max.append(np.max(np.abs(self.force[j]), axis=1)[idx] / self.force_scl_fct)
        self.force_max = np.array(self.force_max)
        # Vector F for integration
        F = np.linspace(0, np.max(self.force_max, axis=0), self.nb_int_step)

        print("Running Varandas model")
        # progress bar
        pbar = tqdm(
            total=len(self.cumulative_nb_cycles),
            unit_scale=True,
            unit="steps",
        )

        h_f = np.zeros(len(self.nodes))
        max_val_force = np.zeros((len(self.nodes), self.number_trains)).T

        for n, nb_cyc in enumerate(self.cumulative_nb_cycles):
            for tr in range(self.number_trains):
                if nb_cyc <= self.number_cycles[tr]:
                    h_f[self.force_max[tr, :] <= max_val_force[tr, :]] += 1
                    max_val_force[tr, self.force_max[tr, :] > max_val_force[tr, :]] = self.force_max[tr, self.force_max[tr, :] > max_val_force[tr, :]]

                    # compute integral: trapezoidal rule
                    integral = F ** self.alpha * (1 / (h_f + 1)) ** self.beta
                    val = trapz(integral, F, axis=0)
                    # compute displacement on cycle N
                    disp[:, n] += self.gamma / self.M_alpha_beta * val
                    # disp[:, self.index_cumulative_distributed[tr][n]] += self.gamma / self.M_alpha_beta * val
            # update progress bar
            pbar.update(1)

        # close progress bar
        pbar.close()
        # compute displacements
        self.displacement = np.cumsum(disp, axis=1) / self.disp_scl_fct
        # create results dic
        self.create_results()
        return

    def create_results(self):
        """
        Creates the results dictionary
        """
        # collect displacements
        aux = []
        for i, _ in enumerate(self.nodes):
            aux.append(self.displacement[i].tolist())

        # create results struct
        self.results.nodes = list(self.nodes)
        self.results.time = self.cumulative_time.tolist()
        self.results.displacement = aux
        return

    def dump(self, output_file: str):
        """
        Writes results to json file

        :param output_file: filename of the results
        """

        # check if path to output file exists. if not creates
        if not os.path.isdir(os.path.split(output_file)[0]):
            os.makedirs(os.path.split(output_file)[0])

        # dump dict
        with open(output_file, "wb") as f:
            pickle.dump(self.results, f)
        return


class LiSelig(BaseModel):
    def __init__(self, t_ini=0, last_layer_thickness=10):
        r"""
        Accumulation model for soil layer. Based on Li and Selig :cite:`Li_Selig_1996`.
        Implementation based on Punetha et al. :cite:`Punetha_2020`.
        The model has been improved with :cite:`Charoenwong_2022`.

        @param t_ini: (optional) initial time (default 0)
        @param last_layer_thickness: (optional) last layer thickness
        """
        super().__init__()
        # soil classes according to Li & Selig
        self.other = ["a", "ht"]
        self.sand = ["zg", "zm", "zf"]
        self.silt = ["z&s", "zs", "s"]
        self.silt_plas = ["z&h", "zk", "k&s", "z&k"]
        self.clay_low = ["kz", "k", "sd"]
        self.clay_high = ["ko", "k&v", "vk", "v", "o&z"]

        # variables
        self.name = []  # layer name
        self.gamma = []  # volumetric weight
        self.phi = []  # friction angle
        self.cohesion = []  # cohesion
        self.z_coord = []  # coordinate
        self.nb_soils = []
        self.nb_nodes = []  # number of nodes
        self.z_last_layer = last_layer_thickness

        self.a = []  # Li and Selig parameter
        self.b = []  # Li and Selig parameter
        self.m = []  # Li and Selig parameter
        self.soil_id = []  # ID of the soil
        self.thickness = []  # Layer thickness
        self.z_middle = []  # Z coordinate middle layer
        self.sigma_s = []  # Static strength
        self.sigma_v0 = []  # Initial effective vertical stress
        self.sigma_deviatoric = []  # Deviatoric stress
        self.settlement = []  # total settlement
        self.force_scl_fct = 1000  # N -> kN
        self.t_ini = t_ini

    def read_traffic(self, trains: dict, time_days: int):
        """
        Reads the train traffic information

        Parameters
        ----------
        :param trains: Dictionary with train information
        :param time_days: Time in days of the analysis
        """
        # read train info
        train_info(self, trains, time_days)

    def read_SoS(self, soil_sos, soil_id):
        """
        Parses data from SOS json file into the structure for the Li & Selig model.

        @param soil_sos: SOS dictionary
        @param soil_id: ID of each node
        """

        self.nb_soils = len(soil_sos)
        self.soil_id = soil_id
        for s in range(self.nb_soils):
            self.name.append(soil_sos[s]['soil_layers']["soil_name"])
            self.gamma.append(np.array(soil_sos[s]['soil_layers']["gamma_wet"]))
            self.phi.append(np.array(np.array(soil_sos[s]['soil_layers']["friction_angle"])) * (np.pi / 180))  # friction angle in rads
            self.cohesion.append(np.array(soil_sos[s]['soil_layers']["cohesion"]))  # cohesion
            self.z_coord.append(np.array(soil_sos[s]['soil_layers']['top_level']))  # layer thickness

    def initial_stress(self):
        """
        Computes initial vertical effective stress at the middle of the layer
        """
        for i in range(self.nb_soils):
            thickness = np.abs(np.diff(self.z_coord[i]))
            self.thickness.append(np.append(thickness, self.z_last_layer))

            self.z_middle = np.abs((self.z_coord[i] - self.z_coord[i][0])) + self.thickness[i] / 2
            self.sigma_v0.append(self.gamma[i] * self.z_middle[i])

    def classify(self):
        """
        Parameterise soil layers for the Li and Selig model following the SOS name convention.
        """
        for i in range(self.nb_soils):
            a = []
            b = []
            m = []
            for name in self.name[i]:
                if name.split("_")[-1] in self.sand:
                    a.append(0.64)
                    b.append(0.10)
                    m.append(1.7)
                elif name.split("_")[-1] in self.silt:
                    a.append(0.64)
                    b.append(0.10)
                    m.append(1.7)
                elif name.split("_")[-1] in self.silt_plas:
                    a.append(0.84)
                    b.append(0.13)
                    m.append(2.0)
                elif name.split("_")[-1] in self.clay_low:
                    a.append(1.1)
                    b.append(0.16)
                    m.append(2.0)
                elif name.split("_")[-1] in self.clay_high:
                    a.append(1.2)
                    b.append(0.18)
                    m.append(2.4)
                elif name.split("_")[-1] in self.other:
                    a.append(0)
                    b.append(0)
                    m.append(0)
                else:
                    sys.exit(f"ERROR: Soil layer {name} not defined.")
            self.a.append(a)
            self.b.append(b)
            self.m.append(m)

    def strength(self):
        """
        Computes shear strength resistance, assuming MC failure
        """
        for i in range(self.nb_soils):
            self.sigma_s.append(self.sigma_v0[i] * np.tan(self.phi[i]) + self.cohesion[i])

    def dev_stress(self, width, length):
        """
        Computes deviatoric stress based on analytical solution from Flamant (see Verruijt 2018 pg 231-232).
        ToDo: This can be improved for a layered soil.

        @param width: width of the strip load
        @param length: length of the stress distribution
        """

        self.sigma_deviatoric = np.zeros((len(self.nodes), len(self.z_middle), len(self.force)))

        for f, force in enumerate(self.force):
            # Flamant's approximation
            stress = np.max(np.abs(force), axis=1) / (length * width) / self.force_scl_fct
            a = width / 2
            x = np.linspace(-10 * a, 10 * a, 100)

            # for each layer
            for i, z_mid in enumerate(self.z_middle):
                for k, nod in enumerate(self.nodes):
                    theta1 = np.arctan((x + a) / z_mid)
                    theta2 = np.arctan((x - a) / z_mid)

                    stress_zz = stress[nod] / np.pi * ((theta1 - theta2) + np.sin(theta1) * np.cos(theta1) - np.sin(theta2) * np.cos(theta2))
                    stress_xx = stress[nod] / np.pi * ((theta1 - theta2) - np.sin(theta1) * np.cos(theta1) + np.sin(theta2) * np.cos(theta2))
                    stress_xz = stress[nod] / np.pi * (np.cos(theta2)**2 - np.cos(theta1)**2)
                    # principal stress directions
                    sigma_1 = (stress_xx + stress_zz) / 2 + np.sqrt(((stress_xx - stress_zz) / 2)**2 + stress_xz**2)
                    sigma_2 = (stress_xx + stress_zz) / 2 - np.sqrt(((stress_xx - stress_zz) / 2)**2 + stress_xz**2)
                    self.sigma_deviatoric[k, i, f] = np.max((sigma_1 - sigma_2)/2)

    def calculate(self, width, length, idx=None):
        """
        Calculate the settlement

        @param width: width of the stress distribution
        @param length: length of the stress distribution
        """

        # if index is None compute for all nodes
        if not idx:
            idx = range(int(self.nb_nodes))

        # assign nodes
        self.nodes = idx

        # progress bar
        print("Running Li & Selig model")
        pbar = tqdm(
            total=len(self.nodes),
            unit_scale=True,
            unit="steps",
        )

        # parameterise settlement model
        self.classify()
        # compute initial vertical stress
        self.initial_stress()
        # compute strength
        self.strength()
        # deviatoric stress
        self.dev_stress(width, length)
        # strain
        self.settlement = np.zeros((len(self.nodes), len(self.cumulative_nb_cycles)))

        for k, val in enumerate(self.nodes):
            # id soil for the node
            id_s = self.soil_id[val]
            for t in range(len(self.trains)):
                # N = np.linspace(1 + self.t_ini, self.number_cycles[t], len(self.cumulative_nb_cycles))
                # new version from David
                N = np.linspace(1 + np.max(self.number_cycles) * np.max((self.cumulative_time) / 365) * self.t_ini,
                                np.max(self.number_cycles) * (np.max(self.cumulative_time) / 365) * self.t_ini +
                                self.number_cycles[t],
                                len(self.cumulative_nb_cycles))
                for i in range(len(self.thickness[id_s])):
                    # # basic model
                    # strain = self.a[id_s][i] * (self.sigma_deviatoric[k, i, t] / self.sigma_s[id_s][i]) ** self.m[id_s][i] * N ** self.b[id_s][i]
                    strain = self.a[id_s][i] * (self.sigma_deviatoric[k, i, t] / self.sigma_s[id_s][i]) ** self.m[id_s][i] * N ** self.b[id_s][i]
                    self.settlement[k, :] += strain * self.thickness[id_s][i]
            pbar.update()
            self.settlement[k, :] -= self.settlement[k, 0]
        pbar.close()
        self.create_results()

    def create_results(self):
        """
        Creates the results dictionary
        """
        # collect displacements
        aux = []
        for i, _ in enumerate(self.nodes):
            aux.append(self.settlement[i].tolist())

        # create results struct
        self.results.nodes = list(self.nodes)
        self.results.time = self.cumulative_time.tolist()
        self.results.displacement = aux

    def dump(self, output_file: str):
        """
        Writes results to json file

        :param output_file: filename of the results
        """

        # check if path to output file exists. if not creates
        if not os.path.isdir(os.path.split(output_file)[0]):
            os.makedirs(os.path.split(output_file)[0])

        # dump dict
        with open(output_file, "wb") as f:
            pickle.dump(self.results, f)
        return
