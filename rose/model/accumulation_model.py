import os
import pickle
from typing import Union, List
from abc import ABC, abstractmethod
import numpy as np
from scipy.integrate import trapezoid as trapz
from scipy.interpolate import interp1d
from scipy.signal import find_peaks
from tqdm import tqdm


class ReadTrainInfo:
    """
    Read train info and parse it to data structure
    """
    def __init__(self, trains: dict, start_time: int, end_time: int, steps: int = 1):
        """
        Initialise the train information

        :param trains: trains dictionary containing traffic information
        :param start_time: start time of analysis in days
        :param end_time: time of analysis in days
        :param steps: step interval to save results
        """
        self.trains_info = trains
        self.start_time = start_time
        self.end_time = end_time
        self.steps = steps
        self.trains_name = []
        self.nb_cycles_day = []
        self.number_cycles = []
        self.force = []
        self.number_trains = []
        self.cumulative_time = []
        self.cumulative_nb_cycles = []
        self.steps_index = []
        self.index_cumulative_distributed = []
        self.__train_info()

    def __train_info(self):
        """
        Read train info and parse it to data structure
        """
        # determine number of loading cycles
        for t in self.trains_info:
            self.trains_name.append(t)
            # compute number of cycles
            aux = self.trains_info[t]["nb-per-hour"] * self.trains_info[t]["nb-hours"] * self.trains_info[t]["nb-axles"]
            # number axles a day
            self.nb_cycles_day.append(aux)
            # number cycles of train
            self.number_cycles.append(aux * (self.end_time - self.start_time))
            # force for train t
            self.force.append(self.trains_info[t]["forces"])

        # number of trains
        self.number_trains = len(self.trains_name)
        # define cumulative time
        self.cumulative_time = np.linspace(self.start_time, self.end_time, int(np.max(self.number_cycles) / self.steps))
        # define cumulative nb cycles
        self.cumulative_nb_cycles = np.linspace(self.start_time, int(np.max(self.number_cycles)) - 1, int(np.max(self.number_cycles)))
        # index to save results
        self.steps_index = np.linspace(0, int(np.max(self.number_cycles)) - 1, int(np.max(self.number_cycles) / self.steps)).astype(int)

        # index for distributed loading
        for nb in self.number_cycles:
            self.index_cumulative_distributed.append(np.linspace(0, nb - 1, int(np.max(self.number_cycles))) * (np.max(self.number_cycles) / self.steps))
        self.index_cumulative_distributed = np.array(self.index_cumulative_distributed).astype(int)


class AccumulationModel_abc(ABC):
    """
    Abstract class for accumulation models
    """
    @abstractmethod
    def settlement():
        """
        Abstract method to compute the cumulative settlement
        """
        raise Exception("It is not allowed to call the AccumulationModel abstract method")


class Nasrollahi(AccumulationModel_abc):
    def __init__(self, alpha_k: float, beta_k: float, gamma: float, F0: float=1000,
                 threshold_force_inf: float=90e3, threshold_force_zero: float=35e3, reference_nb_load_cycles: float=1e5,
                 nb_samples_peak: int=15):
        """
        Initialisation of the accumulation model of Nasrollahi et al. :cite:`Nasrollahi_2023`.

        Parameters
        ----------
        :param alpha_k: (optional, default 0.6) dependency of settlement with loading amplitude
        :param beta_k: (optional, default 0.82) controls progression of settlement with number of load cycles
        :param gamma: (optional, default 10) accumulated settlement in reference test (with F0, N0)
        :param F0: (optional, default 50) reference load amplitude
        :param threshold_force_inf: (optional, default 90e3) threshold force for infinite settlement
        :param threshold_force_zero: (optional, default 35e3) threshold force for zero settlement
        :param reference_nb_load_cycles: (optional, default 10**5) reference number of load cycles for the Macaulay brackets
        :param nb_samples_peak: (optional, default 15) number of samples for peak detection
        """
        self.alpha_k = alpha_k
        self.beta_k = beta_k
        self.gamma = gamma
        self.F0 = F0
        self.threshold_force_inf = threshold_force_inf
        self.threshold_force_zero = threshold_force_zero
        self.threshold_force = []
        self.reference_nb_load_cycles = reference_nb_load_cycles
        self.max_allowed_displacement_iter = 0.2 / 1000
        self.nb_samples_peak = nb_samples_peak
        self.previous_displacement = None
        self.total_nb_cycles = []


    def settlement(self, train: ReadTrainInfo, nb_nodes: int, idx: list = None, reload=False):
        """
        Computes cumulative settlement following the methodology proposed by Nasrollahi :cite:`Nasrollahi_2023`.

        The settlement :math:`S` of sleeper :math:`N` follows:

        .. math::
            S_{N} = \sum_{n=1}^{N} u_{p, n}

        where :math:`u_{p, n}` is:

        .. math::
            u_{p, n} = \\sum_{k=1}^{K} \\alpha_{k} \\left( \\frac{F_{k, n}}{F_{0}} \\right)^{\\beta_{k}}

        where :math:`K` is the number of load cycles.

        Parameters
        ----------
        :param train: train information object
        :param nb_nodes: number of nodes
        :param idx: (optional, default None) node to compute the calculations. \
                    if None computes the calculations for all nodes
        """

        # if train.number_trains > 1:
        #     raise ValueError("Error: The model Nasrollahi is not implemented for more than one train.")

        # in case of reloading read the previous stage
        if reload:
            previous_displacement = np.copy(self.previous_displacement)
            nb_cycles = np.max(self.total_nb_cycles)
            ini_nb_cycles = np.max(self.total_nb_cycles)
        else:
            self.threshold_force = np.ones((len(idx), train.number_trains)) * self.threshold_force_zero
            nb_cycles = 0
            ini_nb_cycles = 0

        # if index is None compute for all nodes
        if not idx:
            idx = range(int(nb_nodes))

        # assign nodes
        self.nodes = list(idx)

        # cumulative displacement
        self.displacement = np.zeros((int(len(idx)), int(np.max(train.number_cycles) / train.steps)))

        print("Running Kourosh model")
        pbar = tqdm(total=sum(train.number_cycles) - nb_cycles, unit_scale=True, unit="steps")

        perform_update = False

        # compute maximum force per wheel
        for j in range(train.number_trains):
            # compute number of iterations
            iterate = True

            max_force = np.max(np.abs(train.force[j]), axis=1)
            peaks = [find_peaks(np.abs(train.force[j][i, :]), height=0.8 * max_force[i], distance=self.nb_samples_peak) for i in idx]
            peak_forces = [p[1]["peak_heights"] for p in peaks]
            peak_forces = np.array(peak_forces)

            if reload:
                displacement = [previous_displacement]
                nb_cycles = self.total_nb_cycles[j]
            else:
                displacement = [np.zeros(len(idx))]
                nb_cycles = 0

            cycle_number = [nb_cycles]
            while iterate:
                # incremental displacement per wheel
                incremental = np.zeros(len(idx))
                # Macauly brackets
                aux = np.maximum(peak_forces - self.threshold_force[:, j][:, np.newaxis], 0)
                # compute incremental displacement
                incremental = np.sum(self.alpha_k * (aux / self.F0) ** self.beta_k, axis=1) / 1000

                # check if displacement is below the maximum allowed
                maximum_incremental = np.max(incremental)

                # check if displacement is below the maximum allowed and if dynamic analysis is needed
                if maximum_incremental > self.max_allowed_displacement_iter:
                    perform_update = True

                incremental[incremental > self.max_allowed_displacement_iter] = self.max_allowed_displacement_iter

                if maximum_incremental == 0:
                    maximum_incremental = 1e-12
                # perform  interpolation of number of cycles
                update_nb_cycles = int(np.ceil(self.max_allowed_displacement_iter * self.reference_nb_load_cycles / maximum_incremental))
                nb_cycles += update_nb_cycles

                # update threshold force
                self._update_threshold_force(displacement[-1] + incremental, j)

                # check if the number of cycles is below the maximum allowed and below the number of cycles of the train
                if (nb_cycles >= self.reference_nb_load_cycles) and (nb_cycles >= train.number_cycles[j]):
                    incremental = incremental * np.min([self.reference_nb_load_cycles, train.number_cycles[j]]) / nb_cycles
                    nb_cycles = np.min([self.reference_nb_load_cycles, train.number_cycles[j]]) + ini_nb_cycles
                    update_nb_cycles = np.min([self.reference_nb_load_cycles, train.number_cycles[j]]) - cycle_number[-1]  # to update progress bar
                    cycle_number.append(nb_cycles)
                    perform_update = False
                    if train.number_cycles[j] <= self.reference_nb_load_cycles:
                        iterate = False
                # check if the number of cycles is below the maximum allowed
                elif nb_cycles >= self.reference_nb_load_cycles:
                    # trim the displacement to the self.nb_load_cycles
                    incremental = incremental * self.reference_nb_load_cycles / nb_cycles
                    nb_cycles = self.reference_nb_load_cycles + ini_nb_cycles
                    update_nb_cycles = self.reference_nb_load_cycles - cycle_number[-1]  # to update progress bar
                    cycle_number.append(nb_cycles)
                    perform_update = False
                # check if the number of cycles is below the number of cycles of the train
                elif nb_cycles >= train.number_cycles[j]:
                    # trim the displacement to the self.nb_load_cycles
                    incremental = incremental * train.number_cycles[j] / nb_cycles
                    nb_cycles = train.number_cycles[j] + ini_nb_cycles
                    update_nb_cycles = train.number_cycles[j] - cycle_number[-1]  # to update progress bar
                    cycle_number.append(nb_cycles)
                    iterate = False
                    perform_update = False

                if perform_update:
                    #ToDo # re-run dynamic train-track model
                    cycle_number.append(nb_cycles)
                    run_model = True
                    # iterate = False
                    # return iterate

                # cycle_number.append(nb_cycles)
                displacement.append(displacement[-1] + incremental)
                pbar.update(update_nb_cycles)

            pbar.close()

            self.total_nb_cycles.append(nb_cycles)

            # interpolate displacement for the number of cycles
            disp = np.zeros((len(idx), int(max(train.number_cycles) / train.steps)))
            # Convert the list of displacements to a properly shaped array for vectorized interpolation
            displacement_array = np.array(displacement)
            # Create a single interpolation function for all nodes at once (axis=0 interpolates along the first dimension)
            f = interp1d(cycle_number, displacement_array, axis=0)
            # Apply the interpolation function to get values for all nodes at each time step
            disp = f(np.linspace(ini_nb_cycles, train.number_cycles[j]+ini_nb_cycles, int(max(train.number_cycles) / train.steps))).T

            # add displacement to previous
            self.displacement = self.displacement + disp

            previous_displacement = np.zeros(len(idx))

        self.previous_displacement = self.displacement[:, -1]

    def _update_threshold_force(self, displacement: np.ndarray, idx: int):
        """
        Update threshold force

        :param displacement: displacement vector
        """
        self.threshold_force[:, idx] = self.threshold_force_inf - (self.threshold_force_inf - self.threshold_force_zero) *\
                                       np.exp(-self.gamma * displacement * 1000)


class Varandas(AccumulationModel_abc):
    def __init__(self, alpha: float = 0.6, beta: float = 0.82, gamma: float = 10, N0: float = 1e6, F0: float = 50):
        """
        Initialisation of the accumulation model of Varandas :cite:`varandas_2014`.

        Parameters
        ----------
        :param alpha: (optional, default 0.6) dependency of settlement with loading amplitude
        :param beta: (optional, default 0.82) controls progression of settlement with number of load cycles
        :param gamma: (optional, default 10) accumulated settlement in reference test (with F0, N0)
        :param N0: (optional, default 1e6) reference number of cycles
        :param F0: (optional, default 50) reference load amplitude
        """
        #ToDo: improve load distribution accross time

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
        self.nb_nodes = []  # number of nodes
        self.histogram_force = []  # force histogram
        self.force_max = []  # maximum force / train
        self.max_val_force = []  # maximum value force / train
        self.h_f = []  # counter for the number of loads

        # numerical parameters
        self.nb_int_step = 100  # number of integration steps
        self.force_scl_fct = 1000  # N -> kN
        self.disp_scl_fct = 1000  # mm -> m


    def settlement(self, train: ReadTrainInfo, nb_nodes: int, idx: list = None, reload=False):
        """
        Computes cumulative settlement following the methodology proposed by Varandas :cite:`varandas_2014`.

        The settlement :math:`S` of sleeper :math:`N` follows:

        .. math::
            S_{N} = \sum_{n=1}^{N} u_{p, n}

        where :math:`u_{p, n}` is:

        .. math::
            u_{p, n} = \\frac{\\gamma}{M_{\\alpha \\beta}} \\int_{0}^{\\bar{F_{n}}} F^{\\alpha} \\left( \\frac{1}{h(F) + 1} \\right)^{\\beta} dF


        and :math:`M_{\\alpha\\beta}`:

        .. math::
            M_{\\alpha\\beta} = \\frac{F_{0}^{\\alpha + 1}}{\\alpha + 1} \\sum_{n=1}^{N_{0}} \\left(\\frac{1}{n}\\right)^{\\beta}

        :math:`h(F)` corresponds to the load histogram.


        Parameters
        ----------
        :param train: train information object
        :param nb_nodes: number of nodes
        :param idx: (optional, default None) node to compute the calculations. \
                    if None computes the calculations for all nodes
        """

        # in case of reloading read the previous stage
        if reload:
            previous_displacement = self.displacement[:, -1]

        # if index is None compute for all nodes
        if not idx:
            idx = range(int(nb_nodes))

        # assign nodes
        self.nodes = list(idx)

        # cumulative displacement
        self.displacement = np.zeros((int(len(idx)), int(np.max(train.number_cycles) / train.steps)))
        # displacement due to cycle n
        disp = np.zeros((int(len(idx)), int(np.max(train.number_cycles) / train.steps)))

        # compute maximum force
        force_max = []
        for j in range(train.number_trains):
            # histogram.extend(self.number_cycles[j] * [np.max(np.abs(self.force[j]), axis=1) / self.force_scl_fct])
            force_max.append(np.max(np.abs(train.force[j]), axis=1)[idx] / self.force_scl_fct)
        self.force_max = np.array(force_max)
        # Vector F for integration
        F = np.linspace(0, np.max(self.force_max, axis=0), self.nb_int_step)

        print("Running Varandas model")
        # initialise variables
        if not reload:
            self.h_f = np.zeros(len(self.nodes))
            max_val_force = np.zeros((len(self.nodes), train.number_trains)).T
        else:
            # in case of reloading
            # self.h_f = self.h_f
            max_val_force = self.max_val_force

        i = 0
        aux = np.zeros(len(self.nodes))
        for n, nb_cyc in enumerate(tqdm(train.cumulative_nb_cycles)):
            for tr in range(train.number_trains):
                if nb_cyc <= train.number_cycles[tr]:
                    self.h_f[self.force_max[tr, :] <= max_val_force[tr, :]] += 1
                    max_val_force[tr, self.force_max[tr, :] > max_val_force[tr, :]] = self.force_max[tr, self.force_max[tr, :] > max_val_force[tr, :]]

                    # compute integral: trapezoidal rule
                    integral = F ** self.alpha * (1 / (self.h_f + 1)) ** self.beta
                    val = trapz(integral, F, axis=0)

                    aux += self.gamma / self.M_alpha_beta * val

            if n in train.steps_index:
                # compute displacement on cycle N
                disp[:, i] = aux
                aux = np.zeros(len(self.nodes))
                i += 1

        # maximum force
        self.max_val_force = max_val_force
        # compute displacements
        self.displacement = np.cumsum(disp, axis=1) / self.disp_scl_fct
        # in case of reloading
        if reload:
            self.displacement = self.displacement + np.expand_dims(previous_displacement, axis=1)


class LiSelig(AccumulationModel_abc):
    def __init__(self, soil_sos: List[dict], soil_idx: List[int], width_stress: float, lenght_stress: float,
                 t_ini: int = 0, last_layer_depth: int = -20):
        r"""
        Accumulation model for soil layer. Based on Li and Selig :cite:`Li_Selig_1996`.
        Implementation based on Punetha et al. :cite:`Punetha_2020`.
        The model has been improved with :cite:`Charoenwong_2022`.

        :param soil_sos: SoS dictionary
        :param soil_idx: ID of each node for soil SoS
        :param width_stress: width of the stress distribution
        :param lenght_stress: length of the stress distribution
        :param t_ini: (optional, default 0) initial time from construction in years
        :param last_layer_depth: (optional, default -20) last layer depth
        """
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
        self.z_last_layer = last_layer_depth

        self.a = []  # Li and Selig parameter
        self.b = []  # Li and Selig parameter
        self.m = []  # Li and Selig parameter
        self.soil_id = []  # ID of the soil
        self.thickness = []  # Layer thickness
        self.z_middle = []  # Z coordinate middle layer
        self.sigma_s = []  # Static strength
        self.sigma_v0 = []  # Initial effective vertical stress
        self.sigma_deviatoric = []  # Deviatoric stress
        self.force_scl_fct = 1000  # N -> kN
        self.width_stress = width_stress
        self.length_stress = lenght_stress
        self.t_construction = t_ini
        self.n_ini = 0
        self.__read_SoS(soil_sos, soil_idx)


    def __read_SoS(self, soil_sos: dict, soil_id: list):
        """
        Parses data from SOS json file into the structure for the Li & Selig model.

        :param soil_sos: SOS dictionary
        :param soil_id: ID of each node
        """

        self.nb_soils = len(soil_sos)
        self.soil_id = soil_id
        for s in range(self.nb_soils):
            self.name.append(soil_sos[s]['soil_layers']["soil_name"])
            self.gamma.append(np.array(soil_sos[s]['soil_layers']["gamma_wet"]))
            self.phi.append(np.array(np.array(soil_sos[s]['soil_layers']["friction_angle"])) * (np.pi / 180))  # friction angle in rads
            self.cohesion.append(np.array(soil_sos[s]['soil_layers']["cohesion"]))  # cohesion
            self.z_coord.append(np.array(soil_sos[s]['soil_layers']['top_level']))  # layer thickness

    def __initial_stress(self):
        """
        Computes initial vertical effective stress at the middle of the layer
        """
        for i in range(self.nb_soils):
            thickness = np.abs(np.diff(np.append(self.z_coord[i], self.z_last_layer)))
            self.thickness.append(thickness)

            self.z_middle.append(np.abs((self.z_coord[i] - self.z_coord[i][0])) + self.thickness[i] / 2)
            self.sigma_v0.append(self.gamma[i] * self.z_middle[i])

    def __classify(self):
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
                    raise ValueError(f"ERROR: Soil layer {name} not defined.")
            self.a.append(a)
            self.b.append(b)
            self.m.append(m)

    def __strength(self):
        """
        Computes shear strength resistance, assuming MC failure
        """
        for i in range(self.nb_soils):
            self.sigma_s.append(self.sigma_v0[i] * np.tan(self.phi[i]) + self.cohesion[i])

    def __dev_stress(self, force_list: np.array):
        """
        Computes deviatoric stress based on analytical solution from Flamant (see :cite:`Verruijt_2018` pg. 231-232).
        ToDo: This can be improved for a layered soil.

        :param force_list: list of forces
        """

        for i in range(self.nb_soils):
            self.sigma_deviatoric.append(np.zeros((len(self.nodes), len(self.z_middle[i]), len(force_list))))

        for f, force in enumerate(force_list):
            # Flamant's approximation
            stress = np.max(np.abs(force), axis=1) / (self.length_stress * self.width_stress) / self.force_scl_fct
            a = self.width_stress / 2
            x = np.linspace(-10 * a, 10 * a, 100)

            # for each layer
            for k, nod in enumerate(self.nodes):
                id_soil = self.soil_id[nod]
                for i, z_mid in enumerate(self.z_middle[id_soil]):

                    theta1 = np.arctan((x + a) / z_mid)
                    theta2 = np.arctan((x - a) / z_mid)

                    stress_zz = stress[nod] / np.pi * ((theta1 - theta2) + np.sin(theta1) * np.cos(theta1) - np.sin(theta2) * np.cos(theta2))
                    stress_xx = stress[nod] / np.pi * ((theta1 - theta2) - np.sin(theta1) * np.cos(theta1) + np.sin(theta2) * np.cos(theta2))
                    stress_xz = stress[nod] / np.pi * (np.cos(theta2)**2 - np.cos(theta1)**2)
                    # principal stress directions
                    sigma_1 = (stress_xx + stress_zz) / 2 + np.sqrt(((stress_xx - stress_zz) / 2)**2 + stress_xz**2)
                    sigma_2 = (stress_xx + stress_zz) / 2 - np.sqrt(((stress_xx - stress_zz) / 2)**2 + stress_xz**2)
                    self.sigma_deviatoric[id_soil][k, i, f] = np.max((sigma_1 - sigma_2)/2)

    def settlement(self, train: ReadTrainInfo, nb_nodes: int, idx: list = None, reload=False):
        """
        Calculate the settlement

        :param train: train information object
        :param nb_nodes: number of nodes
        :param idx: (optional, default None) node to compute the calculations. \
                    if None computes the calculations for all nodes
        :param reload: (optional, default False) reload last stage
        """
        # in case of reloading read the previous stage
        if reload:
            previous_displacement = self.displacement[:, -1]
        else:
            self.n_ini = np.zeros((len(idx), train.number_trains)).tolist()

        # if index is None compute for all nodes
        if not idx:
            idx = range(int(nb_nodes))

        # assign nodes
        self.nodes = list(idx)

        # progress bar
        print("Running Li & Selig model")

        # parameterise settlement model
        self.__classify()
        # compute initial vertical stress
        self.__initial_stress()
        # compute strength
        self.__strength()
        # deviatoric stress
        self.__dev_stress(train.force)
        # strain
        self.displacement = np.zeros((len(self.nodes), len(train.cumulative_time)))

        n_ini = []
        for k, val in enumerate(tqdm(self.nodes)):
            # id soil for the node
            id_s = self.soil_id[val]
            aux = []
            for t in range(train.number_trains):
                # N = np.linspace(1 + t_ini, self.number_cycles[t], len(self.cumulative_nb_cycles))
                # new version from David
                N = np.linspace(1 + np.sum(train.nb_cycles_day) * 365 * self.t_construction + self.n_ini[k][t],
                                np.sum(train.nb_cycles_day) * 365 * self.t_construction +
                                train.number_cycles[t] + self.n_ini[k][t],
                                len(train.cumulative_time))
                aux.append(N[-1])
                for i in range(len(self.thickness[id_s])):
                    # # basic model
                    # strain = self.a[id_s][i] * (self.sigma_deviatoric[id_s][k, i, t] / self.sigma_s[id_s][i]) ** self.m[id_s][i] * N ** self.b[id_s][i]
                    strain = self.a[id_s][i] * (self.sigma_deviatoric[id_s][k, i, t] / self.sigma_s[id_s][i]) ** self.m[id_s][i] * N ** self.b[id_s][i]
                    self.displacement[k, :] = self.displacement[k, :] + strain * self.thickness[id_s][i]

            self.displacement[k, :] = self.displacement[k, :] - self.displacement[k, 0]
            n_ini.append(aux)

        self.n_ini = n_ini
        # in case of reloading
        if reload:
            self.displacement = self.displacement + np.expand_dims(previous_displacement, axis=1)

class AccumulationModel:
    r"""
    Accumulation model

    Computation of the cumulative settlement. Currently the following models are supported:
    - Varandas :cite:`varandas_2014`
    - Li & Selig :cite:`Li_Selig_1996`
    - Nasrollahi: :cite:`Nasrollahi_2023`

    """
    def __init__(self, accumulation_model: Union[Varandas, LiSelig, Nasrollahi], steps: int = 1):
        """
        Initialisation of the accumulation model

        :param accumulation_model: Accumulation model
        :param steps: step interval to save results
        """

        self.trains = []
        self.nb_nodes = []
        self.previous_stage = []
        self.displacement = []
        self.end_time = 0
        self.steps = steps
        self.results = {}
        self.reload = False
        self.previous_stage_results = {}

        self.accumulation_model = accumulation_model

    def read_traffic(self, trains: dict, end_time: int, start_time: int = 0):
        """
        Reads the train traffic information

        :param trains: Dictionary with train information
        :param end_time: Time in days of the analysis
        :param start_time: (optional, default 0) start time of analysis in days
        """

        nb_nodes = []
        # determine number of nodes
        for t in trains:
            nb_nodes.append(len(trains[t]["forces"]))

        # check if number of nodes is the same for all the trains
        if len(set(nb_nodes)) == 1:
            self.nb_nodes = nb_nodes[0]
        else:
            raise ValueError("Error: number of nodes is not the same for the different trains")

        # read train info
        self.trains = ReadTrainInfo(trains, start_time, end_time, steps=self.steps)

    def calculate_settlement(self, idx: list = None, reload: bool = False):
        """
        Computes the cumulative settlement

        :param idx: (optional, default None) node to compute the calculations.
        :param reload: (optional, default False) reload last stage
        """

        self.reload = reload
        # run the accumulation model
        self.accumulation_model.settlement(self.trains, self.nb_nodes, idx, reload=self.reload)

        # create results
        self.__create_results()

        # assign results to previous stage
        self.previous_stage_results = {"time": self.results["time"], "displacement": self.results["displacement"]}

    def write_results(self, file_name: str):
        """
        Writes results to binary pickle file

        :param file_name: filename of the results
        """

        # check if path to output file exists. if not creates
        if not os.path.isdir(os.path.split(file_name)[0]):
            os.makedirs(os.path.split(file_name)[0])

        # dump dict
        with open(file_name, "wb") as f:
            pickle.dump(self.results, f)


    def __create_results(self):
        """
        Creates the results dictionary
        """
        # collect displacements
        aux = []
        for i, _ in enumerate(self.accumulation_model.nodes):
            # if reloading => append previous results
            if self.reload:
                aux.append(np.hstack([self.previous_stage_results["displacement"][i], self.accumulation_model.displacement[i].tolist()]).tolist())
            else:
                aux.append(self.accumulation_model.displacement[i].tolist())

        # if reloading => append results
        if self.reload:
            time = np.hstack([self.previous_stage_results["time"], self.trains.cumulative_time])
        else:
            time = self.trains.cumulative_time

        # create results struct
        self.results["nodes"] = self.accumulation_model.nodes
        self.results["time"] = time.tolist()
        self.results["displacement"] = aux

