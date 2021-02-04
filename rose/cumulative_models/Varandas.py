import sys
import numpy as np
from scipy.integrate import trapz
import os
import json


class AccumulationModel:
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
        self.number_trains = []  # number of trains
        self.trains = []  # name of train types
        self.number_cycles = []  # number of loading cycles
        self.force = []
        self.cumulative_time = []
        self.nodes = []
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
        nb_nodes = []
        # determine number of loading cycles
        for t in trains:
            self.trains.append(t)
            # compute number of cycles
            aux = time_days * trains[t]["nb-per-hour"] * trains[t]["nb-hours"] * trains[t]["nb-axles"]
            # number cycles of train t
            self.number_cycles.append(aux)
            # force for train t
            self.force.append(trains[t]["forces"])
            # nb of nodes
            nb_nodes.append(len(trains[t]["forces"]))

        # number of trains
        self.number_trains = len(self.trains)
        # define cumulative time
        self.cumulative_time = np.linspace(0, time_days, int(np.max(self.number_cycles) + 1))

        # check if number of nodes is the same for all the trains
        if all(nb_nodes[0] == x for x in nb_nodes):
            self.nb_nodes = nb_nodes[0]
        else:
            sys.exit("Error: number of nodes is not the same for the different trains")

        return

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

        # compute maximum force
        for j in range(self.number_trains):
            # histogram.extend(self.number_cycles[j] * [np.max(np.abs(self.force[j]), axis=1) / self.force_scl_fct])
            self.force_max.append(np.max(np.abs(self.force[j]), axis=1)[idx] / self.force_scl_fct)

        # histogram
        hist = np.ones((int(len(idx)), int(np.sum(self.number_cycles)))) * np.inf
        # cumulative displacement
        self.displacement = np.zeros((int(len(idx)), int(np.max(self.number_cycles))))
        # displacement due to cycle n
        disp = np.zeros((int(len(idx)), int(np.max(self.number_cycles)) + 1))
        # Vector F for integration
        F = np.linspace(0, np.max(self.force_max, axis=0), self.nb_int_step)

        # incremental number of cycles
        nb_cycle = 0

        # for the maximum number of cycles
        for xx in range(int(np.max(self.number_cycles))):
            # for each train
            for j in range(self.number_trains):

                # if the number of cycles > than number of cycles imposed by train j -> continue
                if nb_cycle > self.number_cycles[j]:
                    nb_cycle += 1
                    continue

                # add force to histogram
                hist[:, nb_cycle] = self.force_max[j]

                # find number of loads in histogram
                h_f = np.where(self.force_max[j] >= hist[:, :nb_cycle + 1].T, 1, 0)
                h_f = np.sum(h_f, axis=0)

                # compute integral: trapezoidal rule
                integral = F ** self.alpha * (1 / (h_f + 1)) ** self.beta
                val = trapz(integral, F, axis=0)

                # compute displacement
                disp[:, nb_cycle] += self.gamma / self.M_alpha_beta * val

                # increase nb cycle
                nb_cycle += 1

        # compute displacements
        self.displacement = np.cumsum(disp, axis=1) / self.disp_scl_fct
        return

    def dump(self, output_file: str):
        """
        Writes results to json file

        :param output_file: filename of the results
        """

        # check if path to output file exists. if not creates
        if not os.path.isdir(os.path.split(output_file)[0]):
            os.makedirs(os.path.split(output_file)[0])

        # collect displacements
        aux = {}
        for i, n in enumerate(self.nodes):
            aux.update({str(n): self.displacement[i].tolist()})

        # create results dict
        results = {"time": self.cumulative_time.tolist(),
                   "settlement": aux}

        # dump ditch
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)

        return