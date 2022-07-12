import sys
import numpy as np
from scipy.integrate import trapz
import os
import pickle
from tqdm import tqdm


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
        self.number_cycles = []  # number of total loading cycles
        self.nb_cycles_day = []  # number of loading cycles / day
        self.force = []
        self.cumulative_time = []
        self.cumulative_nb_cycles = []
        self.index_cumulative_distributed = []
        self.nodes = []
        self.displacement = []
        self.results = {}
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
            aux = trains[t]["nb-per-hour"] * trains[t]["nb-hours"] * trains[t]["nb-axles"]
            # number axles a day
            self.nb_cycles_day.append(aux)
            # number cycles of train
            self.number_cycles.append(aux * time_days)
            # force for train t
            self.force.append(trains[t]["forces"])
            # nb of nodes
            nb_nodes.append(len(trains[t]["forces"]))

        # number of trains
        self.number_trains = len(self.trains)
        # define cumulative time
        self.cumulative_time = np.linspace(0, time_days-1, int(np.max(self.number_cycles)))
        # define cumulative nb cycles
        self.cumulative_nb_cycles = np.linspace(0, int(np.max(self.number_cycles)) - 1, int(np.max(self.number_cycles)))

        # index for distributed loading
        for nb in self.number_cycles:
            self.index_cumulative_distributed.append(np.linspace(0, nb-1, int(np.max(self.number_cycles))) * (np.max(self.number_cycles) / nb))
        self.index_cumulative_distributed = np.array(self.index_cumulative_distributed).astype(int)

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
        aux = {}
        for i, n in enumerate(self.nodes):
            aux.update({str(n): self.displacement[i].tolist()})

        # create results dict
        self.results = {"time": self.cumulative_time.tolist(),
                        "settlement": aux}
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
