
import scipy.optimize as op
import numpy as np
from typing import List
import copy


class ModelResults:
    def __init__(self):
        self.result_names:List[str] = None # array of parameter names to be compared with observations e.g.: ["displacements_out"]
        self.result_indices = None          # global indices of results to be compared with observations
        self.time_step_indices = None       # indices of the time steps which are to be taken into account
        self.result_function = None         # optional post process function
        self.args = {}                      # arguments for optional post process functions


class OptimisationModelPart:
    def __init__(self):
        self.model_part = None                              # model part to be optimised
        self.optimisation_parameter_names:List[str] = None  # [parameter names within model part to be optimised
                                                            # e.g. ['stiffness', 'damping']


class Optimisation():
    def __init__(self):
        self.model = None               # global system
        self.parameter_array = None     # array of parameters to be optimised e.g. bulk stiffness and damping
        self.observations: np.ndarray = None    # observations
        self.optimisation_model_parts: List[OptimisationModelPart] = None       # list of model parts to be optimised
        self.model_results: List[ModelResults] = None   # list of model results
        self.result = None              # optimisation results

        self.__original_model = None
        self.__optimisation_model_part_indices = None

    def initialise(self):
        """
        Initialise optimisation
        :return:
        """
        self.__original_model = copy.deepcopy(self.model)
        self.__optimisation_model_part_indices = [self.model.track.model_parts.index(optimisation_model_part.model_part)
                                                  for optimisation_model_part in self.optimisation_model_parts ]

    def reset_model(self):
        """
        Reset numerical model
        :return:
        """
        self.model = copy.deepcopy(self.__original_model)


    def residual_function(self, parameters, method='maximum'):
        """
        computes the minimisation function

        :param parameters: ordered 1d-array of input parameters
        :return:
        """

        # initialise model
        self.reset_model()

        # set parameters in numerical model
        # todo set for general model, currently it only works for model parts on the track
        i = 0
        for idx, model_part in zip(self.__optimisation_model_part_indices, self.optimisation_model_parts):
            for parameter_name in model_part.optimisation_parameter_names:
                setattr(self.model.track.model_parts[idx], parameter_name, parameters[i])
                i += 1

        # run numerical model
        self.model.main()

        # get results from numerical model
        results = []
        for model_result in self.model_results:

            # get results from numerical model without using a post process funtion
            if model_result.result_function is None:
                for res_name in model_result.result_names:
                    result = getattr(self.model, res_name)[model_result.time_step_indices, :]
                    result = result[:, model_result.result_indices]
                    if method == 'maximum':
                        result = np.max(np.abs(result),axis=0)
                        results.append(result)
                    else:
                        raise("Method is not implemented")

                results = np.array(results)

            # get results from numerical model while using a post process function
            else:
                for res_name in model_result.result_names:
                    signal = getattr(self.model, res_name)[model_result.time_step_indices, :]
                    signal = signal[:, model_result.result_indices]

                    result = np.array(model_result.result_function(signal, model_result.args))
                    results.append(result)

                results = np.array(results)

        # calculate residual and return
        residual = results - self.observations

        # todo, make write function general
        # todo, correctly
        with open('tmp_res.txt', 'a') as file:
            file.write(f"{parameters[0]};{results[0][1]} \n")

        return residual[0]



    def least_square(self, x0, method = "lm", ftol=1e-8,xtol=1e-8,gtol=1e-8):
        """
        Solve a nonlinear least-squares problem

        :param x0: initial parameters
        :param method: method of optimisation method, "trf", "dogbox", "lm"
                    trf is best for large bounded sparse problems
                    lm is best for smaller unbounded non sparsed matrices
        :param ftol: tolerance of cost function return
        :param xtol: tolerance of variable change
        :param gtol: tolerance of norm of the gradient
        :return:
        """


        self.result = op.least_squares(self.residual_function, x0, method=method, ftol=ftol, xtol=xtol, gtol=gtol,
                                       diff_step=0.001)



