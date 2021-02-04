from __future__ import print_function, absolute_import, division #makes KratosMultiphysics backward compatible with python 2.6 and 2.7

import sys
import os
import numpy as np
sys.path.append(r"D:\software_development\ROSE\KratosGeoMechanics")

import KratosMultiphysics as Kratos
from KratosMultiphysics.GeoMechanicsApplication.geomechanics_analysis import GeoMechanicsAnalysis

import KratosMultiphysics.ExternalSolversApplication
import KratosMultiphysics.StructuralMechanicsApplication as KratosStruct
import KratosMultiphysics.GeoMechanicsApplication as KratosGeo

from KratosMultiphysics.analysis_stage import AnalysisStage


class KratosElement():
    def __init__(self, element):
        self.element = element
        # self.nodes = [node for node in element.GetNodes()]
        self.nodes = [node for node in element]
        self.coords = np.array([[node.X, node.Y, node.Z] for node in self.nodes])
        self.local_coordinates = None

    def PointLocalCoordinates(self, glob_coord):
        # todo
        pass



class Triangle2D3(KratosElement):
    def __init__(self, triangle: KratosMultiphysics.Triangle2D3):
        super().__init__(triangle)
        self.triangle = triangle
        self.n_dim = self.triangle.WorkingSpaceDimension()


    def calculate_jacobian_matrix(self):
        dxs = self.coords[1:,0] - self.coords[0,0]
        dys = self.coords[1:,1] - self.coords[0,1]
        return np.array([dxs,dys])

    def PointLocalCoordinates(self, glob_coord):

        J = self.calculate_jacobian_matrix()
        inv_J = np.linalg.inv(J)

        DeltaXi = np.zeros(self.n_dim)
        result = np.zeros(self.n_dim)
        for i in range(self.n_dim):
            for j in range(self.n_dim):
                DeltaXi[i] += inv_J[i,j] *glob_coord[j]
            result[i] += DeltaXi[i]

        self.local_coordinates = result
        return np.array(result)


    def calculate_shape_functions(self):
        n_1 = 1 - self.local_coordinates[0] - self.local_coordinates[1]
        n_2 = self.local_coordinates[0]
        n_3 = self.local_coordinates[1]

        self.shape_functions = np.array([n_1, n_2, n_3])


class Line2D2(KratosElement):

    def __init__(self, line: KratosMultiphysics.Line2D2):
        super().__init__(line)
        self.line = line


    def PointLocalCoordinates(self,glob_coord):
        """
        Copied from Kratos, set into python code
        """

        x_glob = glob_coord[0]
        y_glob = glob_coord[1]

        tolerance = 1e-14
        # length = self.line.GetGeometry().Length()
        length = self.line.Length()

        length_1 = np.sqrt( np.power(x_glob - self.coords[0,0], 2)
                                                + np.power(y_glob - self.coords[0,1], 2))

        length_2 = np.sqrt( np.power(x_glob - self.coords[1,0], 2)
                                                + np.power(y_glob - self.coords[1,1], 2))

        if (length_1 <= (length + tolerance) and length_2 <= (length + tolerance)):
            xi = 2.0 * length_1/(length + tolerance) - 1.0
        elif (length_1 > (length + tolerance)):
            xi = 2.0 * length_1/(length + tolerance) - 1.0 # NOTE: The same value as before, but it will be > than 1
        elif (length_2 > (length + tolerance)):
            xi = 1.0 - 2.0 * length_2/(length + tolerance)
        else:
            xi = 2.0 # Out of the line!!!

        self.local_coordinates = np.array([xi,0,0])

        return self.local_coordinates

    def calculate_shape_functions(self):
        n1 = 0.5 * (1.0 - self.local_coordinates[0])
        n2 = 0.5 * (1.0 + self.local_coordinates[0])
        self.shape_functions = np.array([n1, n2])


class KratosGeomechanics(GeoMechanicsAnalysis):


    def __init__(self,model,parameters):

        nodal_results = parameters["output_processes"]["gid_output"][0]["Parameters"]["postprocess_parameters"]["result_file_configuration"]["nodal_results"]

        super(KratosGeomechanics, self).__init__(model,parameters)
        self.nodes = None
        self.elements = None
        self.conditions = None

        self.contact_elements = None

    def get_displacements(self, nodes):
        return [node.GetSolutionStepValue(KratosMultiphysics.DISPLACEMENT,1) for node in nodes]

    def get_rotations(self, nodes):
        return [node.GetSolutionStepValue(KratosMultiphysics.ROTATION,1) for node in nodes]


    def set_point_load(self, node, x_load: float = None, y_load: float = None, z_load: float=None):
        if x_load is not None:
            node.SetSolutionStepValue(KratosStruct.POINT_LOAD_X, 0, x_load)
        if y_load is not None:
            node.SetSolutionStepValue(KratosStruct.POINT_LOAD_Y, 0, y_load)
        if y_load is not None:
            node.SetSolutionStepValue(KratosStruct.POINT_LOAD_Z, 0, z_load)


    def set_point_moment(self, node, x_moment: float = None, y_moment: float = None, z_moment: float=None):
        if x_moment is not None:
            node.SetSolutionStepValue(KratosStruct.POINT_MOMENT_X, 0, x_moment)
        if y_moment is not None:
            node.SetSolutionStepValue(KratosStruct.POINT_MOMENT_Y, 0, y_moment)
        if z_moment is not None:
            node.SetSolutionStepValue(KratosStruct.POINT_MOMENT_Z, 0, z_moment)


    def __up_scale_step_size(self):
        KratosMultiphysics.Logger.PrintInfo(self._GetSimulationName(), "Up-scaling with factor: ", self.increase_factor)
        self.delta_time *= self.increase_factor
        t = self._GetSolver().GetComputingModelPart().ProcessInfo[KratosMultiphysics.TIME]
        corrected_time = t + self.delta_time
        if (corrected_time > self.end_time):
            corrected_time = self.end_time
            self.delta_time = corrected_time - t

    def __down_scale_step_size(self):
        KratosMultiphysics.Logger.PrintInfo(self._GetSimulationName(), "Down-scaling with factor: ", self.reduction_factor)
        self.delta_time *= self.reduction_factor

        # Reset displacements to the initial
        KratosMultiphysics.VariableUtils().UpdateCurrentPosition(self._GetSolver().GetComputingModelPart().Nodes, KratosMultiphysics.DISPLACEMENT,1)
        for node in self._GetSolver().GetComputingModelPart().Nodes:
            dold = node.GetSolutionStepValue(KratosMultiphysics.DISPLACEMENT,1)
            node.SetSolutionStepValue(KratosMultiphysics.DISPLACEMENT,0,dold)

    def Initialize(self):
        super(KratosGeomechanics,self).Initialize()

        self.nodes = [node for node in self._GetSolver().GetComputingModelPart().Nodes]
        self.elements = [element for element in self._GetSolver().GetComputingModelPart().Elements]
        self.conditions = [cond for cond in self._GetSolver().GetComputingModelPart().Conditions]

    def RunSolutionLoop(self):


        point_load_values = [np.array(load["Parameters"]["value"].GetVector()) for load in parameters["processes"]["loads_process_list"] if load["Parameters"]["variable_name"].GetString() == "POINT_LOAD"]
        new_condition = KratosMultiphysics.Condition

        model_parts = [model_part for model_part in self._GetSolver().GetComputingModelPart().SubModelParts]

        beam_elements = [cond.GetGeometry() for cond in self.conditions if isinstance(cond.GetGeometry(), KratosMultiphysics.Line2D2)]

        # self._GetSolver().GetComputingModelPart().SetConditions()

        beam_nodes = [cond.GetGeometry()[0] for cond in self.conditions if not isinstance(cond.GetGeometry(), KratosMultiphysics.Line2D2)]


        # beam_model_part = model_parts[1]
        # beam_nodes = [cond.GetNodes()[0] for cond in self.conditions[22:]]
        # beam_nodes = [node for node in beam_model_part.Nodes]
        # beam_elements =  [node for node in beam_model_part.Elements]

        # beam_elements =  [cond.GetGeometry() for cond in self.conditions[:20]]

        # beam_elements
        beam_elements = sorted(beam_elements,key = lambda x: x.Center().X )
        # [cond.GetGeometry() for cond in self.conditions[:22]][0].Center().X

        beam_coords = np.array([[node.X, node.Y, node.Z] for node in beam_nodes])
        diff = np.diff(beam_coords,axis=0)
        distances = np.sqrt(np.sum(np.power(diff,2),axis=1))
        cum_distances = np.concatenate(([0], np.cumsum(distances)))

        velocity= 10
        point_loc = 0
        point_load_v = -10000

        if self._GetSolver().settings["reset_displacements"].GetBool():
            old_total_displacements = [node.GetSolutionStepValue(KratosGeo.TOTAL_DISPLACEMENT)
                                       for node in self._GetSolver().GetComputingModelPart().Nodes]

        while self.KeepAdvancingSolutionLoop():
            if(self.delta_time > self.max_delta_time):
                self.delta_time = self.max_delta_time
                KratosMultiphysics.Logger.PrintInfo(self._GetSimulationName(), "reducing delta_time to max_delta_time: ", self.max_delta_time)
            t = self._GetSolver().GetComputingModelPart().ProcessInfo[KratosMultiphysics.TIME]
            new_time = t + self.delta_time
            if (new_time > self.end_time):
                new_time = self.end_time
                self.delta_time = new_time - t
            self._GetSolver().GetComputingModelPart().ProcessInfo[KratosMultiphysics.STEP] += 1
            self._GetSolver().main_model_part.CloneTimeStep(new_time)
            self._GetSolver().main_model_part.ProcessInfo[KratosMultiphysics.START_TIME] = self.time
            self._GetSolver().main_model_part.ProcessInfo[KratosMultiphysics.END_TIME] = self.end_time

            converged = False
            number_cycle = 0
            while (not converged and number_cycle < self.number_cycles):

                number_cycle +=1
                KratosMultiphysics.Logger.PrintInfo(self._GetSimulationName(), "cycle: ", number_cycle)
                t = self._GetSolver().GetComputingModelPart().ProcessInfo[KratosMultiphysics.TIME]
                corrected_time = t - self._GetSolver().GetComputingModelPart().ProcessInfo[KratosMultiphysics.DELTA_TIME] + self.delta_time
                self._GetSolver().GetComputingModelPart().ProcessInfo[KratosMultiphysics.TIME] = corrected_time
                self._GetSolver().GetComputingModelPart().ProcessInfo[KratosMultiphysics.DELTA_TIME] = self.delta_time

                self.InitializeSolutionStep()
                self._GetSolver().Predict()

                #todo remove this

                # elements = [element for element in self._GetSolver().GetComputingModelPart().Elements]
                # conditions = [cond for cond in self._GetSolver().GetComputingModelPart().Conditions]
                # geom = elements[10].GetGeometry()
                # nodes = [node for node in self._GetSolver().GetComputingModelPart().Nodes]
                # displacement = [node.GetSolutionStepValue(KratosMultiphysics.DISPLACEMENT_Y,1) for node in nodes]
                point_load = [node.GetSolutionStepValue(KratosStruct.POINT_LOAD_Y,1) for node in beam_nodes]
                for node in self.nodes:
                    node.SetSolutionStepValue(KratosStruct.POINT_LOAD_Y,0, 0)

                dx = self.delta_time * velocity
                point_loc = point_loc+dx
                res = np.where(cum_distances>=point_loc)
                if res[0].size>0:
                    element_idx = res[0][0] - 1
                else:
                    element_idx = None
                # element_idx = np.where(cum_distances>=point_loc)[0][0] -1

                if element_idx is not None:

                    beam_element = Line2D2(beam_elements[element_idx])
                    beam_element.PointLocalCoordinates([point_loc,1,0])
                    beam_element.calculate_shape_functions()
                    new_point_load = point_load_v * beam_element.shape_functions

                    for node,load in zip(beam_element.nodes, new_point_load):
                        node.SetSolutionStepValue(KratosStruct.POINT_LOAD_Y,0, load)
                    # node.SetSolutionStepValue(KratosStruct.POINT_LOAD_Y,1, load)




                # contact_element=
                # nodes[10].SetSolutionStepValue(KratosStruct.POINT_LOAD_Y,0, point_load[10]*2)
                # point_load2 = [node.GetSolutionStepValue(KratosStruct.POINT_LOAD_Y,1) for node in nodes]

                # elements[0].GetGeometry()
                # element = Triangle2D3(elements[0].GetGeometry())
                # PointLocalCoordinates
                # test = element.PointLocalCoordinates([0.9,0.1])


                #####

                # run solver
                converged = self._GetSolver().SolveSolutionStep()

                # alter step size if required
                n_iterations = self._GetSolver().GetComputingModelPart().ProcessInfo[KratosMultiphysics.NL_ITERATION_NUMBER]
                if (n_iterations >= self.max_iterations or not converged):
                    self.__down_scale_step_size()
                elif (n_iterations < self.min_iterations):
                    self.__up_scale_step_size()

            if self._GetSolver().settings["reset_displacements"].GetBool() and converged:
                for idx, node in enumerate(self._GetSolver().GetComputingModelPart().Nodes):
                    self._CalculateTotalDisplacement(node, old_total_displacements[idx])

            if (not converged):
                raise Exception('The maximum number of cycles is reached without convergence!')

            self.FinalizeSolutionStep()
            self.OutputSolutionStep()



    def Finalize(self):
        super(KratosGeomechanics,self).Finalize()

if __name__ == '__main__':

    wd = r"D:\software_development\ROSE\Kratos_calculations\moving_load.gid"
    os.chdir(wd)
    parameter_file_name = r"ProjectParameters.json"
    with open(parameter_file_name,'r') as parameter_file:
        parameters = Kratos.Parameters(parameter_file.read())

    model = Kratos.Model()
    simulation = KratosGeomechanics(model,parameters)

    simulation.Initialize()
    simulation.RunSolutionLoop()
    simulation.Finalize()