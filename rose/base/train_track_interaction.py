from rose.train_model.train_model import TrainModel
from rose.base.global_system import GlobalSystem
from rose.base.model_part import ElementModelPart, TimoshenkoBeamElementModelPart
from rose.base.boundary_conditions import MovingPointLoad
from rose.utils.utils import *
from rose.utils.mesh_utils import *
from rose.solver.solver import StaticSolver

import numpy as np
from scipy import sparse
from copy import deepcopy

class CoupledTrainTrack(GlobalSystem):
    def __init__(self):
        super(CoupledTrainTrack, self).__init__()
        self.train: TrainModel                        # train system (carts, bogies, wheels)
        self.track: GlobalSystem                      # track system model part (track, soil, BC's, etc.)
        self.rail: TimoshenkoBeamElementModelPart     # rail model part which is in contact with train

        self.herzian_contact_coef = None
        self.herzian_power = None

        self.irregularities_at_wheels = None

        self.velocities = None
        self.wheel_loads = None

        self.static_force_vector = None

        self.velocities = None

        self.deformation_wheels = None
        self.deformation_track_at_wheels = None
        self.irregularities_at_wheels = None
        self.total_static_load = None

        self.contact_dofs = None

        self.track_elements = None
        self.wheel_distances = None
        self.wheel_node_dist = None
        self.y_shape_factors_track = None

    def get_position_wheels(self, t: int) -> List:
        """
        Gets distance of the wheels relative to 0 at time t
        :param t: time index
        :return:
        """
        position_wheels = [wheel.distances for wheel in self.train.wheels]
        position_wheels_at_t = [position_wheel[t] for position_wheel in position_wheels]
        return position_wheels_at_t

    def set_wheel_load_on_track(self, contact_track_elements, wheel_loads, t):
        """
        Sets vertical wheel load on track element which is in contact with the wheels

        :param contact_track_elements: elements which are in contact with the wheels
        :param wheel_loads: vertical wheel load
        :param t: time index
        :return:
        """
        # t_idxs = np.ones((len(wheel_loads), 4), dtype=int) * t

        global_indices = self.track_global_indices[:, t, :]

        # for idx, track_element in enumerate(contact_track_elements):
        #     global_indices[idx] = np.array([track_element.nodes[0].index_dof[1], track_element.nodes[0].index_dof[2],
        #                                track_element.nodes[1].index_dof[1], track_element.nodes[1].index_dof[2]])

        force_vector = self.y_shape_factors[:, t, :] * wheel_loads[:, None]
        mask = global_indices != np.array(None)

        track_global_force_vector = self.track.global_force_vector[:, t].toarray()[:,0]
        # track_global_force_vector = self.track.global_force_vector[:, t]
        # self.track.global_force_vector[global_indices[mask], t_idxs[mask]] = force_vector[mask]
        track_global_force_vector[global_indices[mask].astype(int)] = force_vector[mask]
        return track_global_force_vector

    def initialize_track_elements(self):
        """
        Finds track elements which are in contact with each wheel on each time step
        :return:
        """

        self.track_elements = np.empty((len(self.wheel_loads), len(self.time)),dtype=Element)
        self.track_global_indices = np.zeros((len(self.wheel_loads), len(self.time), 4), dtype=object)

        # wheel_loads_np = np.array([np.array(wheel_load.elements) for wheel_load in self.wheel_loads])
        # active_elements_ids = np.array([np.array(wheel_load.active_elements.nonzero()[0]) for wheel_load in self.wheel_loads])
        # active_elements = wheel_loads_np[:, active_elements_ids]

        # self.track_global_indices[:, :, :] = np.array([np.array([np.array([element.nodes[0].index_dof[1], element.nodes[0].index_dof[2],
        #                                                  element.nodes[1].index_dof[1], element.nodes[1].index_dof[2]]) for element in wheel_load]) for wheel_load in active_elements])

        for idx, wheel_load in enumerate(self.wheel_loads):
            wheel_load_np = np.array(wheel_load.elements)
            active_elements = wheel_load_np[wheel_load.active_elements.nonzero()[0]]
            # for t, element in enumerate(test):
            # global_indices_at_wheel = np.array([np.array([element.nodes[0].index_dof[1], element.nodes[0].index_dof[2],
            #                                               element.nodes[1].index_dof[1], element.nodes[1].index_dof[2]]) for element in active_elements])
            # mask = global_indices_at_wheel != np.array(None)
            #
            # self.track_global_indices[idx,:,:] = np.array([np.array([element.nodes[0].index_dof[1], element.nodes[0].index_dof[2],
            #                                     element.nodes[1].index_dof[1], element.nodes[1].index_dof[2]]) for element in active_elements])

            self.track_global_indices[idx,:,:] = np.array([np.array([element.nodes[0].index_dof[1], element.nodes[0].index_dof[2],
                                                                     element.nodes[1].index_dof[1], element.nodes[1].index_dof[2]]) for element in active_elements])

            # self.track_elements.append(wheel_load_np[wheel_load.active_elements.nonzero()[0]])
            self.track_elements[idx,:] = active_elements

    def calculate_distance_wheels_track_nodes(self):
        """
        Calculates distance of the wheels to the first node of the element which the wheel is in contact with

        :return:
        """

        # get distances at each time step of each wheel in numpy array
        self.wheel_distances = np.array([wheel.distances for wheel in self.train.wheels])

        # get first node of each contact element at each time step in numpy array
        nodal_coordinates = np.array([np.array([element.nodes[0].coordinates for element in wheel])
                                      for wheel in self.track_elements])


        # calculate absolute distance between 0-point and the first node for each contact element at time 0
        initial_distance = np.array([[distance_np(np.array([0, 0, 0]), wheel_coords[0]) * np.sign(wheel_coords[0, 0])
                                     for wheel_coords in nodal_coordinates]])

        # calculate cumulative distance of each first node of each contact element at each timestep
        cumulative_dist_nodes = np.array([calculate_cum_distances_coordinate_array(wheel_coords)
                                          for wheel_coords in nodal_coordinates])

        cumulative_dist_nodes = np.add(cumulative_dist_nodes,initial_distance.transpose())

        # calculate distance from wheel to first node of contact element at each time step
        self.wheel_node_dist = np.array([wheel - cumulative_dist_nodes[idx] for idx, wheel in enumerate(self.wheel_distances)])

    def calculate_y_shape_factors_rail(self):
        """
        Calculate y-shape factors for the contact rail elements at each time step
        :return:
        """

        self.y_shape_factors = np.zeros((len(self.wheel_node_dist),len(self.wheel_node_dist[0]), 4))
        for w_idx , w_track_elements in enumerate(self.track_elements):
            for idx, element in enumerate(w_track_elements):
                for model_part in element.model_parts:
                    if isinstance(model_part, TimoshenkoBeamElementModelPart):
                        model_part.set_y_shape_functions(self.wheel_node_dist[0, idx])
                        self.y_shape_factors[w_idx,idx, :] = copy.deepcopy(model_part.y_shape_functions)


    def get_track_element_at_wheels(self, t):
        """
        Get contact track elements at time t
        :param t: time index
        :return:
        """
        return self.track_elements[:, t]

    def __get_wheel_dofs(self):
        return [wheel.nodes[0].index_dof[1] for wheel in self.train.wheels]


    def __calculate_elastic_wheel_deformation(self, t):

        elastic_wheel_deformation = (
            # self.static_wheel_deformation
                - self.train.irregularities_at_wheels[:, t]
        )

        return elastic_wheel_deformation

    def calculate_static_contact_deformation(self):
        """
        Calculates static contact deformation based on herzion contact theory
        :return:
        """

        G = self.herzian_contact_coef
        pow = self.herzian_power

        self.static_contact_deformation = np.array([np.sign(wheel.total_static_load) * G * abs(wheel.total_static_load)
                                                    ** (1/pow) for wheel in self.train.wheels])

    def calculate_wheel_rail_contact_force(self, t, du_wheels):
        """
        Calculate contact force between wheels and rail based on herzian contact theory
        :param t: time index
        :param du_wheels: differential displacement wheels
        :return:
        """

        if np.isnan(du_wheels).any():
            raise ValueError("displacement in wheels is NAN, check if wheels are on track or reduce time steps")

        elastic_wheel_deformation = self.__calculate_elastic_wheel_deformation(t)

        contact_force = np.sign(elastic_wheel_deformation - du_wheels) * \
                        np.nan_to_num(np.power(1 / self.herzian_contact_coef * np.abs((elastic_wheel_deformation - du_wheels))
                        , self.herzian_power))

        return contact_force

    def get_disp_track_at_wheels(self, track_elements, t, u):
        """
        Get displacement of the track at the location of the wheels
        :param track_elements: contact track elements
        :param t: time index
        :param u: displacements at time t
        :return:
        """

        global_indices = self.track_global_indices[:,t,:]
        disp = np.zeros(global_indices.shape)

        # get displacement at indices
        mask = global_indices != np.array(None)
        # mask = ~np.isnan(global_indices)
        disp[mask] = u[global_indices[mask].astype(int)]

        # calculate displacement at the location of the wheels
        disp_at_wheels = np.sum(disp * self.y_shape_factors[:, t, :], axis=1)
        return disp_at_wheels

    def update_force_vector_contact(self, u: np.ndarray, t: int, F: np.ndarray) -> np.ndarray:
        """
        Updates the complete force vector due to wheel-rail contact forces

        :param u: displacements at time t
        :param t: time index
        :param F: Force vector at time t
        :return F:  Force vector at time t
        """

        # get displacement of wheels
        # u_wheels = u[self.train.contact_dofs]

        # determine contact force
        u_stat = self.static_contact_deformation
        contact_track_elements = self.get_track_element_at_wheels(t)
        disp_at_wheels = self.get_disp_track_at_wheels(contact_track_elements, t, u)
        contact_force = self.calculate_wheel_rail_contact_force(t, u[self.train.contact_dofs] + u_stat - disp_at_wheels)

        # add contact force to train
        # F[self.train.contact_dofs] += np.reshape(contact_force, (contact_force.size, 1))

        F[self.train.contact_dofs] += contact_force
        # add contact force to track
        track_global_force_vector = self.set_wheel_load_on_track(contact_track_elements, -contact_force, t)
        # F[:self.track.total_n_dof] = self.track.global_force_vector[:, t].toarray()[:,0]
        F[:self.track.total_n_dof] = track_global_force_vector

        return F

    def update_force_vector(self, u: np.ndarray, t: int) -> np.ndarray:
        """
        Updates the complete force vector at time t

        :param u: displacements at time t
        :param t: time index
        :return F: Force vector at time t
        """

        # Get force vector at time t
        # F = self.global_force_vector[:, t].copy()
        # F = self.global_force_vector[:, t].copy()
        # Update force vector due to contact force between rail and wheels
        # F = self.update_force_vector_contact(u, t, F)
        # F = self.update_force_vector_contact(u, t, self.global_force_vector[:, t])
        return self.update_force_vector_contact(u, t, self.global_force_vector[:, t].toarray()[:, 0])
        # return self.update_force_vector_contact(u, t, self.global_force_vector[:, t])

    def combine_global_matrices(self):
        """
        Combines global matrices of the train and the track

        :return:
        """
        # add track
        self.global_stiffness_matrix[:self.track.total_n_dof, :self.track.total_n_dof] = self.track.global_stiffness_matrix
        self.global_damping_matrix[:self.track.total_n_dof, :self.track.total_n_dof] = self.track.global_damping_matrix
        self.global_mass_matrix[:self.track.total_n_dof,:self.track.total_n_dof] = self.track.global_mass_matrix
        self.global_force_vector[:self.track.total_n_dof,:] = self.track.global_force_vector

        self.solver.u[:, :self.track.total_n_dof] = self.track.solver.u[:, :]
        self.solver.v[:, :self.track.total_n_dof] = self.track.solver.v[:, :]

        # add train
        self.global_stiffness_matrix[self.track.total_n_dof:self.total_n_dof,
        self.track.total_n_dof:self.total_n_dof] = self.train.global_stiffness_matrix

        self.global_damping_matrix[self.track.total_n_dof:self.total_n_dof,
        self.track.total_n_dof:self.total_n_dof] = self.train.global_damping_matrix

        self.global_mass_matrix[self.track.total_n_dof:self.total_n_dof,
        self.track.total_n_dof:self.total_n_dof] = self.train.global_mass_matrix

        self.global_force_vector[self.track.total_n_dof:self.total_n_dof, :] = self.train.global_force_vector

        self.solver.u[:, self.track.total_n_dof:self.total_n_dof] = self.train.solver.u[:, :]
        self.solver.v[:, self.track.total_n_dof:self.total_n_dof] = self.train.solver.v[:, :]

    def initialise_ndof(self):
        """
        Initialises number of the degree of freedom for the train and track
        :return:
        """

        # add track n_dof track to n_dof train
        for node in self.train.nodes:
            for idx, index_dof in enumerate(node.index_dof):
                if index_dof is not None:
                    node.index_dof[idx] += self.track.total_n_dof

        # recalculate contact dofs train
        self.train.get_contact_dofs()

        self.total_n_dof = self.track.total_n_dof + self.train.total_n_dof

    def assign_results_to_nodes(self):
        """
        Assigns all solver results to all nodes in the mesh
        :return:
        """
        self.track.mesh.nodes = list(
            map(lambda node: self._assign_result_to_node(node), self.track.mesh.nodes)
        )
        self.train.mesh.nodes = list(
            map(lambda node: self._assign_result_to_node(node), self.train.mesh.nodes)
        )

    def calculate_initial_displacement_track(self):
        """
        Calculates initial displacement of track
        :return:
        """

        self.track.calculate_initial_displacement()

    def calculate_initial_displacement_train(self, wheel_displacements):
        """
        Calculates initial displacement of the train

        :param wheel_displacements: displacement of track below initial location of the wheels
        :return:
        """
        self.train.calculate_initial_displacement(wheel_displacements, shift_in_ndof=self.track.total_n_dof)

    def calculate_initial_state(self):
        """
        Calculates initial state of coupled track and train system
        :return:
        """
        print("Initial static displacement of the train and the track")
        self.calculate_initial_displacement_track()

        contact_track_elements = self.get_track_element_at_wheels(0)
        disp_at_wheels = self.get_disp_track_at_wheels(contact_track_elements, 0, self.track.solver.u[0,:])

        self.calculate_initial_displacement_train(disp_at_wheels)
        self.calculate_static_contact_deformation()
        self.combine_global_matrices()

    def initialize_wheel_loads(self):
        """
        Initialises wheel loads on track
        :return:
        """
        self.wheel_loads = []
        for wheel in self.train.wheels:

            load = MovingPointLoad(normal_dof=self.rail.normal_dof,y_disp_dof=self.rail.normal_dof,
                                   z_rot_dof=self.rail.normal_dof, start_coord= [wheel.distances[0], 0, 0])
            load.time = self.time
            load.velocities = self.velocities
            load.contact_model_part = self.rail
            load.y_force = wheel.total_static_load
            load.nodes = self.rail.nodes
            load.elements = self.rail.elements

            # load = add_moving_point_load_to_track(
            #     self.rail,
            #     self.time,
            #     0,
            #     self.velocities,
            #     y_load=wheel.total_static_load, start_coords=[wheel.distances[0], 0, 0]
            # )
            # todo set y and z start coords

            # self.wheel_loads.extend(list(load.values()))
            self.wheel_loads.append(load)
        self.track.model_parts.extend(self.wheel_loads)

    def initialise_track_wheel_interaction(self):
        """
        Initialises interaction between wheel and track. And vectorises relevant arrays
            Finds elements which are in contact with the wheels at time t
            Calculates distance between wheel and node 1 of contact track element at time t
            Calculates y-shape factors of the track elements at time t

        :return:
        """
        print("Initialising track wheel interaction")
        self.initialize_track_elements()
        self.calculate_distance_wheels_track_nodes()
        self.calculate_y_shape_factors_rail()

    def initialise_train(self):
        """
        Initialises train
        :return:
        """
        print("Initialising train")

        self.train.solver = self.solver.__class__()
        self.train.initialise()

    def initialise_track(self):
        """
        Initialises track.
            Adds wheel loads to track
        :return:
        """
        print("Initialising track and subsoil")

        self.track.solver = self.solver.__class__()
        self.initialize_wheel_loads()
        self.track.initialise()

    def initialise(self):
        """
        Initialises train track interaction
        :return:
        """

        # initialize train and track in this order
        self.initialise_train()
        self.initialise_track()

        self.initialise_ndof()
        self.initialise_global_matrices()
        self.initialise_track_wheel_interaction()

        # Get stages from time array
        self.set_stage_time_ids()

        # initialise solver
        self.solver.initialise(self.total_n_dof, self.time)
        self.solver.load_func = self.update_force_vector

    def calculate_stage(self, start_time_id, end_time_id):
        
        self.track.global_force_vector = self.track.global_force_vector.tocsc()
        super(CoupledTrainTrack, self).calculate_stage(start_time_id, end_time_id)
    
    

    def finalise(self):
        """
        Finalises calculation
        :return:
        """

        super().finalise()
        self.assign_results_to_nodes()

    def main(self):
        """
        Performs main procedure
        :return:
        """

        self.validate_input()
        self.initialise()
        self.calculate_initial_state()

        # calculate stages
        for i in range(len(self.stage_time_ids) - 1):
            self.update(self.stage_time_ids[i], self.stage_time_ids[i + 1])
            self.calculate_stage(self.stage_time_ids[i], self.stage_time_ids[i + 1])

        self.finalise()
