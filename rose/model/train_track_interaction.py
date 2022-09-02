from rose.model.train_model import TrainModel
from rose.model.global_system import GlobalSystem
from rose.model.model_part import TimoshenkoBeamElementModelPart
from rose.model.boundary_conditions import MovingPointLoad
from rose.model.utils import *
from rose.pre_process.mesh_utils import *

import numpy as np
import itertools


class CoupledTrainTrack(GlobalSystem):
    """
    Class which contains a coupled system of a track and a train. This class bases from
    :class:`~rose.model.global_system.GlobalSystem`.

    :Attributes:

        - :self.train:                  The complete train system (carts, bogies, wheels)
        - :self.track:                  The complete track system (rail, sleepers, dampers, soil, boundary conditions)
        - :self.rail:                   Model part of the track which is in contact with the train
        - :self.hertzian_contact_coef:  Coefficient used in  Hertzian contact theory
        - :self.hertzian_power:         Power used in Hertzian contact theory
        - :self.wheel_loads:            List of moving load model parts at the location of the train wheels
        - :self.static_contact_deformation: Static deformation at contact following Hertzian theory
        - :self.track_elements:         Contact track elements at each time step for each contact force
        - :self.track_global_indices:   Indices of the nodal degrees of freedom of the contact track elements
        - :self.wheel_distances:        Distance from [0,0] coordinate of each wheel at each time step
        - :self.wheel_node_dist:        Distance of each wheel to first node of contact element at each time step
        - :self.y_shape_factors:        Vertical load shape vectors for each contact element at each time step

    """
    def __init__(self):
        super(CoupledTrainTrack, self).__init__()
        self.train: TrainModel = None
        self.track: GlobalSystem = None
        self.rail: TimoshenkoBeamElementModelPart = None

        self.hertzian_contact_coef: float = None
        self.hertzian_power: float = None

        self.wheel_loads: List = None
        self.static_contact_deformation: np.ndarray = None

        self.track_elements: np.ndarray = None
        self.track_global_indices: np.ndarray = None

        self.wheel_distances: np.ndarray = None
        self.wheel_node_dist: np.ndarray = None

        self.y_shape_factors: np.ndarray = None

    def set_wheel_load_on_track(self, wheel_loads: np.ndarray, t: int) -> np.ndarray:
        """
        Sets vertical wheel load on track element which is in contact with the wheels at time t

        :param wheel_loads: vertical wheel load of each wheel at time t
        :param t: time index
        :return:
        """

        # find global contact indices at time t
        global_indices = self.track_global_indices[:, t, :]

        # calculate force vector
        force_vector = self.y_shape_factors[:, t, :] * wheel_loads[:, None]

        # set wheel load on track
        mask = global_indices != np.array(None)
        track_global_force_vector = self.track.global_force_vector[:, t].toarray()[:,0]
        track_global_force_vector[global_indices[mask].astype(int)] = force_vector[mask]
        return track_global_force_vector

    def initialize_track_elements(self):
        """
        Finds track elements and degree of freedom indices in the global system which are in contact with each wheel on
        each time step.

        # todo currently track global indices are the y disp index and z-rot index, make this more general such that
           it works for an inclined track
        :return:
        """

        # intitialise contact track elements and indices in the global system
        self.track_elements = np.empty((len(self.wheel_loads), len(self.time)), dtype=Element)
        self.track_global_indices = np.zeros((len(self.wheel_loads), len(self.time), 4), dtype=object)

        # for each wheel load, get the contact elements at each time step
        for idx, wheel_load in enumerate(self.wheel_loads):

            # Active elements are the elements where the force is not zero
            wheel_load_np = np.array(wheel_load.elements)
            active_elements = wheel_load_np[wheel_load.active_elements.nonzero()[0]]

            # Track global indices are the y-disp and z-rot global indices of the two nodes of the contact elements
            # todo, make this more general. Such that different degrees of freedom are allowed
            self.track_global_indices[idx, :, :] = np.array(
                [np.array([element.nodes[0].index_dof[1], element.nodes[0].index_dof[2],
                           element.nodes[1].index_dof[1], element.nodes[1].index_dof[2]])
                 for element in active_elements])

            # track elements are the active elements where the force is not zero
            self.track_elements[idx, :] = active_elements

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

        # initialise y shape factors as a [n wheels, n time steps, 4] np array
        self.y_shape_factors = np.zeros((len(self.wheel_node_dist), len(self.wheel_node_dist[0]), 4))

        # loop over wheels
        for w_idx, w_track_elements in enumerate(self.track_elements):
            # loop over track elements per wheel
            for idx, element in enumerate(w_track_elements):
                # find Timoshenko beam model part
                for model_part in element.model_parts:
                    if isinstance(model_part, TimoshenkoBeamElementModelPart):
                        # set y shape factors and apply to y shape factors array
                        model_part.set_y_shape_functions(self.wheel_node_dist[w_idx, idx])
                        self.y_shape_factors[w_idx,idx, :] = np.copy(model_part.y_shape_functions)

    def __calculate_elastic_wheel_deformation(self, t: int) -> np.ndarray:
        elastic_wheel_deformation = (- self.train.irregularities_at_wheels[:, t])

        return elastic_wheel_deformation

    def calculate_static_contact_deformation(self):
        """
        Calculates static contact deformation at each wheel based on Hertzian contact theory. Note that there is no loss
        of contact.
        :return:
        """

        G = self.hertzian_contact_coef
        pow = self.hertzian_power

        self.static_contact_deformation = np.array([np.sign(wheel.total_static_load) * G * abs(wheel.total_static_load)
                                                    ** (1/pow) for wheel in self.train.wheels])

    def calculate_wheel_rail_contact_force(self, t: int, du_wheels: np.ndarray) -> np.ndarray:
        """
        Calculate contact force between wheels and rail based on Hertzian contact theory. Note that there is no loss
        of contact.

        :param t: time index
        :param du_wheels: differential displacement wheels at time t
        :return:
        """

        # Check if any differential displacement of the wheels is NAN. If so, raise an error
        if np.isnan(du_wheels).any():
            raise ValueError("displacement in wheels is NAN, check if wheels are on track or reduce time steps")

        # calculate elastic wheel deformation at time t
        elastic_wheel_deformation = self.__calculate_elastic_wheel_deformation(t)

        # calculate contact force
        contact_force = np.sign(elastic_wheel_deformation - du_wheels) * \
                        np.nan_to_num(np.power(1 / self.hertzian_contact_coef * (elastic_wheel_deformation - du_wheels)
                                               , self.hertzian_power))

        return contact_force

    def get_disp_track_at_wheels(self, t: int, u: np.ndarray) -> np.ndarray:
        """
        Get displacement of the track at the location of the wheels at time t

        :param t: time index
        :param u: displacements at time t
        :return:
        """

        # find global indices of contact track elements at time t
        global_indices = self.track_global_indices[:,t,:]

        # initialise displacement vector
        disp = np.zeros(global_indices.shape)

        # get displacement at contact global indices
        mask = global_indices != np.array(None)
        disp[mask] = u[global_indices[mask].astype(int)]

        # calculate vertical displacement at the location of the wheels
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
        u_wheels = u[self.train.contact_dofs]

        # get static contact force
        u_stat = self.static_contact_deformation

        # get track displacement at the wheel location
        disp_at_wheels = self.get_disp_track_at_wheels(t, u)

        # calculate contact force
        contact_force = self.calculate_wheel_rail_contact_force(t, u_wheels + u_stat - disp_at_wheels)

        # add contact force to train
        F[self.train.contact_dofs] += contact_force

        # add contact force to track
        track_global_force_vector = self.set_wheel_load_on_track(-contact_force, t)
        F[:self.track.total_n_dof] = track_global_force_vector

        return F

    def update_force_vector(self, u: np.ndarray, t: int) -> np.ndarray:
        """
        Updates the complete force vector at time t

        :param u: displacements at time t
        :param t: time index
        :return F: Force vector at time t
        """

        # Update force vector due to contact force between rail and wheels
        return self.update_force_vector_contact(u, t, self.global_force_vector[:, t].toarray()[:, 0])

    def combine_global_matrices(self):
        """
        Combines global matrices of the train and the track

        :return:
        """
        # add track global matrices to the coupled system
        self.global_stiffness_matrix[:self.track.total_n_dof, :self.track.total_n_dof] = \
            self.track.global_stiffness_matrix
        self.global_damping_matrix[:self.track.total_n_dof, :self.track.total_n_dof] = self.track.global_damping_matrix
        self.global_mass_matrix[:self.track.total_n_dof, :self.track.total_n_dof] = self.track.global_mass_matrix
        self.global_force_vector[:self.track.total_n_dof, :] = self.track.global_force_vector

        # add track displacement and velocity to global system
        self.solver.u[:, :self.track.total_n_dof] = self.track.solver.u[:, :]
        self.solver.v[:, :self.track.total_n_dof] = self.track.solver.v[:, :]

        # add train global matrices to the coupled system
        self.global_stiffness_matrix[self.track.total_n_dof:self.total_n_dof, self.track.total_n_dof:self.total_n_dof] \
            = self.train.global_stiffness_matrix
        self.global_damping_matrix[self.track.total_n_dof:self.total_n_dof, self.track.total_n_dof:self.total_n_dof] \
            = self.train.global_damping_matrix
        self.global_mass_matrix[self.track.total_n_dof:self.total_n_dof, self.track.total_n_dof:self.total_n_dof] \
            = self.train.global_mass_matrix
        self.global_force_vector[self.track.total_n_dof:self.total_n_dof, :] = self.train.global_force_vector

        # add train displacement and velocity to global system
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

        # calculate total number of degrees of freedom
        self.total_n_dof = self.track.total_n_dof + self.train.total_n_dof

    def assign_results_to_nodes(self):
        """
        Assigns all solver results to all nodes in the mesh
        :return:
        """

        # apply results to the track
        self.track.mesh.nodes = list(
            map(lambda node: self._assign_result_to_node(node), self.track.mesh.nodes)
        )

        # apply results to the train
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

        # calculate initial displacement of the track system
        self.calculate_initial_displacement_track()

        # calculate initial displacement of the train
        disp_at_wheels = self.get_disp_track_at_wheels(0, self.track.solver.u[0,:])
        self.calculate_initial_displacement_train(disp_at_wheels)

        # calculate initial Hertzian contact deformation
        self.calculate_static_contact_deformation()

        # combine train and track global matrices
        self.combine_global_matrices()

        # calculate rayleigh damping
        self.calculate_rayleigh_damping()

    def __check_train_position(self, rail_nodes: List):
        # get limits track
        x_coords_track = [node.coordinates[0] for node in rail_nodes]
        limits_track = np.min(x_coords_track), np.max(x_coords_track)

        # get limits wheels
        limits_wheel_distances = np.array([[min(wheel.distances), max(wheel.distances)] for wheel in self.train.wheels])

        # precision is machine float64 precision * n time steps
        precision = np.finfo(float).eps * len(self.train.wheels[0].distances)

        # check if train is on track at all times
        if (limits_wheel_distances < limits_track[0] - precision).any() or \
                (limits_wheel_distances > limits_track[1] + precision).any():
            raise ValueError(
                "At some point in time, one or more wheels of the train are located outside the track geometry")

    def initialize_wheel_loads(self):
        """
        Initialises wheel loads on track
        :return:
        """

        # get all nodes and elements from all rail model parts,
        rail_model_parts = [model_part for model_part in self.track.model_parts if isinstance(model_part,TimoshenkoBeamElementModelPart)]
        rail_nodes = [part.nodes for part in rail_model_parts]
        rail_nodes = list(itertools.chain.from_iterable(rail_nodes))
        rail_node_idxs = [node.index for node in rail_nodes]
        _, unique_idxs = np.unique(rail_node_idxs, return_index=True)
        rail_nodes = list(np.array(rail_nodes)[unique_idxs])

        rail_elements = [part.elements for part in rail_model_parts]
        rail_elements = list(itertools.chain.from_iterable(rail_elements))

        self.__check_train_position(rail_nodes)

        self.wheel_loads = []
        # loop over the wheels
        for wheel in self.train.wheels:
            # todo set y and z start coords, currently wheels are placed at y = z = 0

            # initialise wheel load as a moving point load
            load = MovingPointLoad(x_disp_dof=rail_model_parts[0].normal_dof, y_disp_dof=rail_model_parts[0].y_disp_dof,
                                   z_rot_dof=rail_model_parts[0].z_rot_dof, start_distance=wheel.distances[0])
            load.time = self.time

            # add wheel properties to the moving load
            load.velocities = self.train.velocities
            load.y_force = wheel.total_static_load

            # add track properties to the moving load
            load.contact_model_parts = rail_model_parts
            load.contact_model_part = self.rail
            load.nodes = rail_nodes
            load.elements = rail_elements

            # add moving load to the wheel loads list
            self.wheel_loads.append(load)

        # Get current non-moving load model parts
        current_track_model_parts = [model_part for model_part in self.track.model_parts if
                                     not isinstance(model_part, MovingPointLoad)]

        # add wheel loads to the model parts list
        self.track.model_parts = current_track_model_parts + self.wheel_loads

    def initialise_track_wheel_interaction(self):
        """
        Initialises interaction between wheel and track. And vectorises relevant arrays for performance purpose.
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
        Initialises train system
        :return:
        """
        print("Initialising train")

        self.train.solver = self.solver.__class__()
        self.train.initialise()

    def clean_model(self):
        """
        Cleans model, this is required if multiple calculations are performed with the same model.
        - Resets degrees of freedom.
        - Clears model parts from nodes and elements.
        - Resets train mesh.
        :return:
        """
        self.total_n_dof = None
        self.track.total_n_dof = None
        self.train.total_n_dof = None
        for node in self.track.mesh.nodes:
            node.index_dof = np.array([None, None, None])
            node.model_parts = []
        for element in self.track.mesh.elements:
            element.model_parts = []

        self.train.mesh = Mesh()

    def initialise_track(self):
        """
        Initialises track system and adds wheel loads to track
        :return:
        """
        print("Initialising track and subsoil")

        # set element and node ids
        self.track.mesh.reorder_element_ids()
        self.track.mesh.reorder_node_ids()

        self.track.solver = self.solver.__class__()
        self.initialize_wheel_loads()
        self.track.initialise()

    def initialise(self):
        """
        Initialises train, track, train-track interaction, stages and the solver.
        :return:
        """

        # clean model
        self.clean_model()

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
        """
        Calculates stage and sets track global force vector to csc sparse matrix
        :param start_time_id: start time index
        :param end_time_id: end time index
        :return:
        """

        self.track.global_force_vector = self.track.global_force_vector.tocsc()
        super(CoupledTrainTrack, self).calculate_stage(start_time_id, end_time_id)

    def finalise(self):
        """
        Finalises calculation. Calculate force in the track elements and assign results to the nodes.
        :return:
        """

        super().finalise()
        self.track.calculate_force_in_elements()
        self.assign_results_to_nodes()

    def main(self):
        """
        Performs main procedure. Validate input, initialise input, calculate the initial state, calculate each stage and
        finalise the calculation
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
