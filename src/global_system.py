from src import track, soil, utils, geometry

from scipy import sparse
import numpy as np

class GlobalSystem:
    def __init__(self, n_sleepers):
        # self.track = track.UTrack(n_sleepers)
        # self.soil = soil.Soil()

        self.nodes = np.array([])
        self.elements = np.array([])

        self.global_mass_matrix = None
        self.global_stiffness_matrix = None
        self.global_damping_matrix = None

        self.force = None
        self.time = None

        self.model_parts=[]

        self.total_n_dof = None

    def __add_track_to_geometry(self):
        track_nodes = self.track.nodes
        track_elements = self.track.elements
        self.nodes = np.append(self.nodes, track_nodes)
        self.elements = np.append(self.elements, track_elements)

        return track_nodes, track_elements

    def __add_soil_to_geometry(self):
        soil_nodes = self.soil.nodes
        soil_elements = self.soil.elements

        self.nodes = np.append(self.nodes, soil_nodes)
        self.elements = np.append(self.elements, soil_elements)

    def _add_model_part_to_geometry(self, model_part):
        model_part_nodes = model_part.mesh.nodes
        model_part_elements = model_part.mesh.elements

        self.nodes = np.append(self.nodes, model_part_nodes)
        self.elements = np.append(self.elements, model_part_elements)


    def __get_unique_items(self, L):
        seen = set()
        seen2 = set()
        seen_add = seen.add
        seen2_add = seen2.add
        for item in L:
            if item in seen:
                seen2_add(item)
            else:
                seen_add(item)
        return np.array(seen2)

    def __merge_geometry(self):
        self.nodes = self.__get_unique_items(self.nodes)
        self.elements = self.__get_unique_items(self.elements)

    def set_geometry(self):


        for model_part in self.model_parts:
            model_part.set_geometry()
            self._add_model_part_to_geometry(model_part)

        # self.track.set_geometry()

        # track_nodes, _ = self.__add_track_to_geometry()
        # self.__add_soil_to_geometry()

        self.__merge_geometry()




    # def __add_soil_to_global_stiffness_matrix(self):
    #     for i in range(self.__n_sleepers):
    #         self.global_stiffness_matrix[i + self.rail.ndof, i + self.rail.ndof] \
    #             += self.soil.aux_stiffness_matrix[0, 0]
    #         self.global_stiffness_matrix[i + self.rail.ndof + self.__n_sleepers, i + self.rail.ndof] \
    #             += self.soil.aux_stiffness_matrix[1, 0]
    #         self.global_stiffness_matrix[i + self.rail.ndof, i + self.rail.ndof + self.__n_sleepers] \
    #             += self.soil.aux_stiffness_matrix[0, 1]
    #         self.global_stiffness_matrix[i + self.rail.ndof + self.__n_sleepers, i + self.rail.ndof + self.__n_sleepers] \
    #             += self.soil.aux_stiffness_matrix[1, 1]

    def set_global_stiffness_matrix(self):
        """

        :return:
        """
        self.global_stiffness_matrix = sparse.csr_matrix((self.total_n_dof, self.total_n_dof))



        self.track.set_global_stiffness_matrix()
        self.soil.set_global_stiffness_matrix()


    def calculte_ndof(self):
        for node in self.nodes:
            self.total_n_dof = self.total_n_dof + node.ndof



        # self.soil.set_aux_stiffness_matrix()
        # self.__add_soil_to_global_stiffness_matrix()


    # def apply_no_disp_boundary_condition(self):
    #     bottom_node_idxs = np.arange(self.rail.ndof + self.__n_sleepers,
    #                                  self.rail.ndof + 2 * self.__n_sleepers).tolist()
    #
    #     self.force = utils.delete_from_csr(self.force, row_indices=bottom_node_idxs)
    #     self.global_mass_matrix = utils.delete_from_csr(self.global_mass_matrix, row_indices=bottom_node_idxs,
    #                                                     col_indices=bottom_node_idxs)
    #     self.global_stiffness_matrix = utils.delete_from_csr(self.global_stiffness_matrix, row_indices=bottom_node_idxs,
    #                                                          col_indices=bottom_node_idxs)
    #     self.global_damping_matrix = utils.delete_from_csr(self.global_damping_matrix, row_indices=bottom_node_idxs,
    #                                                        col_indices=bottom_node_idxs)

    def initialise_track(self, rail, sleeper, rail_pads):
        self.track.rail = rail
        self.track.sleeper = sleeper
        self.track.rail_pads = rail_pads

        self.track.initialise_track()





