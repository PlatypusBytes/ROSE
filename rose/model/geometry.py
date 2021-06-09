from __future__ import annotations
import numpy as np


class Node:
    """
       Class which contains a Node.

       :Attributes:

           - :self.index:              index of the node within the mesh
           - :self.index_dof:          numpy array of the degrees of freedom indices within the global system
           - :self.coordinates:        numpy array of the x,y and z coordinate of the node
           - :self.x_disp_dof:         Boolean which is true if displacement of node in x direction is permitted
           - :self.y_disp_dof:         Boolean which is true if displacement of node in y direction is permitted
           - :self.z_rot_dof:          Boolean which is true if rotation of node around z-axis is permitted
           - :self.displacements:      Numpy array of displacements of node over time for each degree of freedom
           - :self.velocities:         Numpy array of velocities of node over time for each degree of freedom
           - :self.accelerations:      Numpy array of accelerations of node over time for each degree of freedom
           - :self.force:              Numpy array of force in node over time for each degree of freedom
           - :self.self.model_parts:   List of model parts which are connected to current node

    """
    def __init__(self, x: float, y: float, z: float):
        self.index = None
        self.index_dof = np.array([None, None, None])
        self.coordinates = np.array([x, y, z])
        self.x_disp_dof = True
        self.y_disp_dof = True
        self.z_rot_dof = True

        self.displacements = None
        self.velocities = None
        self.accelerations = None
        self.force = None

        self.model_parts = []

    def set_dof(self, dof_idx, is_active):
        """
        Sets a degree of freedom of the node on active or inactive

        :param dof_idx: index of the degree of freedom, 0 => x disp; 1 => y disp; 2 => z rot
        :param is_active: Boolean which is true if nodal degree of freedom should be active
        :return:
        """
        if dof_idx == 0:
            self.x_disp_dof = is_active
        elif dof_idx == 1:
            self.y_disp_dof = is_active
        elif dof_idx == 2:
            self.z_rot_dof = is_active

    def assign_result(self, displacements: np.ndarray, velocities: np.ndarray, accelerations: np.ndarray,
                      force: np.ndarray = None):
        """
        Assigns results to the node

        :param displacements: numpy array of displacements over time
        :param velocities: numpy array of velocities over time
        :param accelerations: numpy array of accelerations over time
        :param force: numpy array of forces over time
        :return:
        """

        # number of possible degrees of freedom at node
        ndof = 3  # todo increase for 3d

        # initialise arrays
        self.displacements = np.zeros((displacements.shape[0], ndof), dtype=float)
        self.velocities = np.zeros((velocities.shape[0], ndof), dtype=float)
        self.accelerations = np.zeros((accelerations.shape[0], ndof), dtype=float)

        # create a mask of the active degrees of freedom
        mask = [bool(self.x_disp_dof), bool(self.y_disp_dof), bool(self.z_rot_dof)]

        # assign results
        self.displacements[:, mask] = displacements[:, :]
        self.velocities[:, mask] = velocities[:, :]
        self.accelerations[:, mask] = accelerations[:, :]

        # if force results is available, assign the force to the node
        if force is not None:
            self.force = np.empty((accelerations.shape[0], ndof), dtype=float)
            self.force[:, mask] = force[:, :]

    def __eq__(self, other: Node):
        """
        Custom equality check. Nodes are considered equal if each coordinate is within the tolerance. If nodes are equal
        the active degrees of freedom for each node are combined.

        :param other: other node to be checked for equality with respect to current node
        :return:
        """

        # set absolute tolerance
        abs_tol = 1e-9

        # return not equal if any coordinate in other node is different from current node
        for idx, coordinate in enumerate(self.coordinates):
            if abs(coordinate - other.coordinates[idx]) > abs_tol:
                return False

        # if nodes are at equal location, combine degree of freedom indices

        # indices of the active degrees of freedom are set to the values of the other node.
        mask = other.index_dof != None # inequality check with "!=" is required, check with "is None" is not possible.
        self.index_dof[mask] = other.index_dof[mask]

        # Degrees of freedom are true if degree of freedom in any node is true
        self.x_disp_dof = bool(self.x_disp_dof + other.x_disp_dof)
        self.y_disp_dof = bool(self.y_disp_dof + other.y_disp_dof)
        self.z_rot_dof = bool(self.z_rot_dof + other.z_rot_dof)


        # combine all model parts of the current node and other node
        self.model_parts = self.model_parts + other.model_parts
        return True

    @property
    def ndof(self):
        """
        Number of active degrees of freedom at node.

        :return:
        """
        return self.x_disp_dof + self.z_rot_dof + self.y_disp_dof


class Element:
    """
       Class which contains an Element.

       :Attributes:

           - :self.index:              index of the element within the mesh
           - :self.nodes:              All the nodes which belong to the element
           - :self.self.model_parts:   List of model parts which are connected to current element
           - :self.force:              Numpy array of force in the element over time for each degree of freedom
    """

    def __init__(self, nodes):
        """
        :param nodes: All the nodes which belong to the element
        """
        self.index = None
        self.nodes = nodes
        self.model_parts = []
        self.force: np.ndarray = None

    def __eq__(self, other):
        """
        Custom equality check. Elements are considered equal if each node of the element is considered equal. If nodes
        are equal, the model parts for each element are combined.

        :param other: other element to be checked for equality with respect to current element
        :return:
        """
        for node in self.nodes:
            if node not in other.nodes:
                return False

        self.model_parts = self.model_parts + other.model_parts

        return True

    def add_model_part(self, model_part):
        """
        Adds a model part to the current element, also adds the model part to the connected nodes.

        :param model_part: model part
        :return:
        """
        if model_part not in self.model_parts:
            self.model_parts.append(model_part)
            for node in self.nodes:
                node.model_parts.append(model_part)

    def assign_force(self, force, mask):
        """
        Assigns force results to the current element
        :param force: Force results in the element on all the active degrees of freedom
        :param mask: mask of the active degrees of freedom
        :return:
        """

        ndof = 3  # todo increase for 3d

        self.force = np.zeros((force.shape[0], ndof), dtype=float)
        self.force[:, mask] = force[:, :]


class Mesh:
    """
       Class which contains a Mesh.

       :Attributes:

           - :self.nodes:              All the nodes which belong to the mesh
           - :self.elements:           All the elements which belong to the mesh
    """
    def __init__(self):
        self.nodes = np.array([])
        self.elements = np.array([])

    def reorder_node_ids(self):
        """
        Reorders the indexes of all the nodes in the mesh. The index order is equal to the order of the nodes in the
        self.nodes array.

        :return:
        """
        for idx, node in enumerate(self.nodes):
            node.index = idx

    def reorder_element_ids(self):
        """
        Reorders the indexes of all the elements in the mesh. The index order is equal to the order of the elements in
        the self.elements array.

        :return:
        """
        for idx, element in enumerate(self.elements):
            element.index = idx

    def add_unique_nodes_to_mesh(self, nodes):
        """
        Checks if node is unique and adds the node to the mesh if the node is unique.

        :param nodes: Nodes to be added to the mesh, if unique.
        :return:
        """
        for node in nodes:
            if node not in self.nodes:
                self.nodes = np.append(self.nodes, [node])

    def add_unique_elements_to_mesh(self, elements):
        """
        Checks if element is unique and adds the element to the mesh if the element is unique.

        :param elements: Elements to be added to the mesh, if unique.
        :return:
        """
        for element in elements:
            if element not in self.elements:
                self.elements = np.append(self.elements, element)
