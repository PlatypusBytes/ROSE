import numpy as np
import rose.model.utils as utils
import logging
from typing import List
import abc


# todo move this class
class Material:
    """
    This class contains the material of the rail.

    :Attributes:

        - :self.youngs_modulus:  youngs modulus of the material [N/m2]
        - :self.poisson_ratio:   poisson ratio of the material [-]
        - :self.density:         density of the material [kg/m3]
    """
    def __init__(self):
        self.youngs_modulus: float = None
        self.poisson_ratio: float = None
        self.density: float = None

    def validate_input(self):
        """
        Validates material input. Checks if youngs modulus, poisson ratio and density are defined.
        :return:
        """
        if self.youngs_modulus is None:
            logging.error("Youngs modulus not defined")
        if self.poisson_ratio is None:
            logging.error("Poissons ratio not defined")
        if self.density is None:
            logging.error("Density not defined")

    @property
    def shear_modulus(self):
        """
        Shear modulus of the material [N/m2]

        ..math
            G = E/(2*(1+\nu))
        :return:
        """
        return self.youngs_modulus / (2 * (1 + self.poisson_ratio))


# todo move this class
class Section:
    """
    This class contains the cross section of the rail.

    :Attributes:

        - :self.area:                    section area [m^2]
        - :self.sec_moment_of_inertia:   second moment of inertia [m^4]
        - :self.shear_factor:            shear factor (kr=0 - Euler-Bernoulli beam, kr>0 - Timoshenko beam)
    """
    def __init__(self):

        self.area: float = None  # [m^2]
        self.sec_moment_of_inertia: float = None  # [m^4]
        self.shear_factor: float = (
            0  # shear factor (kr=0 - Euler-Bernoulli beam, kr>0 - Timoshenko beam)
        )

    def validate_input(self):
        if self.area is None:
            logging.error("Area not defined")
        if self.sec_moment_of_inertia is None:
            logging.error("Second moment of inertia not defined")
        if self.shear_factor is None:
            logging.error("Shear factor not defined")


class ModelPart:
    """
    Base model part class. This class is the base for boundary conditions and element model parts.
    The model part has to consist of the same element types.


    :Attributes:

        - :self.name:            name of the model part [-]
        - :self.nodes:           all nodes within the model part [-]
        - :self.elements:        all elements within the model part [-]
        - :self.total_n_dof:     total number of degree of freedoms within the model part [-]
    """
    def __init__(self):
        self.name = ""
        self.nodes = np.array([])
        self.elements = np.array([])
        self.total_n_dof = None

    @property
    def normal_dof(self):
        """

        :return: is normal degree of freedom activated
        """
        return False

    @property
    def y_disp_dof(self):
        """

        :return: is y degree of freedom activated
        """
        return False

    @property
    def z_disp_dof(self):
        """

        :return: is z degree of freedom activated
        """
        return False

    @property
    def x_rot_dof(self):
        """

        :return: is rotation around local x axis activated
        """
        return False

    @property
    def y_rot_dof(self):
        """

        :return: is rotation around local y axis activated
        """
        return False

    @property
    def z_rot_dof(self):
        """

        :return: is rotation around local z axis activated
        """
        return False

    def validate_input(self):
        """
        Validates model part input. Checks if nodes and elements are defined in a list or np array
        :return:
        """
        if not isinstance(self.nodes, (List, np.ndarray)):
            logging.error("Nodes in model part are not defined in a list or ndarray")

        if not isinstance(self.elements, (List, np.ndarray)):
            logging.error("Nodes in model part are not defined in a list or ndarray")

    def initialize(self):
        """
        Inititalises model parts. Calculates total number of degree of freedom and maps model part on each element

        :return:
        """
        self.calculate_total_n_dof()
        [element.add_model_part(self) for element in self.elements]

    def update(self):
        pass

    def set_geometry(self):
        pass

    def set_aux_force_vector(self):
        pass

    def calculate_total_n_dof(self):
        """
        Calculates total number of degree of freedom. Number of nodes times number of degree of freedom per node.
        :return:
        """
        self.total_n_dof = len(self.nodes) * (
            self.normal_dof + self.y_disp_dof + self.z_disp_dof + self.x_rot_dof + self.y_rot_dof + self.z_rot_dof
        )


class ElementModelPart(ModelPart):
    """
    Element model part class. This class is the base for each element model part type.
    The model part has to consist of the same element types. This class bases from
    :class:`~rose.model.model_part.ModelPart`.
    #todo extend for 2d/3d elements.

    :Attributes:

        - :self.aux_stiffness_matrix:           auxiliary local stiffness matrix
        - :self.aux_damping_matrix:             auxiliary local damping matrix
        - :self.aux_mass_matrix:                auxiliary local mass matrix
        - :self.static_force_vector:            local static force vector within the element
    """

    def __init__(self):
        super(ElementModelPart, self).__init__()
        self.aux_stiffness_matrix = None
        self.aux_damping_matrix = None
        self.aux_mass_matrix = None

        self.static_force_vector = None

        self._normal_shape_functions = None
        self._y_shape_functions = None
        self._z_rot_shape_functions = None
        self.__rotation_matrix = None

    def initialize(self):
        """
        Initialises model part. Local stiffness, mass and damping matrices are initialised.
        :return:
        """
        super().initialize()
        self.set_aux_stiffness_matrix()
        self.set_aux_mass_matrix()

        # important that damping matrix is set last, as rayleigh damping needs mass and stiffness
        self.set_aux_damping_matrix()

    @property
    def normal_shape_functions(self):
        """
        Shape functions for the local normal axis
        :return:
        """
        return self._normal_shape_functions

    @property
    def y_shape_functions(self):
        """
        Shape functions for the local shear axis
        :return:
        """
        return self._y_shape_functions

    @property
    def z_rot_shape_functions(self):
        """
        Shape functions for the local rotation around the z-axis
        :return:
        """
        return self._z_rot_shape_functions

    @property
    def rotation_matrix(self):
        """
        Rotation matrix which is used to rotate from the local to the global system.

        :return:
        """
        return self.__rotation_matrix

    def set_rotation_matrix(self, rotation, dim):
        """
        Sets the rotation matrix. This function is meant as an abstract method

        :param rotation: rotation of the element
        :param dim: dimensions of the element
        :return:
        """
        pass

    def set_normal_shape_functions(self, x):
        """
        Sets the local normal shape functions at distance x. This function is meant as an abstract method

        :param x: local distance from the first node to the desired locatin within the element
        :return:
        """
        pass

    def set_y_shape_functions(self, x):
        """
        Sets the local shear shape functions at distance x. This function is meant as an abstract method

        :param x: local distance from the first node to the desired locatin within the element
        :return:
        """
        pass

    def set_z_rot_shape_functions(self, x):
        """
        Sets the rotation around the z-axis shape functions at distance x. This function is meant as an abstract method

        :param x: local distance from the first node to the desired location within the element
        :return:
        """
        pass

    @abc.abstractmethod
    def set_aux_stiffness_matrix(self):
        """
        Sets the local auxiliary stiffness matrix. This function is meant as an abstract method
        :return:
        """
        pass

    @abc.abstractmethod
    def set_aux_damping_matrix(self):
        """
        Sets the local auxiliary damping matrix. This function is meant as an abstract method
        :return:
        """
        pass

    @abc.abstractmethod
    def set_aux_mass_matrix(self):
        """
        Sets the local auxiliary mass matrix. This function is meant as an abstract method
        :return:
        """
        pass


class RodElementModelPart(ElementModelPart):
    """
    Rod element model part class. This class bases from
    :class:`~rose.model.model_part.ElementModelPart`. A rod element only interacts in normal direction.

    :Attributes:

        - :self.mass:                           mass of the rod element
        - :self.stiffness:                      stiffness of the rod element
        - :self.damping :                       of the rod element
        - :self.length_element:                 length of the rod element
    """
    def __init__(self):
        super().__init__()
        self.mass = None
        self.stiffness = None
        self.damping = None
        self.length_element = None

    @property
    def normal_dof(self):
        """

        :return: is normal degree of freedom activated
        """
        return True

    @property
    def rotation_matrix(self):
        """
        Rotation matrix which is used to rotate from the local to the global system.

        :return:
        """
        return self.__rotation_matrix

    def validate_input(self):
        """
        Validates input. Checks if rod model part contains a stiffness, elements and nodes
        :return:
        """
        if self.stiffness is None:
            logging.error("Stiffness of rod element is not defined")

        if len(self.elements) == 0:
            logging.error("Rod model part does not contain elements")

        if len(self.nodes) == 0:
            logging.error("Rod model part does not contain nodes")

    def set_rotation_matrix(self, rotation, dim):
        """
        Sets 2D rotation matrix

        :param rotation: rotation in the global system
        :param dim:     dimension of the work space
        :return:
        """
        if dim ==2:
            self.__rotation_matrix = np.zeros((6, 6))
            self.__rotation_matrix [[0, 1, 3, 4], [0, 1, 3, 4]] = np.cos(rotation)
            self.__rotation_matrix [[0, 3], [1, 4]] = np.sin(rotation)
            self.__rotation_matrix [[1, 4], [0, 3]] = -np.sin(rotation)
            self.__rotation_matrix [[2, 5], [2, 5]] = 1

    def set_aux_mass_matrix(self):
        """
        Sets the local auxiliary mass matrix for the rod element. Only if the rod contains a mass value
        :return:
        """
        if self.mass is not None:
            self.aux_mass_matrix = np.zeros((2, 2))
            self.aux_mass_matrix[0, 0] = 2
            self.aux_mass_matrix[1, 0] = 1
            self.aux_mass_matrix[0, 1] = 1
            self.aux_mass_matrix[1, 1] = 2
            self.aux_mass_matrix = self.aux_mass_matrix.dot(self.mass/6)

    def set_aux_stiffness_matrix(self):
        """
        Sets the local auxiliary stiffness matrix for the rod element. Only if the rod contains a stiffness value.
        :return:
        """
        if self.stiffness is not None:
            self.aux_stiffness_matrix = np.zeros((2, 2))
            self.aux_stiffness_matrix[0, 0] = self.stiffness
            self.aux_stiffness_matrix[1, 0] = -self.stiffness
            self.aux_stiffness_matrix[0, 1] = -self.stiffness
            self.aux_stiffness_matrix[1, 1] = self.stiffness

    def set_aux_damping_matrix(self):
        """
        Sets the local auxiliary damping matrix for the rod element. Only if the rod contains a damping value.
        :return:
        """
        if self.damping is not None:
            self.aux_damping_matrix = np.zeros((2, 2))
            self.aux_damping_matrix[0, 0] = self.damping
            self.aux_damping_matrix[1, 0] = -self.damping
            self.aux_damping_matrix[0, 1] = -self.damping
            self.aux_damping_matrix[1, 1] = self.damping

    def initialize_shape_functions(self):
        """
        Initialises normal shape functions for the rod element.
        :return:
        """
        self._normal_shape_functions = np.zeros(2)

    def set_normal_shape_functions(self, x):
        """
        Sets the local normal shape functions for the rod element at distance x.

        :param x: local distance from the first node to the desired location within the element
        :return:
        """
        self._normal_shape_functions[0] = 1 - x / self.length_element
        self._normal_shape_functions[1] = x / self.length_element


class TimoshenkoBeamElementModelPart(ElementModelPart):
    """
    Timoshenko beam element model part class. This class bases from
    :class:`~rose.model.model_part.ElementModelPart`. This class can be used as an euler beam, if a timoshenko factor of
    0 is used.

    :Attributes:

        - :self.material:           material of the timoshenko beam
        - :self.section:            cross section of the timoshenko beam
        - :self.length_element:     length of the timoshenko beam element
        - :self.nodal_ndof:         number of nodal degrees of freedom in the timoshenko beam (3 in a 2D space)
        - :self.mass:               mass of the timoshenko beam element
    """

    def __init__(self):
        super().__init__()

        self.material = Material()
        self.section = Section()

        self.length_element = None
        self.nodal_ndof = 3

        self.mass = None

        self._normal_shape_functions = np.zeros(2)
        self._y_shape_functions = np.zeros(4)
        self._z_rot_shape_functions = np.zeros(4)

        self.__timoshenko_factor = 0
        self.spring_stiffness1 = 0
        self.spring_stiffness2 = 0

    @property
    def normal_dof(self):
        """

        :return: is normal degree of freedom activated
        """
        return True

    @property
    def y_disp_dof(self):
        """

        :return: is y degree of freedom activated
        """
        return True

    @property
    def z_rot_dof(self):
        """

        :return: is rotation around z-axis degree of freedom activated
        """
        return True

    @property
    def timoshenko_factor(self):
        """
        Ratio of beam bending to shear stiffness.

        :return:
        """
        return self.__timoshenko_factor

    @property
    def rotation_matrix(self):
        """
        Rotation matrix which is used to rotate from the local to the global system.

        :return:
        """
        return self.__rotation_matrix

    def validate_input(self):
        """
        Validates Timoshenko beam model part. Validates material, validates cross section. Checks if model part contains
        nodes and elements.
        :return:
        """
        if not isinstance(self.material, Material):
            logging.error("Material not defined")
        else:
            self.material.validate_input()

        if not isinstance(self.section, Section):
            logging.error("Section not defined")
        else:
            self.section.validate_input()

        if len(self.elements) == 0:
            logging.error("Timoshenko beam model part does not contain elements")

        if len(self.nodes) == 0:
            logging.error("Timoshenko beam model part does not contain nodes")

    def calculate_mass(self):
        """
        Calculates the mass per meter of the Timoshenko beam from the cross section and the material density.

        :return:
        """
        self.mass = self.section.area * self.material.density

    def calculate_timoshenko_factor(self):
        """
        Calculates the ratio of the beam bending to shear stiffness.

        :return:
        """
        if self.section.shear_factor > 0:
            self.__timoshenko_factor = (
                12
                * self.material.youngs_modulus
                * self.section.sec_moment_of_inertia
                / (
                    self.length_element ** 2
                    * self.material.shear_modulus
                    * self.section.area
                    * self.section.shear_factor
                )
            )

    def set_rotation_matrix(self, rotation, dim):
        """
        Sets 2D rotation matrix of the timoshenko beam
        :param rotation: rotation in the global system
        :param dim: dimension of the workspace
        :return:
        """
        if dim ==2:
            self.__rotation_matrix = np.zeros((6, 6))
            self.__rotation_matrix[[0, 1, 3, 4], [0, 1, 3, 4]] = np.cos(rotation)
            self.__rotation_matrix[[0, 3], [1, 4]] = np.sin(rotation)
            self.__rotation_matrix[[1, 4], [0, 3]] = -np.sin(rotation)
            self.__rotation_matrix[[2, 5], [2, 5]] = 1

    def __set_translational_aux_mass_matrix(self):
        """
        Timoshenko straight beam auxiliary mass matrix associated with translational inertia
        :return:
        """
        phi = self.__timoshenko_factor
        l = self.length_element

        constant = (
            self.material.density * self.section.area * l / (210 * (1 + phi) ** 2)
        )

        if self.nodal_ndof == 3:
            trans_aux_mass_matrix = np.zeros((6, 6))

            trans_aux_mass_matrix[[0, 3], [0, 3]] = 70 * (1 + phi) ** 2
            trans_aux_mass_matrix[[3, 0], [0, 3]] = 35 * (1 + phi) ** 2

            trans_aux_mass_matrix[[1, 4], [1, 4]] = 70 * phi ** 2 + 147 * phi + 78
            trans_aux_mass_matrix[[1, 4], [4, 1]] = 35 * phi ** 2 + 63 * phi + 27

            trans_aux_mass_matrix[[1, 2], [2, 1]] = (
                (35 * phi ** 2 + 77 * phi + 44) * l / 4
            )

            trans_aux_mass_matrix[[1, 5], [5, 1]] = (
                -(35 * phi ** 2 + 63 * phi + 26) * l / 4
            )

            trans_aux_mass_matrix[[2, 5], [2, 5]] = (7 * phi ** 2 + 14 * phi + 8) * (
                l ** 2 / 4
            )
            trans_aux_mass_matrix[[2, 5], [5, 2]] = -(7 * phi ** 2 + 14 * phi + 6) * (
                l ** 2 / 4
            )

            trans_aux_mass_matrix[[2, 4], [4, 2]] = (35 * phi ** 2 + 63 * phi + 26) * (
                l / 4
            )

            trans_aux_mass_matrix[[4, 5], [5, 4]] = -(35 * phi ** 2 + 77 * phi + 44) * (
                l / 4
            )

            trans_aux_mass_matrix = utils.reshape_aux_matrix(
                2, [True, True, True], trans_aux_mass_matrix
            )
            return trans_aux_mass_matrix.dot(constant)
        return None

    def __set_rotational_aux_mass_matrix(self):
        """
        Timoshenko straight beam auxiliary mass matrix associated with rotatory inertia
        :return:
        """
        phi = self.__timoshenko_factor
        l = self.length_element

        constant = (
            self.material.density
            * self.section.sec_moment_of_inertia
            / (30 * (1 + phi) ** 2 * l)
        )

        if self.nodal_ndof == 3:
            rot_aux_mass_matrix = np.zeros((6, 6))

            rot_aux_mass_matrix[[1, 4], [1, 4]] = 36
            rot_aux_mass_matrix[[1, 4], [4, 1]] = -36

            rot_aux_mass_matrix[[1, 1, 2, 5], [2, 5, 1, 1]] = -(15 * phi - 3) * l

            rot_aux_mass_matrix[[2, 5], [2, 5]] = (10 * phi ** 2 + 5 * phi + 4) * l ** 2
            rot_aux_mass_matrix[[2, 5], [5, 2]] = (5 * phi ** 2 - 5 * phi - 1) * l ** 2

            rot_aux_mass_matrix[[2, 4, 4, 5], [4, 2, 5, 4]] = (15 * phi - 3) * l

            rot_aux_mass_matrix = utils.reshape_aux_matrix(
                2, [True, True, True], rot_aux_mass_matrix
            )
            return rot_aux_mass_matrix.dot(constant)
        return None

    def set_aux_mass_matrix(self):
        """
        Timoshenko straight beam auxiliar mass matrix. If timoshenko factor is equal to 0, no rotational part of the
        mass matrix is added. This is equivalent to an euler beam.
        :return:
        """
        self.aux_mass_matrix = self.__set_translational_aux_mass_matrix()
        if self.timoshenko_factor != 0:
            self.aux_mass_matrix += self.__set_rotational_aux_mass_matrix()


    def __set_rigid_part_stiffness_matrix(self):
        """
        sets part of the stiffness matrix which sets bending rigidity
        :return:
        """
        l = self.length_element
        EI = self.material.youngs_modulus * self.section.sec_moment_of_inertia
        s1 = self.spring_stiffness1  # springstiffness 1
        s2 = self.spring_stiffness2  # springstiffness 2
        # set fixity factor
        alpha1 = 1/(1+3*EI/s1*l)
        alpha2 = 1/(1+3*EI/s2*l)

        rigid_mat = np.zeros((6, 6))
        rigid_mat[[0, 3], [0, 3]] = 1
        rigid_mat[[3, 0], [0, 3]] = 1
        rigid_mat[[1, 4, 1, 4], [1, 4, 4, 1]] = (alpha1+alpha2 + alpha1*alpha2)/(4-alpha1*alpha2)
        rigid_mat[[1, 2, 2, 4], [2, 1, 4, 2]] = (2 * alpha1 + alpha1*alpha2)/(4-alpha1*alpha2)
        rigid_mat[[1, 5, 4, 5], [5, 1, 5, 4]] = (2 * alpha2 + alpha1 * alpha2) / (4 - alpha1 * alpha2)
        rigid_mat[2, 2] = 3 * alpha1 / (4 - alpha1*alpha2)
        rigid_mat[5, 5] = 3 * alpha2 / (4 - alpha1 * alpha2)
        rigid_mat[[2, 5], [5, 2]] = 3*alpha1*alpha2/(4-alpha1*alpha2)
        return rigid_mat

    def set_aux_stiffness_matrix(self):
        """
        Timoshenko straight beam auxiliary stiffness matrix
        :return:
        """
        phi = self.__timoshenko_factor
        l = self.length_element

        EI = self.material.youngs_modulus * self.section.sec_moment_of_inertia
        constant = EI / ((1 + phi) * l ** 3)

        self.aux_stiffness_matrix = np.zeros((6, 6))
        self.aux_stiffness_matrix[[0, 3], [0, 3]] = (
            self.section.area / self.section.sec_moment_of_inertia * (1 + phi) * l ** 2
        )
        self.aux_stiffness_matrix[[3, 0], [0, 3]] = (
            -self.section.area / self.section.sec_moment_of_inertia * (1 + phi) * l ** 2
        )

        self.aux_stiffness_matrix[[1, 4], [1, 4]] = 12
        self.aux_stiffness_matrix[[1, 4], [4, 1]] = -12

        self.aux_stiffness_matrix[[1, 1, 2, 5], [2, 5, 1, 1]] = 6 * l
        self.aux_stiffness_matrix[[2, 4, 4, 5], [4, 2, 5, 4]] = -6 * l

        self.aux_stiffness_matrix[[2, 5], [2, 5]] = (4 + phi) * l ** 2
        self.aux_stiffness_matrix[[2, 5], [5, 2]] = (2 - phi) * l ** 2

        # add rigidity if spring stiffness is applied
        if self.spring_stiffness1 + self.spring_stiffness2 > 1e-10:
            self.aux_stiffness_matrix = self.aux_stiffness_matrix * self.__set_rigid_part_stiffness_matrix()

        self.aux_stiffness_matrix = self.aux_stiffness_matrix.dot(constant)
        self.aux_stiffness_matrix = utils.reshape_aux_matrix(
            2, [True, True, True], self.aux_stiffness_matrix
        )

    def set_aux_damping_matrix(self):
        """
        Timoshenko straight beam auxiliary damping matrix
        :return:
        """
        self.aux_damping_matrix = np.zeros((6, 6))

    def initialize_shape_functions(self):
        """
        Initialises timoshenko beam shape functions as numpy arrays.
        :return:
        """
        self._normal_shape_functions = np.zeros(2)
        self._y_shape_functions = np.zeros(4)
        self._z_rot_shape_functions = np.zeros(4)

    def set_normal_shape_functions(self, x):
        """
        Sets the local normal shape functions for the timoshenko beam element at distance x.

        :param x: local distance from the first node to the desired location within the element
        :return:
        """
        self._normal_shape_functions[0] = 1 - x / self.length_element
        self._normal_shape_functions[1] = x / self.length_element

    def set_y_shape_functions(self, x):
        """
        |Sets y shape functions of the Timoshenko beam element

        |B.S.Gan, An Isogeometric Approach to Beam Structures, Chapter 3
        DOI10.1007/978-3-319-56493-7_3

        :param x: local coordinate of the element [m]
        :return:
        """
        l = self.length_element
        phi = self.__timoshenko_factor
        constant = 1 / (1 + phi)
        x_l = x/l
        x_l2 = x_l**2

        x2 = x**2
        x3 = x**3

        self._y_shape_functions[0] = constant * (
            1
            + 2 * x_l ** 3
            - 3 * x_l2
            + phi * (1 - x_l)
        )
        self._y_shape_functions[1] = constant * (
            x
            + (x3 / l ** 2)
            - 2 * (x2 / l)
            + phi
            / 2
            * (x_l - x_l2)
        )
        self._y_shape_functions[2] = constant * (
            -2 * x_l ** 3
            + 3 * x_l2
            + phi * x_l
        )
        self._y_shape_functions[3] = constant * ((x3 / l ** 2)- (x2 / l)+ phi/ 2* (x_l2 - x_l)
        )

    def set_z_rot_shape_functions(self, x):
        # todo set z_rot shape functions
        pass

    def initialize(self):
        """
        Initialises timoshenko beam model part. Calculates the timoshenko factor; calculates the mass of the beam;
        initialises shape functions; initialises parent class.
        :return:
        """
        self.calculate_timoshenko_factor()
        self.calculate_mass()
        self.initialize_shape_functions()
        super(TimoshenkoBeamElementModelPart, self).initialize()


class ConditionModelPart(ModelPart):
    """
    Condition model part class. This class is the base for each boundary condition model part type. This class bases
    from :class:`~rose.model.model_part.ModelPart`.
    """
    def __init__(self):
        super(ConditionModelPart, self).__init__()


class ConstraintModelPart(ConditionModelPart):
    """
    Constraint model part class. This class is a model part which indicates rotational or translational constraints.
    This class bases from :class:`~rose.model.model_part.ConditionModelPart`.
    """
    def __init__(self, x_disp_dof=False, y_disp_dof=False, z_rot_dof=False):
        super(ConstraintModelPart, self).__init__()
        self.__x_disp_dof = x_disp_dof
        self.__y_disp_dof = y_disp_dof
        self.__z_rot_dof = z_rot_dof

    @property
    def x_disp_dof(self):
        """
        :return: is degree of freedom  in x direction activated
        """
        return self.__x_disp_dof

    @property
    def y_disp_dof(self):
        """
        :return: is degree of freedom  in y direction activated
        """
        return self.__y_disp_dof

    @property
    def z_rot_dof(self):
        """
        :return: is degree of freedom  around z axis activated
        """
        return self.__z_rot_dof

    def set_constraint_condition(self):
        """
        Sets the constraint boundary condition on the nodes.
        :return:
        """
        for node in self.nodes:
            node.x_disp_dof = self.__x_disp_dof
            node.z_rot_dof = self.__z_rot_dof
            node.y_disp_dof = self.__y_disp_dof
