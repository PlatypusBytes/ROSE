import numpy as np
import src.utils as utils

# todo move this class
class Material:
    def __init__(self):
        self.youngs_modulus = None
        self.poisson_ratio = None
        self.density = None

    @property
    def shear_modulus(self):
        return self.youngs_modulus / (2 * (1 + self.poisson_ratio))


# todo move this class
class Section:
    def __init__(self):
        self.area = None  # [m^2]
        self.sec_moment_of_inertia = None  # [m^4]
        self.shear_factor = (
            0  # shear factor (kr=0 - Euler-Bernoulli beam, kr>0 - Timoshenko beam)
        )


class ModelPart:
    """
    The model part has to consist of the same element types
    """

    def __init__(self):
        self.name = ""
        self.nodes = np.array([])
        self.elements = np.array([])
        self.total_n_dof = None

    @property
    def normal_dof(self):
        return False

    @property
    def y_disp_dof(self):
        return False

    @property
    def z_disp_dof(self):
        return False

    @property
    def x_rot_dof(self):
        return False

    @property
    def y_rot_dof(self):
        return False

    @property
    def z_rot_dof(self):
        return False

    def initialize(self):
        self.calculate_total_n_dof()


    def update(self):
        pass

    def set_geometry(self):
        pass

    def set_aux_force_vector(self):
        pass

    def calculate_total_n_dof(self):
        self.total_n_dof = len(self.nodes) * (
            self.normal_dof + self.y_disp_dof + self.z_disp_dof + self.x_rot_dof + self.y_rot_dof + self.z_rot_dof
        )


class ElementModelPart(ModelPart):
    def __init__(self):
        super(ElementModelPart, self).__init__()
        self.aux_stiffness_matrix = None
        self.aux_damping_matrix = None
        self.aux_mass_matrix = None

        self._normal_shape_functions = None
        self._y_shape_functions = None
        self._z_rot_shape_functions = None

    def initialize(self):
        self.set_aux_stiffness_matrix()
        self.set_aux_mass_matrix()

        # import that damping matrix is set last, as rayleigh damping needs mass and stiffness
        self.set_aux_damping_matrix()

    @property
    def normal_shape_functions(self):
        return self._normal_shape_functions

    @property
    def y_shape_functions(self):
        return self._y_shape_functions

    @property
    def z_rot_shape_functions(self):
        return self._z_rot_shape_functions




    def set_normal_shape_functions(self, x):
        pass

    def set_y_shape_functions(self, x):
        pass

    def set_z_rot_shape_functions(self, x):
        pass

    def set_aux_stiffness_matrix(self):
        pass

    def set_aux_damping_matrix(self):
        pass

    def set_aux_mass_matrix(self):
        pass


class RodElementModelPart(ElementModelPart):
    def __init__(self):
        super().__init__()
        self.stiffness = None
        self.damping = None
        self.length_element = None

    # todo fix this, for now only vertical rod elements are possible
    # @property
    # def normal_dof(self):
    #     return True

    @property
    def y_disp_dof(self):
        return True

    def set_aux_stiffness_matrix(self):
        self.aux_stiffness_matrix = np.zeros((2, 2))
        self.aux_stiffness_matrix[0, 0] = self.stiffness
        self.aux_stiffness_matrix[1, 0] = -self.stiffness
        self.aux_stiffness_matrix[0, 1] = -self.stiffness
        self.aux_stiffness_matrix[1, 1] = self.stiffness

    def set_aux_damping_matrix(self):
        self.aux_damping_matrix = np.zeros((2, 2))
        self.aux_damping_matrix[0, 0] = self.damping
        self.aux_damping_matrix[1, 0] = -self.damping
        self.aux_damping_matrix[0, 1] = -self.damping
        self.aux_damping_matrix[1, 1] = self.damping

    def initialize_shape_functions(self):
        self._normal_shape_functions = np.zeros(2)

    def set_normal_shape_functions(self, x):
        self._normal_shape_functions[0] = 1 - x / self.length_element
        self._normal_shape_functions[1] = x / self.length_element


class TimoshenkoBeamElementModelPart(ElementModelPart):
    def __init__(self):
        super().__init__()

        self.material = Material()
        self.section = Section()

        self.__timoshenko_factor = 0

        self.length_element = None
        self.nodal_ndof = 3

        self.damping_ratio = None
        self.radial_frequency_one = None
        self.radial_frequency_two = None

    @property
    def normal_dof(self):
        return True

    @property
    def y_disp_dof(self):
        return True

    @property
    def z_rot_dof(self):
        return True

    @property
    def timoshenko_factor(self):
        return self.__timoshenko_factor

    def calculate_timoshenko_factor(self):
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

    def __set_translational_aux_mass_matrix(self):
        """
        timoshenko straight beam auxiliar mass matrix associated with translational inertia
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
        timoshenko straight beam auxiliar mass matrix associated with rotatory inertia
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
        timoshenko straight beam auxiliar mass matrix
        :return:
        """
        self.aux_mass_matrix = (
            self.__set_translational_aux_mass_matrix()
            + self.__set_rotational_aux_mass_matrix()
        )

    def set_aux_stiffness_matrix(self):
        """
        timoshenko straight beam auxiliar stiffness matrix
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

        self.aux_stiffness_matrix = self.aux_stiffness_matrix.dot(constant)
        self.aux_stiffness_matrix = utils.reshape_aux_matrix(
            2, [True, True, True], self.aux_stiffness_matrix
        )

    def __calculate_rayleigh_damping_factors(self):
        """
        Calculate rayleigh damping coefficients
        :return:
        """
        constant = (
            2
            * self.damping_ratio
            / (self.radial_frequency_one + self.radial_frequency_two)
        )
        a0 = self.radial_frequency_one * self.radial_frequency_two * constant
        a1 = constant
        return a0, a1

    def set_aux_damping_matrix(self):
        """
        Damping matrix is calculated with the assumption of Rayleigh damping
        :return:
        """
        a0, a1 = self.__calculate_rayleigh_damping_factors()
        self.aux_damping_matrix = self.aux_mass_matrix.dot(
            a0
        ) + self.aux_stiffness_matrix.dot(a1)

    def initialize_shape_functions(self):
        self._normal_shape_functions = np.zeros(2)
        self._y_shape_functions = np.zeros(4)
        self._z_rot_shape_functions = np.zeros(4)

    def set_normal_shape_functions(self, x):
        self._normal_shape_functions[0] = 1 - x / self.length_element
        self._normal_shape_functions[1] = x / self.length_element

    def set_y_shape_functions(self, x):
        """
        Sets y shape functions of the timoshenko beam element

        B.S.Gan, An Isogeometric Approach to Beam Structures, Chapter 3
        DOI10.1007/978-3-319-56493-7_3
        :param x:
        :return:
        """
        l = self.length_element
        phi = self.__timoshenko_factor
        constant = 1 / (1 + phi)

        self._y_shape_functions[0] = constant * (
            1
            + 2 * (x / l) ** 3
            - 3 * (x / l) ** 2
            + phi * (1 - x / l)
        )
        self._y_shape_functions[1] = constant * (
            x
            + (x ** 3 / l ** 2)
            - 2 * (x ** 2 / l)
            + phi
            / 2
            * (x / l - (x / l) ** 2)
        )
        self._y_shape_functions[2] = constant * (
            -2 * (x / l) ** 3
            + 3 * (x / l) ** 2
            + phi * (x / l)
        )
        self._y_shape_functions[3] = constant * (
            (x ** 3 / l ** 2)
            - ((x ** 2) / l)
            + phi
            / 2
            * ((x / l) ** 2 - (x / l))
        )

    def set_z_rot_shape_functions(self, x):
        # todo set z_rot shape functions
        pass

    def initialize(self):
        self.initialize_shape_functions()
        super(TimoshenkoBeamElementModelPart, self).initialize()


class ConditionModelPart(ModelPart):
    def __init__(self):
        super(ConditionModelPart, self).__init__()


class ConstraintModelPart(ConditionModelPart):
    def __init__(self, normal_dof=False, y_disp_dof=False, z_rot_dof=False):
        super(ConstraintModelPart, self).__init__()
        self.__normal_dof = normal_dof
        self.__y_disp_dof = y_disp_dof
        self.__z_rot_dof = z_rot_dof

    @property
    def normal_dof(self):
        return self.__normal_dof

    @property
    def y_disp_dof(self):
        return self.__y_disp_dof

    @property
    def z_rot_dof(self):
        return self.__z_rot_dof

    def set_scalar_condition(self):
        pass

    def set_constraint_condition(self):
        for node in self.nodes:
            node.normal_dof = self.__normal_dof
            node.z_rot_dof = self.__z_rot_dof
            node.y_disp_dof = self.__y_disp_dof
