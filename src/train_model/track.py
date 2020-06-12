import numpy as np
from scipy import sparse
import time


class Material():
    def __init__(self):
        self.youngs_modulus = None
        self.poisson_ratio = None
        self.density = None

    @property
    def shear_modulus(self):
        return self.youngs_modulus / (2 * (1 + self.poisson_ratio))


class Section():
    def __init__(self):
        self.area = None  # [m^2]
        self.sec_moment_of_inertia = None  # [m^4]
        self.shear_factor = 0  # shear factor (kr=0 - Euler-Bernoulli beam, kr>0 - Timoshenko beam)

class Rail():
    def __init__(self, n_sleepers):
        self.material = Material()
        self.section = Section()
        self.__n_sleepers = n_sleepers
        self.level = np.zeros((1, self.__n_sleepers))

        self.length_rail = None
        self.mass = None

        self.timoshenko_factor = 0  # ???
        self.n_nodes = None
        self.ndof = None

        self.damping_ratio = None
        self.radial_frequency_one = None
        self.radial_frequency_two = None

        self.aux_mass_matrix = None
        self.global_mass_matrix = None

        self.aux_stiffness_matrix = None
        self.global_stiffness_matrix = None

        self.aux_damping_matrix = None

        self.ndof_per_node = 3

    def set_top_level_to_zero(self):
        self.level = self.level - np.max(self.level)

    def calculate_length_rail(self, distance_between_sleepers):
        self.length_rail = distance_between_sleepers

    def calculate_mass(self):
        self.mass = self.section.area * self.material.density

    def calculate_timoshenko_factor(self):
        if self.section.shear_factor > 0:
            self.timoshenko_factor = 12 * self.material.youngs_modulus * self.section.sec_moment_of_inertia / (
                    self.length_rail ** 2 * self.material.shear_modulus *
                    self.section.area * self.section.shear_factor)

    def calculate_n_dof(self):
        self.n_nodes = self.__n_sleepers
        self.ndof = self.n_nodes * self.ndof_per_node

    # def calculate_mass_matrix(self):
    #     self.mass_matrix = np.zeros((1, self.ndof))
    #     self.mass_matrix[0, 0] = self.mass * self.length_rail / 2
    #     self.mass_matrix[0, 1:self.ndof - 1] = self.mass * self.length_rail
    #     self.mass_matrix[0, -1] = self.mass * self.length_rail / 2

    def __set_translational_aux_mass_matrix(self):
        """
        timoshenko straight beam auxiliar mass matrix associated with translational inertia
        :return:
        """
        phi = self.timoshenko_factor
        l = self.length_rail

        constant = self.material.density * self.section.area * l / (210 * (1 + phi) ** 2)

        if self.ndof_per_node == 3:
            trans_aux_mass_matrix = np.zeros((6, 6))

            trans_aux_mass_matrix[[0, 3], [0, 3]] = 70 * (1 + phi) ** 2
            trans_aux_mass_matrix[[3, 0], [0, 3]] = 35 * (1 + phi) ** 2

            trans_aux_mass_matrix[[1, 4], [1, 4]] = 70 * phi**2 + 147*phi + 78
            trans_aux_mass_matrix[[1, 4], [4, 1]] = 35 * phi**2 + 63*phi + 27

            trans_aux_mass_matrix[[1, 2], [2, 1]] = (35 * phi ** 2 + 77 * phi + 44) * l / 4

            trans_aux_mass_matrix[[1, 5], [5, 1]] = -(35 * phi ** 2 + 63 * phi + 26) * l / 4

            trans_aux_mass_matrix[[2, 5], [2, 5]] = (7 * phi ** 2 + 14 * phi + 8) * (l**2 / 4)
            trans_aux_mass_matrix[[2, 5], [5, 2]] = -(7 * phi ** 2 + 14 * phi + 6) * (l**2 / 4)

            trans_aux_mass_matrix[[2, 4], [4, 2]] = (35 * phi ** 2 + 63 * phi + 26) * (l / 4)

            trans_aux_mass_matrix[[4, 5], [5, 4]] = -(35 * phi ** 2 + 77 * phi + 44) * (l / 4)

            return trans_aux_mass_matrix.dot(constant)
        return None

    def __set_rotational_aux_mass_matrix(self):
        """
        timoshenko straight beam auxiliar mass matrix associated with rotatory inertia
        :return:
        """
        phi = self.timoshenko_factor
        l = self.length_rail

        constant = self.material.density * self.section.sec_moment_of_inertia / (30 * (1 + phi)**2 * l)

        if self.ndof_per_node == 3:
            rot_aux_mass_matrix = np.zeros((6, 6))

            rot_aux_mass_matrix[[1, 4], [1, 4]] = 36
            rot_aux_mass_matrix[[1, 4], [4, 1]] = -36

            rot_aux_mass_matrix[[1, 1, 2, 5], [2, 5, 1, 1]] = -(15*phi-3)*l

            rot_aux_mass_matrix[[2, 5], [2, 5]] = (10 * phi**2 + 5*phi + 4) * l ** 2
            rot_aux_mass_matrix[[2, 5], [5, 2]] = (5 * phi**2 - 5*phi - 1) * l ** 2

            rot_aux_mass_matrix[[2, 4, 4, 5], [4, 2, 5, 4]] = (15 * phi - 3) * l

            return rot_aux_mass_matrix.dot(constant)
        return None

    def set_aux_mass_matrix(self):
        """
        timoshenko straight beam auxiliar mass matrix
        :return:
        """
        self.aux_mass_matrix = self.__set_translational_aux_mass_matrix() + self.__set_rotational_aux_mass_matrix()

    def set_aux_stiffness_matrix(self):
        """
        timoshenko straight beam auxiliar stiffness matrix
        :return:
        """
        EI = self.material.youngs_modulus * self.section.sec_moment_of_inertia
        constant = EI / ((1 + self.timoshenko_factor) * self.length_rail ** 3)

        if self.ndof_per_node == 3:
            self.aux_stiffness_matrix = np.zeros((6, 6))
            self.aux_stiffness_matrix[[0, 3], [0, 3]] = self.section.area / self.section.sec_moment_of_inertia * \
                                                        (1 + self.timoshenko_factor)
            self.aux_stiffness_matrix[[3, 0], [0, 3]] = -self.section.area / self.section.sec_moment_of_inertia * \
                                                        (1 + self.timoshenko_factor)

            self.aux_stiffness_matrix[[1, 4], [1, 4]] = 12
            self.aux_stiffness_matrix[[1, 4], [4, 1]] = -12

            self.aux_stiffness_matrix[[1, 1, 2, 5], [2, 5, 1, 1]] = 6 * self.length_rail
            self.aux_stiffness_matrix[[2, 4, 4, 5], [4, 2, 5, 4]] = -6 * self.length_rail

            self.aux_stiffness_matrix[[2, 5], [2, 5]] = (4 + self.timoshenko_factor) * self.length_rail ** 2
            self.aux_stiffness_matrix[[2, 5], [5, 2]] = (2 - self.timoshenko_factor) * self.length_rail ** 2

            self.aux_stiffness_matrix = self.aux_stiffness_matrix.dot(constant)


    def __calculate_rayleigh_damping_factors(self):
        """
        Calculate rayleigh damping coefficients
        :return:
        """
        constant = 2 * self.damping_ratio / (self.radial_frequency_one + self.radial_frequency_two)
        a0 = self.radial_frequency_one * self.radial_frequency_two * constant
        a1 = constant
        return a0, a1

    def calculate_rayleigh_damping_matrix(self):
        """
        Damping matrix is calculated with the assumption of Rayleigh damping
        :return:
        """
        a0, a1 = self.__calculate_rayleigh_damping_factors()
        self.aux_damping_matrix =  self.aux_mass_matrix.dot(a0) + self.aux_stiffness_matrix.dot(a1)



class Sleeper:
    def __init__(self):
        self.mass = None
        self.distance_between_sleepers = None

class RailPad:
    def __init__(self):
        self.stiffness = None
        self.damping = None
        self.aux_stiffness_matrix = None
        self.aux_damping_matrix = None

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


class ContactRailWheel:
    def __init__(self):
        self.stiffness = None
        self.damping = None
        self.exponent = None


class ContactSleeperBallast:
    def __init__(self):
        self.stiffness = None
        self.damping = None


class Support:
    def __init__(self, n_sleepers):
        self.linear_stiffness = None
        self.non_linear_stiffness = None
        self.non_linear_exponent = None
        self.initial_voids = None
        self.tensile_stiffness_ballast = None
        self.damping_ratio = None
        self.__n_sleepers = n_sleepers

        self.linear_stiffness_matrix = None
        self.non_linear_stiffness_matrix = None
        self.non_linear_exponent_matrix = None
        self.initial_voids_matrix = None
        self.tensile_stiffness_ballast_matrix = None
        self.damping_ratio_matrix = None

    def initialise_matrices(self):
        self.linear_stiffness_matrix = np.ones((1, self.__n_sleepers)) * self.linear_stiffness
        self.non_linear_stiffness_matrix = np.ones((1, self.__n_sleepers)) * self.non_linear_stiffness
        self.non_linear_exponent_matrix = np.ones((1, self.__n_sleepers)) * self.non_linear_exponent
        self.initial_voids_matrix = np.ones((1, self.__n_sleepers)) * self.initial_voids
        self.tensile_stiffness_ballast_matrix = np.ones((1, self.__n_sleepers)) * self.tensile_stiffness_ballast
        self.damping_ratio_matrix = np.ones((1, self.__n_sleepers)) * self.damping_ratio


class Ballast:
    def __init__(self, n_sleepers):
        self.mass = None
        self.stiffness = None
        self.damping = None
        self.__n_sleepers = n_sleepers

        self.mass_matrix = None
        self.stiffness_matrix = None
        self.damping_matrix = None

    def initialise_matrices(self):
        self.mass_matrix = np.ones((1, self.__n_sleepers - 1)) * self.mass
        self.stiffness_matrix = np.ones((1, self.__n_sleepers - 1)) * self.stiffness
        self.damping_matrix = np.ones((1, self.__n_sleepers - 1)) * self.damping

class Soil:
    def __init__(self):
        self.stiffness = None
        self.damping = None

        self.aux_stiffness_matrix = None
        self.aux_mass_matrix = None

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



class UTrack:
    def __init__(self, n_sleepers):
        self.__n_sleepers = n_sleepers

        self.rail = Rail(n_sleepers)
        self.sleeper = Sleeper()
        self.rail_pads = RailPad()
        self.ballast = Ballast(n_sleepers)
        self.contact_sleeper_ballast = ContactSleeperBallast()
        self.Support = Support(n_sleepers)
        self.contact_rail_wheel = ContactRailWheel()
        self.soil = Soil()

        self.global_mass_matrix = None
        self.global_stiffness_matrix = None
        self.global_damping_matrix = None

        self.force = None
        self.time = np.linspace(0, 10, 1000)

        self.__total_length = None
        self.__n_dof_rail = None
        self.n_dof_track = None

    def set_global_stiffness_matrix(self):
        """
        sparse.csr_matrix((self.ndof, self.ndof))

        :return:
        """
        self.global_stiffness_matrix = sparse.csr_matrix((self.n_dof_track, self.n_dof_track))

        self.rail.set_aux_stiffness_matrix()
        self.__add_rail_to_global_stiffness_matrix()

        self.rail_pads.set_aux_stiffness_matrix()
        self.__add_rail_pads_to_global_stiffness_matrix()

        self.soil.set_aux_stiffness_matrix()
        self.__add_soil_to_global_stiffness_matrix()

    def __add_rail_to_global_stiffness_matrix(self):
        """
        Set rail stiffness to global stifness matrix, dofs => normal direction, perpendicular direction, bending
        :return:
        """

        stiffness_matrix = sparse.csr_matrix((self.rail.ndof, self.rail.ndof))

        ndof_node = self.rail.ndof_per_node
        for i in range(self.rail.n_nodes - 1):
            stiffness_matrix[i * ndof_node:i * ndof_node + ndof_node * 2, i * ndof_node:i * ndof_node + ndof_node * 2] \
                += self.rail.aux_stiffness_matrix

        self.global_stiffness_matrix[0:self.rail.ndof, 0:self.rail.ndof] += stiffness_matrix

    def __add_rail_pads_to_global_stiffness_matrix(self):
        """
        Set rail pad stiffness to global stiffness matrix, dofs => perpendicular direction
        :return:
        """
        n_dof_between_rail_pads = self.rail.ndof_per_node

        for i in range(self.__n_sleepers):
            self.global_stiffness_matrix[i*n_dof_between_rail_pads + 1, i*n_dof_between_rail_pads + 1] \
                += self.rail_pads.aux_stiffness_matrix[0, 0]
            self.global_stiffness_matrix[i + self.rail.ndof, i * n_dof_between_rail_pads + 1] \
                += self.rail_pads.aux_stiffness_matrix[1, 0]
            self.global_stiffness_matrix[i*n_dof_between_rail_pads + 1, i + self.rail.ndof] \
                += self.rail_pads.aux_stiffness_matrix[0, 1]
            self.global_stiffness_matrix[i + self.rail.ndof, i + self.rail.ndof] \
                += self.rail_pads.aux_stiffness_matrix[1, 1]

    def __add_soil_to_global_stiffness_matrix(self):
        for i in range(self.__n_sleepers):
            self.global_stiffness_matrix[i + self.rail.ndof, i + self.rail.ndof] \
                += self.soil.aux_stiffness_matrix[0, 0]
            self.global_stiffness_matrix[i + self.rail.ndof + self.__n_sleepers, i + self.rail.ndof] \
                += self.soil.aux_stiffness_matrix[1, 0]
            self.global_stiffness_matrix[i + self.rail.ndof, i + self.rail.ndof + self.__n_sleepers] \
                += self.soil.aux_stiffness_matrix[0, 1]
            self.global_stiffness_matrix[i + self.rail.ndof + self.__n_sleepers, i + self.rail.ndof + self.__n_sleepers] \
                += self.soil.aux_stiffness_matrix[1, 1]

    def __add_rail_to_global_mass_matrix(self):
        mass_matrix = sparse.csr_matrix((self.rail.ndof, self.rail.ndof))

        ndof_node = self.rail.ndof_per_node
        for i in range(self.rail.n_nodes - 1):
            mass_matrix[i * ndof_node:i * ndof_node + ndof_node * 2, i * ndof_node:i * ndof_node + ndof_node * 2] \
                += self.rail.aux_mass_matrix

        self.global_mass_matrix[0:self.rail.ndof, 0:self.rail.ndof] \
            += mass_matrix

    def __add_sleeper_to_global_mass_matrix(self):
        for i in range(self.__n_sleepers):
            self.global_mass_matrix[i + self.rail.ndof, i + self.rail.ndof] += self.sleeper.mass

    def set_global_mass_matrix(self):
        """
        Set global mass matrix
        :return:
        """
        self.global_mass_matrix = sparse.csr_matrix((self.n_dof_track, self.n_dof_track))

        self.rail.set_aux_mass_matrix()

        self.__add_rail_to_global_mass_matrix()
        self.__add_sleeper_to_global_mass_matrix()

    def __add_rail_to_global_damping_matrix(self):
        damping_matrix = sparse.csr_matrix((self.rail.ndof, self.rail.ndof))

        ndof_node = self.rail.ndof_per_node
        for i in range(self.rail.n_nodes - 1):
            damping_matrix[i * ndof_node:i * ndof_node + ndof_node * 2, i * ndof_node:i * ndof_node + ndof_node * 2] \
                += self.rail.aux_damping_matrix

        self.global_damping_matrix[0:self.rail.ndof, 0:self.rail.ndof] \
            += damping_matrix

    def __add_rail_pad_to_global_damping_matrix(self):
        n_dof_between_rail_pads = self.rail.ndof_per_node

        for i in range(self.__n_sleepers):
            self.global_damping_matrix[i*n_dof_between_rail_pads + 1, i*n_dof_between_rail_pads + 1] \
                += self.rail_pads.aux_damping_matrix[0, 0]
            self.global_damping_matrix[i + self.rail.ndof, i * n_dof_between_rail_pads + 1] \
                += self.rail_pads.aux_damping_matrix[1, 0]
            self.global_damping_matrix[i*n_dof_between_rail_pads + 1, i + self.rail.ndof] \
                += self.rail_pads.aux_damping_matrix[0, 1]
            self.global_damping_matrix[i + self.rail.ndof, i + self.rail.ndof] \
                += self.rail_pads.aux_damping_matrix[1, 1]

    def __add_soil_to_global_damping_matrix(self):
        for i in range(self.__n_sleepers):
            self.global_damping_matrix[i + self.rail.ndof, i + self.rail.ndof] \
                += self.soil.aux_damping_matrix[0, 0]
            self.global_damping_matrix[i + self.rail.ndof + self.__n_sleepers, i + self.rail.ndof] \
                += self.soil.aux_damping_matrix[1, 0]
            self.global_damping_matrix[i + self.rail.ndof, i + self.rail.ndof + self.__n_sleepers] \
                += self.soil.aux_damping_matrix[0, 1]
            self.global_damping_matrix[i + self.rail.ndof + self.__n_sleepers, i + self.rail.ndof + self.__n_sleepers] \
                += self.soil.aux_damping_matrix[1, 1]

    def set_global_damping_matrix(self):
        self.global_damping_matrix = sparse.csr_matrix((self.n_dof_track, self.n_dof_track))

        self.rail.calculate_rayleigh_damping_matrix()
        self.rail_pads.set_aux_damping_matrix()
        self.soil.set_aux_damping_matrix()

        self.__add_rail_to_global_damping_matrix()
        self.__add_rail_pad_to_global_damping_matrix()
        self.__add_soil_to_global_damping_matrix()
        pass

    def set_force(self):
        self.force = sparse.csr_matrix((self.n_dof_track, len(self.time)))
        frequency=1
        self.force[4,:] = np.sin(frequency * self.time) * 15000

    def delete_from_csr(self, mat, row_indices=[], col_indices=[]):
        """
        Remove the rows (denoted by ``row_indices``) and columns (denoted by ``col_indices``) from the CSR sparse matrix ``mat``.
        WARNING: Indices of altered axes are reset in the returned matrix
        """
        if not isinstance(mat, sparse.csr_matrix):
            raise ValueError("works only for CSR format -- use .tocsr() first")

        rows = []
        cols = []
        if row_indices:
            rows = list(row_indices)
        if col_indices:
            cols = list(col_indices)

        if len(rows) > 0 and len(cols) > 0:
            row_mask = np.ones(mat.shape[0], dtype=bool)
            row_mask[rows] = False
            col_mask = np.ones(mat.shape[1], dtype=bool)
            col_mask[cols] = False
            return mat[row_mask][:, col_mask]
        elif len(rows) > 0:
            mask = np.ones(mat.shape[0], dtype=bool)
            mask[rows] = False
            return mat[mask]
        elif len(cols) > 0:
            mask = np.ones(mat.shape[1], dtype=bool)
            mask[cols] = False
            return mat[:, mask]
        else:
            return mat

    def apply_boundary_condition(self):
        bottom_node_idxs = np.arange(self.rail.ndof + self.__n_sleepers,self.rail.ndof + 2*self.__n_sleepers).tolist()


        self.force = self.delete_from_csr(self.force, row_indices=bottom_node_idxs)
        self.global_mass_matrix = self.delete_from_csr(self.global_mass_matrix, row_indices=bottom_node_idxs, col_indices=bottom_node_idxs)
        self.global_stiffness_matrix = self.delete_from_csr(self.global_stiffness_matrix, row_indices=bottom_node_idxs, col_indices=bottom_node_idxs)
        self.global_damping_matrix = self.delete_from_csr(self.global_damping_matrix, row_indices=bottom_node_idxs, col_indices=bottom_node_idxs)

        # self.force = np.delete(self.force, idx, axis=0)
        # self.global_mass_matrix = np.delete(np.delete(self.global_mass_matrix, idx, axis=0),
        #                                          idx, axis=1)
        # self.global_stiffness_matrix = np.delete(np.delete(self.global_stiffness_matrix, idx, axis=0),
        #                                          idx, axis=1)
        # self.global_damping_matrix = np.delete(np.delete(self.global_damping_matrix, idx, axis=0),
        #                                          idx, axis=1)



    def calculate_n_dofs(self):
        """
        todo calculate ndof soil, add train, add ballast
        :return:
        """
        self.rail.calculate_n_dof()

        ndof_rail_pads = self.__n_sleepers
        ndof_soil = self.__n_sleepers

        self.n_dof_track = self.rail.ndof + ndof_rail_pads + ndof_soil

    def calculate_length_track(self):
        self.__total_length = (self.__n_sleepers - 1) * self.sleeper.distance_between_sleepers


if __name__ == "__main__":
    pass
    # # do stuff
    #
    # sleeper = Sleeper()
    # sleeper.ms = 0.1625
    # sleeper.d = 0.6
    # sleeper.damping_ratio = 0.04
    # sleeper.radial_frequency_one = 2
    # sleeper.radial_frequency_two = 500
    #
    # support = Support()
    # support.linear_stiffness = 999
    # support.n_sleepers = 100
    # t = time.time()
    # for i in range(1000):
    #     support.linear_stiffness_matrix
    # elapsed1 = time.time() - t
    # print(elapsed1)
    #
    # support.calculate_matrix()
    # t = time.time()
    # for i in range(1000):
    #     a = support.linear_stiffness_matrix2
    # elapsed1 = time.time() - t
    # print(elapsed1)

    # track = Track()
    #
    # track.damping_ratio = 3
    # track.radial_frequency_one = 2
    # track.radial_frequency_two = 4
    #
    #
    # t = time.time()
    # for i in range(10000):
    #     track.calculate_damping_factors()
    # elapsed1 = time.time() - t
    # print(elapsed1)
