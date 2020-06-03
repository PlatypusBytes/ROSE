import numpy as np
from scipy import sparse
import time


class Material():
    def __init__(self):
        self.youngs_modulus = None
        self.poisson_ratio = None
        self.density= None

    @property
    def shear_modulus(self):
        return self.youngs_modulus/(2*(1+self.poisson_ratio))

class Section():
    def __init__(self):
        self.area = None # [m^2]
        self.sec_moment_of_inertia = None # [m^4]
        self.shear_factor = 0 #shear factor (kr=0 - Euler-Bernoulli beam, kr>0 - Timoshenko beam)
        self.n_rail_per_sleeper = None

class Rail():
    def __init__(self, n_sleepers):
        self.material = Material()
        self.section = Section()
        self.__n_sleepers = n_sleepers
        self.level = np.zeros(1, n_sleepers)

        self.length_rail = None
        self.mass = None

        self.timoshenko_factor = 0 # ???
        self.ndof = None

        self.mass_matrix = None



    def set_top_level_to_zero(self):
        self.level = self.level - np.max(self.level)

    def calculate_length_rail(self,distance_between_sleepers):
        self.length_rail = distance_between_sleepers/self.section.n_rail_per_sleeper

    def calculate_mass(self):
        self.mass = self.section.area * self.material.density

    def calculate_timoshenko_factor(self):
        if self.section.shear_factor > 0:
            self.timoshenko_factor = 12 * self.material.youngs_modulus * self.section.sec_moment_of_inertia / (
                                     self.length_rail**2 * self.material.shear_modulus *
                                     self.section.area * self.section.shear_factor)

    def calculate_n_dof(self):
        self.ndof = self.section.n_rail_per_sleeper * (self.__n_sleepers - 1) + 1

    def calculate_mass_matrix(self):
        self.mass_matrix = np.array([self.mass * self.length_rail/2,
                       self.mass*self.length_rail * np.ones(1, self.ndof-2),
                       self.mass*self.length_rail/2])


class Sleeper:
    def __init__(self):
        self.ms=None
        self.distance_between_sleepers =None
        self.damping_ratio = None
        self.radial_frequency_one = None
        self.radial_frequency_two = None

class RailPad:
    def __init__(self):
        self.mass = None
        self.stiffness = None
        self.damping = None

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

        self.mass_matrix_track = None


        self.__total_length = None
        self.__n_dof_rail = None
        self.__n_dof_track = None


    def calculate_mass_matrix(self):
        self.rail.calculate_mass_matrix()
        rail_mass = self.rail.mass_matrix

        self.mass_matrix_track = np.zeros(self.__n_dof_track, self.__n_dof_track)

        self.mass_matrix_track = sparse.csr_matrix([rail_mass, 0],([range(self.rail.ndof), self.__n_dof_track - 1],
                                                                    [range(self.rail.ndof), self.__n_dof_track - 1]))



        # for i in range(self.rail.ndof):
        #     self.mass_matrix_track[i,i]
        #
        #
        # self.mass_matrix_track = sparse.csr_matrix([rail_mass, 0],())



    def calculate_n_dofs(self):
        self.rail.calculate_n_dof()
        self.n_dof_track = self.rail.ndof + self.__n_sleepers

    def calculate_length_track(self):
        self.__total_length = (self.__n_sleepers - 1) * self.sleeper.distance_between_sleepers




        # self.mass_matrix = None
        # self.damping_matrix = None
        # self.stiffness_matrix = None
        # self.support_stiffness_matrix = None
        # self.damping_ratio = None
        # self.radial_frequency_one = None
        # self.radial_frequency_two = None


    def calculate_damping_factors(self):
        """
        Calculate rayleigh damping coefficients
        :return:
        """
        contant = 2*self.damping_ratio/(self.radial_frequency_one + self.radial_frequency_two)
        damping_factors = self.radial_frequency_one*self.radial_frequency_two * contant, contant
        return damping_factors

    def calculate_damping_matrix(self, damping_factors):
        """
        Calculate rayghleigh damping matrix
        :param damping_factors:
        :return:
        """
        self.damping_matrix = damping_factors[0].dot(self.mass_matrix) + \
                              damping_factors[1].dot(self.stiffness_matrix + self.support_stiffness_matrix)


    # def calculate_damping_matrix(self):

if __name__=="__main__":

    # do stuff


    sleeper = Sleeper()
    sleeper.ms = 0.1625
    sleeper.d=0.6
    sleeper.damping_ratio = 0.04
    sleeper.radial_frequency_one=2
    sleeper.radial_frequency_two=500



    support = Support()
    support.linear_stiffness = 999
    support.n_sleepers = 100
    t = time.time()
    for i in range(1000):
        support.linear_stiffness_matrix
    elapsed1 = time.time() - t
    print(elapsed1)

    support.calculate_matrix()
    t = time.time()
    for i in range(1000):
        a = support.linear_stiffness_matrix2
    elapsed1 = time.time() - t
    print(elapsed1)

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




