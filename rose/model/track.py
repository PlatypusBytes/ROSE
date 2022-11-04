"""
This module contains all model parts which are part of the track. I.e. the rail, sleeper and rail pads

"""

import numpy as np
from rose.model.model_part import ElementModelPart, RodElementModelPart, TimoshenkoBeamElementModelPart
import logging


class InvalidRailException(Exception):
    """
    Invalid Rail Exception class. This class bases from
    :class:`Exception`. This Exception class indicates that the rail is invalid

    :Attributes:

    """
    def __init__(self, message):
        """
        Initialises InvalidRailException and logs an error message
        :param message: message to be logged
        """
        logging.error(message)
        super().__init__(message)


class Rail(TimoshenkoBeamElementModelPart):
    """
    Rail element model part class. This class bases from
    :class:`~rose.model.model_part.TimoshenkoBeamElementModelPart`.

    :Attributes:

    """

    def __init__(self):
        """
        Initialises the rail model part and its base
        """
        super().__init__()

    def calculate_length_rail(self):
        """
        Calculates the length of the rail elements. It is required that all element within the model part have an equal
        size.
        :return:
        """

        # calculate coordinate differences between each node
        xdiff = np.diff([node.coordinates[0] for node in self.nodes])
        ydiff = np.diff([node.coordinates[1] for node in self.nodes])
        zdiff = np.diff([node.coordinates[2] for node in self.nodes])

        # calculate total distances for each element
        distances = np.sqrt(np.square(xdiff) + np.square(ydiff) + np.square(zdiff))

        # check if all elements are the same size
        if distances.size > 0:
            if not np.all(np.isclose(distances[0], distances)):
                raise InvalidRailException("distance between sleepers is not equal")

            # assign length element
            self.length_element = distances[0]

    def initialize(self):
        """
        Initialises rail model part. Input is validated and the length of the rail is calculated
        :return:
        """
        self.validate_input()
        self.calculate_length_rail()
        super(Rail, self).initialize()


class Sleeper(ElementModelPart):
    """
    Sleeper element model part class. This class bases from
    :class:`~rose.model.model_part.ElementModelPart`. This model part is a point which contains a mass.

    :Attributes:

        - :self.mass:           mass of the sleeper
    """
    def __init__(self):
        super().__init__()
        self.mass = None

    @property
    def y_disp_dof(self):
        """
        Displacement in y direction degree of freedom is set on true for the sleeper model part

        :return:
        """
        return True

    def set_aux_stiffness_matrix(self):
        """
        Sets auxiliary stiffness matrix of sleeper as zero

        :return:
        """
        self.aux_stiffness_matrix = np.zeros((1, 1))

    def set_aux_damping_matrix(self):
        """
       Sets auxiliary damping matrix of sleeper as zero

       :return:
       """
        self.aux_damping_matrix = np.zeros((1, 1))

    def set_aux_mass_matrix(self):
        """
        Sets auxiliary mass matrix of sleeper as sleeper mass

        :return:
       """
        self.aux_mass_matrix = np.array([[self.mass]])


class Hinge(ElementModelPart):
    """
    Hinge element model part class. This model part can only be used together with a beam model part.
    This class bases from :class:`~rose.model.model_part.ElementModelPart`. This model part is a point which contains
    a mass and fixity factor which indicates the degree of fixation on the connected beam element.

    :Attributes:

        - :self.mass:           mass of the hinge
        - :self.fixity_factor:  Factor which indicates the degree of fixation of the hinge, ranging from 0-1 (free-fixed) [-]
        - :self.connected_beam: Beam model part which is connected to the hinge
    """
    def __init__(self):
        super().__init__()
        self.mass = None
        self.fixity_factor = None
        self.connected_beam = None

        self.rotational_stiffness = None

    @property
    def y_disp_dof(self):
        """
        Displacement in y direction degree of freedom is set on true for the hinge model part

        :return:
        """
        return True

    def set_aux_stiffness_matrix(self):
        """
        Sets auxiliary stiffness matrix of hinge as zero

        :return:
        """
        self.aux_stiffness_matrix = np.zeros((1, 1))

    def set_aux_damping_matrix(self):
        """
       Sets auxiliary damping matrix of sleeper as zero

       :return:
       """
        self.aux_damping_matrix = np.zeros((1, 1))

    def set_aux_mass_matrix(self):
        """
        Sets auxiliary mass matrix of sleeper as sleeper mass

        :return:
       """
        self.aux_mass_matrix = np.array([[self.mass]])

    def calculate_hinge_stiffness(self):
        """
        Calculates rotational stiffness [Nm] based on fixity factor of hinge, Youngs modulus, second moment of area and
        length of the connected beam element.

        :return:
        """

        E = self.connected_beam.material.youngs_modulus  # Youngs modulus of connected beam [N/m2]
        I = self.connected_beam.section.sec_moment_of_inertia
        l = self.connected_beam.length_element

        if 0 <= self.fixity_factor < 1:
            self.rotational_stiffness = 3 * E * I * self.fixity_factor / ((1 - self.fixity_factor) * l)
        elif np.isclose(self.fixity_factor, 1):
            # >> infinite
            self.rotational_stiffness = 1e30
        else:
            Exception("fixity factor should be between 0 and 1")


class RailPad(RodElementModelPart):
    """
    Rail pad element model part class. This class bases from
    :class:`~rose.model.model_part.RodElementModelPart`.

    :Attributes:

    """
    def __init__(self):
        """
        Initialises rail pad model part and its base
        """
        super().__init__()
