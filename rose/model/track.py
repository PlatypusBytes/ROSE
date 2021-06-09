import numpy as np
from rose.model.model_part import ElementModelPart, RodElementModelPart, TimoshenkoBeamElementModelPart
import logging

class InvalidRailException(Exception):
    def __init__(self, message):
        logging.error(message)
        super().__init__(message)


class Rail(TimoshenkoBeamElementModelPart):
    """
    Rail element model part class. This class bases from
    :class:`~rose.model.model_part.TimoshenkoBeamElementModelPart`.

    :Attributes:

    """

    def __init__(self):
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
        return True

    def set_aux_stiffness_matrix(self):
        self.aux_stiffness_matrix = np.zeros((1, 1))

    def set_aux_damping_matrix(self):
        self.aux_damping_matrix = np.zeros((1, 1))

    def set_aux_mass_matrix(self):
        self.aux_mass_matrix = np.ones((1, 1)) * self.mass


class RailPad(RodElementModelPart):
    """
    Rail pad element model part class. This class bases from
    :class:`~rose.model.model_part.RodElementModelPart`.

    :Attributes:

    """
    def __init__(self):
        super().__init__()
