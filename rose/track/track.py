import numpy as np
from rose.model.model_part import ElementModelPart, RodElementModelPart, TimoshenkoBeamElementModelPart
import logging

class InvalidRailException(Exception):
    def __init__(self, message):
        logging.error(message)
        super().__init__(message)


class Rail(TimoshenkoBeamElementModelPart):
    def __init__(self):
        super().__init__()

    def calculate_length_rail(self):

        xdiff = np.diff([node.coordinates[0] for node in self.nodes])
        ydiff = np.diff([node.coordinates[1] for node in self.nodes])
        zdiff = np.diff([node.coordinates[2] for node in self.nodes])

        distances = np.sqrt(np.square(xdiff) + np.square(ydiff) + np.square(zdiff))

        if distances.size > 0:
            if not np.all(np.isclose(distances[0], distances)):
                raise InvalidRailException("distance between sleepers is not equal")

            self.length_element = distances[0]

    def initialize(self):
        self.validate_input()
        self.calculate_length_rail()
        super(Rail, self).initialize()


class Sleeper(ElementModelPart):
    def __init__(self):
        super().__init__()
        self.mass = None
        self.height_sleeper = 0.1

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
    def __init__(self):
        super().__init__()
        self.stiffness = None
        self.damping = None
        self.aux_stiffness_matrix = None
        self.aux_damping_matrix = None

if __name__ == "__main__":
    pass