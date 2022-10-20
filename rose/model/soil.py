"""
This module contains all soil element model parts

"""

from rose.model.model_part import RodElementModelPart


class Soil(RodElementModelPart):
    """
    Soil element model part class. This class bases from
    :class:`~rose.model.model_part.RodElementModelPart`. A soil element only interacts in normal direction.

    :Attributes:

    """
    def __init__(self):
        """
        Initialises Soil model part and its base
        """
        super(Soil, self).__init__()

