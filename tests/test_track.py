import pytest

from rose.model.geometry import Element, Node

from rose.model.track import *
from rose.model.global_system import *


class TestTrack:
    def test_validate_empty_rail(self):
        """
        Validate an empty rail, assert if logger contains error messages
        :return:
        """
        rail = Rail()
        assert isinstance(rail, TimoshenkoBeamElementModelPart)

        rail.validate_input()
        assert logging.getLogger()._cache.__contains__(40)


    def test_calculate_length_rail(self):

        rail = Rail()

        nodes = [Node(0, 0, 0), Node(3, 4, 0), Node(3, 7, 4)]
        elements = [Element([nodes[0], nodes[1]]),  Element([nodes[1], nodes[2]])]
        rail.nodes = nodes
        rail.elements = elements

        rail.calculate_length_rail()

        assert 5 == pytest.approx(rail.length_element)

    def test_calculate_length_rail_expected_raise(self):
        rail = Rail()

        nodes = [Node(0, 0, 0), Node(3, 4, 0), Node(3, 7, 5)]
        elements = [Element([nodes[0], nodes[1]]), Element([nodes[1], nodes[2]])]
        rail.nodes = nodes
        rail.elements = elements

        with pytest.raises(InvalidRailException):
            rail.calculate_length_rail()


