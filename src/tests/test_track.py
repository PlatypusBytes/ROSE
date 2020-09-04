import pytest

from src.geometry import Element, Node

from src.track import *
from src.global_system import *
import src.model_part as model_part


class TestTrack:
    def test_initialise_rail(self):
        rail = Rail()
        assert isinstance(rail, TimoshenkoBeamElementModelPart)

        with pytest.raises(model_part.ParameterNotDefinedException):
            rail.initialize()

    def test_calculate_length_rail(self):

        rail = Rail()

        nodes = [Node(0, 0, 0), Node(3, 4, 0), Node(3, 7, 4)]
        elements = [Element([nodes[0], nodes[1]]),  Element([nodes[1], nodes[2]])]
        rail.nodes = nodes
        rail.elements = elements

        rail.calculate_length_rail()

        assert 5 == pytest.approx(rail.length_rail)

    def test_calculate_length_rail_expected_raise(self):
        rail = Rail()

        nodes = [Node(0, 0, 0), Node(3, 4, 0), Node(3, 7, 5)]
        elements = [Element([nodes[0], nodes[1]]), Element([nodes[1], nodes[2]])]
        rail.nodes = nodes
        rail.elements = elements

        with pytest.raises(InvalidRailException):
            rail.calculate_length_rail()


