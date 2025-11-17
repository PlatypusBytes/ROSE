import numpy as np
from rose.model.irregularities import RailDefect


class TestRailDefect:

    def test_create_triangle_rail_defect(self):
        """
        Tests the creation of a rail defect. A triangle defect is create 2 meter after the start of the rail
        and extends over 8 meters with a maximum height of 0.005 meters.
        """
        x = np.linspace(0, 10, 11)
        local_defect_geometry_coordinates = [[0, 0], [5, 0.005], [10, 0]]
        start_position = 2

        rail_defect = RailDefect(x, local_defect_geometry_coordinates, start_position)

        expected_irregularities = np.zeros_like(x)
        expected_irregularities[2:8] = np.linspace(0, 0.005, 6)
        expected_irregularities[7:11] = np.linspace(0.005, 0, 4)

        assert np.allclose(rail_defect.irregularities, expected_irregularities)

    def test_create_rectangle_rail_defect(self):
        """
        Tests the creation of a rail defect. A rectangle defect is create 3 meter after the start of the rail
        and extends over 4 meters with a maximum height of 0.01 meters.
        """
        x = np.linspace(0, 10, 11)
        local_defect_geometry_coordinates = [[0, 0], [0, 0.01], [4, 0.01], [4, 0]]
        start_position = 3

        rail_defect = RailDefect(x, local_defect_geometry_coordinates, start_position)

        expected_irregularities = np.zeros_like(x)
        expected_irregularities[3:8] = 0.01

        assert np.allclose(rail_defect.irregularities, expected_irregularities)
