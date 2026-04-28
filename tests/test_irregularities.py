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
        wheel_diameter = 1

        # create rail defect
        rail_defect = RailDefect(x, wheel_diameter, local_defect_geometry_coordinates, start_position)

        # calculate expected irregularities
        expected_irregularities = np.zeros_like(x)
        expected_irregularities[2:8] = -np.linspace(0, 0.005, 6)
        expected_irregularities[7:11] = -np.linspace(0.005, 0.002, 4)


        # add extra displacement due to position of the wheel on the slope
        wheel_radius = wheel_diameter / 2
        extra_disp_due_to_position = -(wheel_radius / np.cos(np.arctan(0.005 / 5)) - wheel_radius)
        slope_mask = np.ones(len(x), dtype=bool)
        no_slope_indices = [0, 1, 7]
        slope_mask[no_slope_indices] = False
        expected_irregularities[slope_mask] += extra_disp_due_to_position

        assert np.allclose(rail_defect.irregularities, expected_irregularities)

    def test_create_rectangle_rail_defect(self):
        """
        Tests the creation of a rail defect. A rectangle defect is create 3 meter after the start of the rail
        and extends over 4 meters with a maximum height of 0.01 meters.
        """
        x = np.linspace(0, 10, 11)
        local_defect_geometry_coordinates = [[0, 0], [0, 0.01], [4, 0.01], [4, 0]]
        start_position = 3
        wheel_diameter = 1

        rail_defect = RailDefect(x, wheel_diameter, local_defect_geometry_coordinates, start_position)

        expected_irregularities = np.zeros_like(x)
        expected_irregularities[3:8] = -0.01

        assert np.allclose(rail_defect.irregularities, expected_irregularities)


    def test_downward_triangle_defect(self):
        """
        In this test a downward triangle defect is created. In this test, the wheel trajectory should not reach the
        bottom of the defect, as the wheel diameter is large enough to bridge part of the defect.
        """

        x = np.linspace(0, 10, 21)
        local_defect_geometry_coordinates = [[0, 0], [5, -5], [10, 0]]
        start_position = 0
        wheel_diameter = 1

        rail_defect = RailDefect(x, wheel_diameter, local_defect_geometry_coordinates, start_position)

        expected_irregularities = np.zeros_like(x)
        expected_irregularities[0:11] = -np.linspace(0, -5, 11)
        expected_irregularities[10:21] = -np.linspace(-5, 0, 11)

        # add extra displacement due to position of the wheel on the slope
        wheel_radius = wheel_diameter / 2
        extra_disp_due_to_position = -(wheel_radius / np.cos(np.arctan(-5/5)) - wheel_radius)
        no_slope_indices = [0, -1]
        slope_mask = np.ones(len(x), dtype=bool)
        slope_mask[no_slope_indices] = False
        expected_irregularities[slope_mask] += extra_disp_due_to_position

        assert np.allclose(rail_defect.irregularities, expected_irregularities)
