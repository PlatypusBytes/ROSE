from src.mesh_utils import *
import pytest

from src.track import *
from src.global_system import *
from src.train_model.train_model import TrainModel

import matplotlib.pyplot as plt


class TestTrack:
    def test_infinite_euler_beam_without_damping(self):
        """
        Test a point load on ann infinitely long euler beam on a winkler foundation without damping.
        Test succeeds if the numerically calculated max displacement is approximately the analytically calculated max
        displacement. ref: https://www.mae.ust.hk/~meqpsun/Notes/CHAPTER4.pdf
        :return:
        """

        # calculate analytical solution
        stiffness_spring = 2.75e5
        distance_springs = 1
        winkler_mod = stiffness_spring / distance_springs

        youngs_mod_beam = 4.41e05
        intertia_beam = 1
        y_load = -18e3
        winkler_beta = (winkler_mod / (4 * youngs_mod_beam * intertia_beam)) ** 0.25

        x = 0
        winkler_const1 = np.exp(-winkler_beta * x) * (
            np.cos(winkler_beta * x) + np.sin(winkler_beta * x)
        )
        expected_max_displacement = (
            winkler_beta * y_load / (2 * winkler_mod) * winkler_const1
        )

        # setup numerical model
        # set time in two stages
        initialisation_time = np.linspace(0, 0.1, 100)
        calculation_time = np.linspace(initialisation_time[-1], 10, 5000)
        time = np.concatenate((initialisation_time, calculation_time[1:]))

        # set geometry
        depth_soil = 0.9
        element_model_parts, mesh = create_horizontal_track(
            100, distance_springs, depth_soil
        )
        bottom_boundary = add_no_displacement_boundary_to_bottom(
            element_model_parts["soil"]
        )

        # fill model parts
        rail_model_part = element_model_parts["rail"]
        rail_pad_model_part = element_model_parts["rail_pad"]
        sleeper_model_part = element_model_parts["sleeper"]
        soil = element_model_parts["soil"]

        # set elements
        material = Material()
        material.youngs_modulus = youngs_mod_beam  # Pa
        material.poisson_ratio = 0.0
        material.density = 0.000001  # 7860

        section = Section()
        section.area = 1
        section.sec_moment_of_inertia = 1
        section.shear_factor = 0

        rail_model_part.section = section
        rail_model_part.material = material
        rail_model_part.damping_ratio = 0.0000
        rail_model_part.radial_frequency_one = 2
        rail_model_part.radial_frequency_two = 500

        rail_model_part.initialize()

        rail_pad_model_part.mass = 0.000001  # 5
        rail_pad_model_part.stiffness = stiffness_spring / 0.1
        rail_pad_model_part.damping = 0  # 12e3

        sleeper_model_part.mass = 0.0000001  # 162.5
        sleeper_model_part.distance_between_sleepers = distance_springs

        soil.stiffness = stiffness_spring / depth_soil  # 300e6
        soil.damping = 0

        # set load
        velocities = np.ones(len(time)) * 10
        load = add_moving_point_load_to_track(
            rail_model_part,
            time,
            len(initialisation_time),
            velocities,
            y_load=y_load,
            start_coords=np.array([5, 0, 0]),
        )

        # set solver
        solver = NewmarkSolver()

        # populate global system
        global_system = GlobalSystem()
        global_system.mesh = mesh
        global_system.time = time
        global_system.solver = solver

        # get all element model parts from dictionary
        model_parts = [
            list(element_model_parts.values()),
            list(bottom_boundary.values()),
            list(load.values()),
        ]
        global_system.model_parts = list(itertools.chain.from_iterable(model_parts))

        # calculate
        global_system.main()

        # get max displacement in middle node of the beam
        max_disp = min(global_system.displacements[:, 151])

        # plt.plot(global_system.displacements[:, 151])
        # plt.show()

        # assert max displacement
        assert max_disp == pytest.approx(expected_max_displacement, rel=1e-2)

    def test_train_on_infinite_euler_beam_without_damping(self):
        """
        Test a point load on ann infinitely long euler beam on a winkler foundation without damping.
        Test succeeds if the numerically calculated max displacement is approximately the analytically calculated max
        displacement. ref: https://www.mae.ust.hk/~meqpsun/Notes/CHAPTER4.pdf
        :return:
        """

        # calculate analytical solution
        stiffness_spring = 2.75e5
        distance_springs = 0.25
        winkler_mod = stiffness_spring / distance_springs

        youngs_mod_beam = 4.41e05
        interntia_beam = 1
        y_load = -18000
        winkler_beta = (winkler_mod / (4 * youngs_mod_beam * interntia_beam)) ** 0.25

        x = 0
        winkler_const1 = np.exp(-winkler_beta * x) * (
            np.cos(winkler_beta * x) + np.sin(winkler_beta * x)
        )
        expected_max_displacement = (
            winkler_beta * y_load / (2 * winkler_mod) * winkler_const1
        )

        # setup numerical model
        # set time in two stages
        initialisation_time = np.linspace(0, 0.1, 100)
        calculation_time = np.linspace(initialisation_time[-1], 10, 5000)
        time = np.concatenate((initialisation_time, calculation_time[1:]))

        # set geometry
        depth_soil = 0.9
        element_model_parts, mesh = create_horizontal_track(
            100, distance_springs, depth_soil
        )
        bottom_boundary = add_no_displacement_boundary_to_bottom(
            element_model_parts["soil"]
        )

        # create train
        train_model = TrainModel()
        train_model.length_cart = 3
        train_model.length_bogie = 1
        train_model.create_cart()

        # load = add_moving_point_load_to_track(element_model_parts['rail'], time, len(initialisation_time), y_load=y_load)

        # fill model parts
        rail_model_part = element_model_parts["rail"]
        rail_pad_model_part = element_model_parts["rail_pad"]
        sleeper_model_part = element_model_parts["sleeper"]
        soil = element_model_parts["soil"]

        # set train
        train_model.mass_cart = 77000
        train_model.mass_bogie = 1100
        train_model.mass_wheel = 1200
        train_model.inertia_cart = 1.2e6
        train_model.inertia_bogie = 760
        train_model.prim_stiffness = 2.14e6
        train_model.sec_stiffness = 5.32e6
        train_model.prim_damping = 4.9e4
        train_model.sec_damping = 7e4
        train_model.herzian_contact_cof = 1

        # set elements
        material = Material()
        material.youngs_modulus = youngs_mod_beam  # Pa
        material.poisson_ratio = 0.0
        material.density = 0.000001  # 7860

        section = Section()
        section.area = 1
        section.sec_moment_of_inertia = 1
        section.shear_factor = 0

        rail_model_part.section = section
        rail_model_part.material = material
        rail_model_part.damping_ratio = 0.0000
        rail_model_part.radial_frequency_one = 2
        rail_model_part.radial_frequency_two = 500

        rail_pad_model_part.mass = 0.000001  # 5
        rail_pad_model_part.stiffness = stiffness_spring / 0.1
        rail_pad_model_part.damping = 0  # 12e3

        sleeper_model_part.mass = 0.0000001  # 162.5
        sleeper_model_part.distance_between_sleepers = distance_springs

        soil.stiffness = stiffness_spring / depth_soil  # 300e6
        soil.damping = 0

        # set solver
        solver = NewmarkSolver()

        # populate global system
        global_system = GlobalSystem()
        global_system.mesh = mesh
        global_system.time = time
        global_system.solver = solver

        global_system.mesh.nodes = np.concatenate(
            (global_system.mesh.nodes, train_model.nodes)
        )
        global_system.mesh.elements = np.concatenate(
            (global_system.mesh.elements, train_model.elements)
        )

        global_system.mesh.reorder_node_ids()
        global_system.mesh.reorder_element_ids()

        # get all element model parts from dictionary
        model_parts = [
            list(element_model_parts.values()),
            list(bottom_boundary.values()),
            [train_model],
        ]  # list(load.values())]
        global_system.model_parts = list(itertools.chain.from_iterable(model_parts))

        # calculate
        global_system.main()

        # get max displacement in middle node of the beam
        max_disp = min(global_system.displacements[:, 151])

        # assert max displacement
        assert max_disp == pytest.approx(expected_max_displacement, rel=1e-3)
