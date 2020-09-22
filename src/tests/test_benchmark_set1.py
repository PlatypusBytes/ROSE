import json
import pytest

from src.global_system import *
from src.train_model.train_model import TrainModel
from src.model_part import Material, Section, TimoshenkoBeamElementModelPart, RodElementModelPart
from src.mesh_utils import *

import src.tests.utils.signal_proc as sp

from analytical_solutions.simple_supported import \
    SimpleSupportEulerNoDamping, \
    SimpleSupportEulerStatic, \
    SimpleSupportEulerWithDamping, \
    SimpleSupportTimoshenkoNoDamping
from analytical_solutions.cantilever_beam import PulseLoadNoDamping

import matplotlib.pyplot as plt


class TestBenchmarkSet1:
    """
    In benchmark set 1, benchmarks are compared with the analytical solution
    """
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
        vertical_displacements_rail = np.array([node.displacements[:,1] for node in global_system.model_parts[0].nodes])
        max_disp = min(vertical_displacements_rail[50, :])

        # assert max displacement
        assert max_disp == pytest.approx(expected_max_displacement, rel=1e-2)

    def test_euler_beam_on_hinch_foundation_dynamic(self):
        """
        Tests an euler beam on a hinch foundation. In this benchmark a point load is added in the middle of the beam
        and the dynamic response is compared to the analytical solution.
        :return:
        """
        length_beam = 0.01
        n_beams = 1001

        # Setup parameters euler beam
        time = np.linspace(0, 10, 5001)
        E = 20e7
        I = 1
        rho = 2000
        A = 1
        L = (n_beams-1)*length_beam
        F = -1000

        # set and calculated analytical solution
        beam_analytical = SimpleSupportEulerNoDamping(ele_size=length_beam)
        beam_analytical.properties(E, I, rho, A, L, F, time)
        beam_analytical.compute()

        # setup numerical model
        beam_nodes = [Node(i*length_beam,0,0) for i in range(n_beams)]
        beam_elements = [Element([beam_nodes[i], beam_nodes[i+1]]) for i in range(n_beams - 1)]

        mesh = Mesh()
        mesh.add_unique_nodes_to_mesh(beam_nodes)
        mesh.add_unique_elements_to_mesh(beam_elements)

        material = Material()
        material.youngs_modulus = E  # Pa
        material.poisson_ratio = 0.0
        material.density = rho

        section = Section()
        section.area = A
        section.sec_moment_of_inertia = I
        section.shear_factor = 0

        beam = TimoshenkoBeamElementModelPart()
        beam.nodes = beam_nodes
        beam.elements = beam_elements

        beam.material = material
        beam.section = section
        beam.length_element = length_beam
        beam.damping_ratio = 0.00000
        beam.radial_frequency_one = 2
        beam.radial_frequency_two = 500
        beam.initialize()

        # set up hinge foundation
        foundation1 = ConstraintModelPart(normal_dof=False, y_disp_dof=False, z_rot_dof=True)
        foundation1.nodes = [beam_nodes[0]]

        foundation2 = ConstraintModelPart(normal_dof=True, y_disp_dof=False, z_rot_dof=True)
        foundation2.nodes = [beam_nodes[-1]]

        # set load on middle node
        load = LoadCondition(normal_dof=False, y_disp_dof=True, z_rot_dof=False)
        load.y_force = np.ones((1,len(time))) * F
        load.time = time
        load.nodes = [beam_nodes[int((n_beams-1)/2)]]

        model_parts = [beam, foundation1, foundation2, load]

        # set solver
        solver = NewmarkSolver()

        # populate global system
        global_system = GlobalSystem()
        global_system.mesh = mesh
        global_system.time = time
        global_system.solver = solver

        global_system.model_parts = model_parts

        # calculate
        global_system.main()

        # get vertical displacements of beam
        vertical_displacements = np.array([node.displacements[:,1] for node in beam_nodes])

        # process signal of numerical and analytical solution
        _, amplitude_num,_,_ = sp.fft_sig(vertical_displacements[int((n_beams-1)/2), :], int(1/ time[1]),
                                          nb_points=2**14)
        _, amplitude_analyt, _, _ = sp.fft_sig(beam_analytical.u[int((n_beams-1)/2), :], int(1 / time[1]),
                                               nb_points=2**14)

        # assert if signal aplitudes are approximately equal at eigen frequency
        assert amplitude_num[163] == pytest.approx(amplitude_analyt[163], rel=1e-2)

    def test_damped_euler_beam_on_hinch_foundation_dynamic(self):
        """
        Tests point on an euler beam which is supported by a hinch on each side of the beam. The calculation is
        dynamic with damping.
        Vertical deflection on the middle node is compared with the analytical solution.
        :return:
        """

        length_beam = 0.05
        n_beams = 201

        # Setup parameters euler beam
        time = np.linspace(0, 100, 10001)
        E = 20e5
        I = 1
        rho = 2000
        A = 1
        L = 10
        F = -1000

        damping_ratio = 0.02#1e3
        omega1 = 2
        omega2 = 5000

        constant = (
            2
            * damping_ratio
            / (omega1 + omega2)
        )
        a0 = omega1 * omega2 * constant
        a1 = constant

        coefs = [a0, a1]

        x_F = 5

        # set and calculated analytical solution
        # beam_analytical = SimpleSupportEulerStatic()
        # beam_analytical.properties(E, I, L, F, x_F)
        # beam_analytical.compute()

        beam_analytical = SimpleSupportEulerWithDamping(n=150, ele_size=length_beam)
        beam_analytical.properties(E, I, rho, A, L, F, coefs, time)
        beam_analytical.compute()

        # setup numerical model
        beam_nodes = [Node(i * length_beam, 0, 0) for i in range(n_beams)]
        beam_elements = [Element([beam_nodes[i], beam_nodes[i + 1]]) for i in range(n_beams - 1)]

        mesh = Mesh()
        mesh.add_unique_nodes_to_mesh(beam_nodes)
        mesh.add_unique_elements_to_mesh(beam_elements)

        material = Material()
        material.youngs_modulus = E  # Pa
        material.poisson_ratio = 0.0
        material.density = rho

        section = Section()
        section.area = A
        section.sec_moment_of_inertia = I
        section.shear_factor = 0

        beam = TimoshenkoBeamElementModelPart()
        beam.nodes = beam_nodes
        beam.elements = beam_elements

        beam.material = material
        beam.section = section
        beam.length_element = length_beam
        beam.damping_ratio = damping_ratio
        beam.radial_frequency_one = omega1
        beam.radial_frequency_two = omega2

        beam.initialize()

        foundation1 = ConstraintModelPart(normal_dof=False, y_disp_dof=False, z_rot_dof=True)
        foundation1.nodes = [beam_nodes[0]]

        foundation2 = ConstraintModelPart(normal_dof=True, y_disp_dof=False, z_rot_dof=True)
        foundation2.nodes = [beam_nodes[-1]]

        load = LoadCondition(normal_dof=False, y_disp_dof=True, z_rot_dof=False)
        load.y_force = np.ones((1, len(time))) * F

        load.time = time
        load.nodes = [beam_nodes[int((n_beams-1)/2)]]

        model_parts = [beam, foundation1, foundation2, load]

        # set solver
        solver = NewmarkSolver()

        # populate global system
        global_system = GlobalSystem()
        global_system.mesh = mesh
        global_system.time = time
        global_system.solver = solver

        global_system.model_parts = model_parts

        # calculate
        global_system.main()
        vertical_displacements = np.array([node.displacements[:, 1] for node in beam_nodes])


        # plt.plot(time, vertical_displacements[100, :])
        # plt.plot(time, beam_analytical.u[100, :])
        # plt.show()
        # assert deflection on each node in the last time step
        # np.testing.assert_allclose(beam_analytical.u, vertical_displacements[:, -1], rtol=1e-5)

        rel_error = (vertical_displacements[int((n_beams-1)/2), :] - beam_analytical.u[int((n_beams-1)/2), :])\
                    /beam_analytical.u[int((n_beams-1)/2), :]
        rmse = np.sqrt(np.mean(rel_error[1:] ** 2))
        assert rmse < 0.01

    def test_undamped_timoshenko_beam_on_hinch_foundation_dynamic(self):
        """
        Tests point on an timoshenko beam which is supported by a hinch on each side of the beam. The calculation is
        dynamic without damping.
        Vertical deflection on the middle node is compared with the analytical solution.
        :return:
        """

        length_beam = 0.05
        n_beams = 201

        # Setup parameters euler beam
        time = np.linspace(0, 100, 5001)
        E = 20e5
        nu = 0
        G = E/(2*(1+nu))
        timoshenko_coef = 5/6 # rectangle

        I = 1
        rho = 2000
        A = 1
        L = 10
        F = -1000

        beam_analytical = SimpleSupportTimoshenkoNoDamping(n=200, ele_size=length_beam)
        beam_analytical.properties(E, G, timoshenko_coef, I, rho, A, L, F, time)
        beam_analytical.compute()

        # setup numerical model
        beam_nodes = [Node(i * length_beam, 0, 0) for i in range(n_beams)]
        beam_elements = [Element([beam_nodes[i], beam_nodes[i + 1]]) for i in range(n_beams - 1)]

        mesh = Mesh()
        mesh.add_unique_nodes_to_mesh(beam_nodes)
        mesh.add_unique_elements_to_mesh(beam_elements)

        material = Material()
        material.youngs_modulus = E  # Pa
        material.poisson_ratio = nu
        material.density = rho

        section = Section()
        section.area = A
        section.sec_moment_of_inertia = I
        section.shear_factor = timoshenko_coef

        beam = TimoshenkoBeamElementModelPart()
        beam.nodes = beam_nodes
        beam.elements = beam_elements

        beam.material = material
        beam.section = section
        beam.length_element = length_beam
        beam.damping_ratio = 0
        beam.radial_frequency_one = 2
        beam.radial_frequency_two = 5000

        beam.initialize()

        foundation1 = ConstraintModelPart(normal_dof=False, y_disp_dof=False, z_rot_dof=True)
        foundation1.nodes = [beam_nodes[0]]

        foundation2 = ConstraintModelPart(normal_dof=True, y_disp_dof=False, z_rot_dof=True)
        foundation2.nodes = [beam_nodes[-1]]

        load = LoadCondition(normal_dof=False, y_disp_dof=True, z_rot_dof=False)
        load.y_force = np.ones((1, len(time))) * F

        load.time = time
        load.nodes = [beam_nodes[int((n_beams-1)/2)]]

        model_parts = [beam, foundation1, foundation2, load]

        # set solver
        solver = NewmarkSolver()

        # populate global system
        global_system = GlobalSystem()
        global_system.mesh = mesh
        global_system.time = time
        global_system.solver = solver

        global_system.model_parts = model_parts

        # calculate
        global_system.main()
        vertical_displacements = np.array([node.displacements[:, 1] for node in beam_nodes])
        vertical_velocities = np.array([node.velocities[:, 1] for node in beam_nodes])
        #
        # plt.plot(time, vertical_displacements[int((n_beams-1)/2), :],marker='x')
        # plt.plot(time, beam_analytical.u[int((n_beams-1)/2), :])
        # # plt.plot(time, euler_beam.u[100,:], marker='v')
        # plt.show()


        # process signal of numerical and analytical solution
        freq_num, amplitude_num,_,_ = sp.fft_sig(vertical_displacements[int((n_beams-1)/2), :], int(1/ time[1]),
                                          nb_points=2**14)
        freq_timo, amplitude_analyt, _, _ = sp.fft_sig(beam_analytical.u[int((n_beams-1)/2), :], int(1 / time[1]),
                                               nb_points=2**14)
        # freq_euler, amplitude_analyt_euler, _, _ = sp.fft_sig(euler_beam.u[int((n_beams-1)/2), :], int(1 / time[1]),
        #                                        nb_points=2**14)
        rel_error_amp = (amplitude_num[142] - amplitude_analyt[142])/ amplitude_analyt[142]

        plt.plot(freq_num, amplitude_num,marker='x')
        plt.plot(freq_timo, amplitude_analyt,marker='o')
        # plt.plot(freq_euler, amplitude_analyt_euler, marker='v')
        plt.show()

        # assert deflection on each node in the last time step
        # np.testing.assert_allclose(beam_analytical.u, vertical_displacements[:, -1], rtol=1e-5)

        rel_error = (vertical_displacements[int((n_beams-1)/2), :] - beam_analytical.u[int((n_beams-1)/2), :])\
                    /beam_analytical.u[int((n_beams-1)/2), :]
        rmse = np.sqrt(np.mean(rel_error[1:] ** 2))
        assert rmse < 0.01


    def test_euler_beam_on_hinch_foundation_static(self):
        """
        Tests point on an euler beam which is supported by a hinch on each side of the beam, with a static calculation.
        Vertical deflection for each node is compared with the analytical solution
        :return:
        """
        length_beam = 0.01
        n_beams = 101

        # Setup parameters euler beam
        time = np.linspace(0, 2, 3)
        E = 20e7
        I = 1

        L = (n_beams - 1) * length_beam
        F = -1000
        x_F = 0.4

        # set and calculated analytical solution
        beam_analytical = SimpleSupportEulerStatic(ele_size=length_beam)
        beam_analytical.properties(E, I, L, F, x_F)
        beam_analytical.compute()

        # setup numerical model
        beam_nodes = [Node(i*length_beam,0,0) for i in range(n_beams)]
        beam_elements = [Element([beam_nodes[i], beam_nodes[i+1]]) for i in range(n_beams - 1)]

        mesh = Mesh()
        mesh.add_unique_nodes_to_mesh(beam_nodes)
        mesh.add_unique_elements_to_mesh(beam_elements)

        material = Material()
        material.youngs_modulus = E  # Pa
        material.poisson_ratio = 0.0
        material.density = 1000

        section = Section()
        section.area = 1
        section.sec_moment_of_inertia = I
        section.shear_factor = 0

        beam = TimoshenkoBeamElementModelPart()
        beam.nodes = beam_nodes
        beam.elements = beam_elements

        beam.material = material
        beam.section = section
        beam.length_element = length_beam
        beam.damping_ratio = 0.0000
        beam.radial_frequency_one = 2
        beam.radial_frequency_two = 500

        beam.initialize()

        foundation1 = ConstraintModelPart(normal_dof=False, y_disp_dof=False, z_rot_dof=True)
        foundation1.nodes = [beam_nodes[0]]

        foundation2 = ConstraintModelPart(normal_dof=True, y_disp_dof=False, z_rot_dof=True)
        foundation2.nodes = [beam_nodes[-1]]

        load = LoadCondition(normal_dof=False, y_disp_dof=True, z_rot_dof=False)
        load.y_force = np.ones((1, len(time))) * F

        load.time = time
        load.nodes = [beam_nodes[int((n_beams-1) * x_F / L)]]

        model_parts = [beam, foundation1, foundation2, load]

        # set solver
        solver = StaticSolver()

        # populate global system
        global_system = GlobalSystem()
        global_system.mesh = mesh
        global_system.time = time
        global_system.solver = solver

        global_system.model_parts = model_parts

        # calculate
        global_system.main()

        vertical_displacements = np.array([node.displacements[:, 1] for node in beam_nodes])

        # assert displacement on each node
        np.testing.assert_allclose(beam_analytical.u, vertical_displacements[:, -1])


    @pytest.mark.workinprogress
    def test_euler_beam_on_varying_foundation(self):

        # calculate analytical solution
        stiffness_spring = 2.75e5
        distance_springs = 1
        winkler_mod = stiffness_spring / distance_springs

        youngs_mod_beam = 4.41e05
        intertia_beam = 1
        y_load = -18e3

        distance_springs = 1
        n_sleepers = 50

        # setup numerical model
        # set time in two stages
        initialisation_time = np.linspace(0, 0.1, 100)
        calculation_time = np.linspace(initialisation_time[-1], 10, 5000)
        time = np.concatenate((initialisation_time, calculation_time[1:]))

        # set geometry
        depth_soil = 0.9
        element_model_parts_1, mesh_1 = create_horizontal_track(
            n_sleepers, distance_springs, depth_soil
        )
        bottom_boundary_1 = add_no_displacement_boundary_to_bottom(
            element_model_parts_1["soil"]
        )

        element_model_parts_2, mesh_2 = create_horizontal_track(
            n_sleepers, distance_springs, depth_soil
        )
        bottom_boundary_2 = add_no_displacement_boundary_to_bottom(
            element_model_parts_2["soil"]
        )

        for node in mesh_2.nodes:
            node.coordinates[0] = node.coordinates[0] + (n_sleepers)*distance_springs

        all_mesh = Mesh()

        # fill model parts
        rail_model_part = element_model_parts_1["rail"]
        rail_pad_model_part = element_model_parts_1["rail_pad"]
        sleeper_model_part = element_model_parts_1["sleeper"]
        soil_1 = element_model_parts_1["soil"]

        rail_model_part.nodes = rail_model_part.nodes + element_model_parts_2["rail"].nodes
        rail_model_part.elements = rail_model_part.elements +\
                                   [Element([rail_model_part.nodes[-1],element_model_parts_2["rail"].nodes[0]] )] +  element_model_parts_2["rail"].elements

        rail_pad_model_part.nodes = rail_pad_model_part.nodes + element_model_parts_2["rail_pad"].nodes
        rail_pad_model_part.elements = rail_pad_model_part.elements + element_model_parts_2["rail_pad"].elements

        sleeper_model_part.nodes = sleeper_model_part.nodes + element_model_parts_2["sleeper"].nodes
        sleeper_model_part.elements = sleeper_model_part.elements + element_model_parts_2["sleeper"].elements

        soil_2 = element_model_parts_2["soil"]

        all_mesh.add_unique_nodes_to_mesh(mesh_1.nodes)
        all_mesh.add_unique_nodes_to_mesh(mesh_2.nodes)

        all_mesh.add_unique_elements_to_mesh(mesh_1.elements)
        all_mesh.add_unique_elements_to_mesh(mesh_2.elements)

        all_mesh.reorder_node_ids()
        all_mesh.reorder_element_ids()

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

        soil_1.stiffness = stiffness_spring / depth_soil  # 300e6
        soil_1.damping = 0

        soil_2.stiffness = stiffness_spring / depth_soil * 10 # 300e6
        soil_2.damping = 0

        # set load
        velocities = np.ones(len(time)) * 10
        load = add_moving_point_load_to_track(
            rail_model_part,
            time,
            len(initialisation_time),
            velocities,
            y_load=y_load,
            start_coords=np.array([0, 0, 0]),
        )

        # set solver
        solver = NewmarkSolver()

        # populate global system
        global_system = GlobalSystem()
        global_system.mesh = all_mesh
        global_system.time = time
        global_system.solver = solver

        # get all element model parts from dictionary
        model_parts = [[rail_model_part, rail_pad_model_part, sleeper_model_part, soil_1, soil_2],
                       list(bottom_boundary_1.values()), list(bottom_boundary_2.values()), list(load.values())
        ]
        global_system.model_parts = list(itertools.chain.from_iterable(model_parts))

        # calculate
        global_system.main()

    @pytest.mark.workinprogress
    def test_moving_load_on_cantilever_beam(self, set_up_material, set_up_euler_section):
        nodes_beam = [Node(0.0,0.0,0.0), Node(1.0, 0.0, 0.0)]
        elements_beam = [Element(nodes_beam)]

        mesh = Mesh()
        mesh.nodes = nodes_beam
        mesh.elements = elements_beam

        beam = TimoshenkoBeamElementModelPart()
        beam.nodes = nodes_beam
        beam.elements = elements_beam

        beam.section = set_up_euler_section
        beam.material = set_up_material

        beam.length_element = 1
        beam.calculate_mass()

        beam.damping_ratio = 0.0
        beam.radial_frequency_one = 1e-12
        beam.radial_frequency_two = 1e-12

        # initialise beam matrices
        beam.initialize()

        # set time
        # time = np.array([0, 1e3, 2e3, 3e3, 4e3, 5e3])
        time = np.array([0,2.1333333333,	4.2666666667,	6.4,	8.5333333333,	10.666666667])

        # set moving load
        force = LineLoadCondition(normal_dof=True, y_disp_dof=True, z_rot_dof=True)
        force.nodes = nodes_beam
        force.elements = elements_beam
        force.time = time

        force.initialize_matrices()

        # set coordinate of moving load per timestep
        moving_coords = [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.25, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [0.75, 0.0, 0.0],
            [1, 0.0, 0.0],
        ]

        # sets moving load on timoshenko beam
        force.set_moving_point_load(beam, moving_coords, time, y_force=np.array([1e-2, 1e5, 1e5, 1e5, 1e5, 1e5]))

        no_disp_boundary_condition = ConstraintModelPart()
        no_disp_boundary_condition.nodes = [nodes_beam[1]]

        # set solver
        solver = NewmarkSolver()

        # populate global system
        global_system = GlobalSystem()
        global_system.mesh = mesh
        global_system.time = time
        global_system.solver = solver

        # get all element model parts from dictionary
        model_parts = [
            beam, force, no_disp_boundary_condition
        ]

        global_system.model_parts = model_parts

        # calculate
        global_system.main()

        y_displacement = global_system.displacements[:,1]





    def test_dynamic_shear_load_on_beam(self):
        """
        Test a vertical load on a horizontal cantilever euler beam. Where the load is added at the free end of the beam.
        Test if the eigen frequency and amplitude are as expected
        :return:
        """
        length_rod = 0.01
        n_beams = 101

        E = 20e6
        I = 1e-4
        A = 0.01
        L = (n_beams - 1) * length_rod
        F = -1000
        rho = 2000

        time = np.linspace(0, 5, 2001)

        # analytically calculate first eigenfrequency cantilever beam
        mass = A * L * rho

        eig_freq_const = 1.875
        first_eig_freq = eig_freq_const**2 * np.sqrt(E*I/(mass *L**4)) / (2*np.pi)

        # analytically calculate load on cantilever beam
        pulse_load = PulseLoadNoDamping(ele_size=length_rod)
        pulse_load.properties(E, I, rho, A, L, F, time)
        pulse_load.compute()

        # setup numerical model
        rod_nodes = [Node(i * length_rod, 0, 0) for i in range(n_beams)]
        rod_elements = [Element([rod_nodes[i], rod_nodes[i + 1]]) for i in range(n_beams - 1)]

        mesh = Mesh()
        mesh.add_unique_nodes_to_mesh(rod_nodes)
        mesh.add_unique_elements_to_mesh(rod_elements)

        material = Material()
        material.youngs_modulus = E  # Pa
        material.poisson_ratio = 0.0
        material.density = rho

        section = Section()
        section.area = A
        section.sec_moment_of_inertia = I
        section.shear_factor = 0

        beam = TimoshenkoBeamElementModelPart()
        beam.nodes = rod_nodes
        beam.elements = rod_elements

        beam.length_element = length_rod

        beam.material = material
        beam.section = section
        beam.damping_ratio = 0.0000
        beam.radial_frequency_one = 2
        beam.radial_frequency_two = 500

        beam.initialize()

        # setup cantilever foundation
        foundation1 = ConstraintModelPart(normal_dof=False, y_disp_dof=False, z_rot_dof=False)
        foundation1.nodes = [rod_nodes[0]]

        # set load at beam end
        load = LoadCondition(normal_dof=False, y_disp_dof=True, z_rot_dof=False)
        load.y_force = np.ones((1, len(time))) * F
        load.time = time
        load.nodes = [rod_nodes[-1]]

        model_parts = [beam, foundation1, load]

        # set solver
        solver = NewmarkSolver()

        # populate global system
        global_system = GlobalSystem()
        global_system.mesh = mesh
        global_system.time = time
        global_system.solver = solver

        global_system.model_parts = model_parts

        # calculate
        global_system.main()

        # get numerical frequencies and amplitudes
        vert_velocities = np.array([node.velocities[:, 1] for node in rod_nodes])

        freq_num, amplitude_num, _, _ = sp.fft_sig(vert_velocities[-1, :], int(1 / time[1]),
                                                   nb_points=2 ** 14)
        freq_ana, amplitude_ana, _, _ = sp.fft_sig(pulse_load.v[-1, :], int(1 / time[1]),
                                                   nb_points=2 ** 14)

        # check if first eigen frequency is as expected
        assert first_eig_freq == pytest.approx(freq_num[amplitude_num == np.max(amplitude_num)][0], rel=0.01)

        # check if amplitude is as expected
        assert max(amplitude_ana) == pytest.approx(max(amplitude_num), rel=0.01)

    @pytest.mark.skip(reason="work in progress")
    def test_train_on_infinite_euler_beam_without_damping(self):
        """
        todo Work in progress
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


@pytest.fixture
def set_up_material():
    # Steel

    material = Material()
    material.youngs_modulus = 1  # Pa
    material.poisson_ratio = 0.0
    material.density = 1000
    return material

    # material = Material()
    # material.youngs_modulus = 200e9  # Pa
    # material.poisson_ratio = 0.0
    # material.density = 8000
    # return material


@pytest.fixture
def set_up_euler_section():

    section = Section()
    section.area = 1e-3
    section.sec_moment_of_inertia = 1
    section.shear_factor = 0
    return section

    # section = Section()
    # section.area = 1e-3
    # section.sec_moment_of_inertia = 2e-5
    # section.shear_factor = 0
    # return section