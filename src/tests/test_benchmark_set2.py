import json
import os.path
import pytest

from src.global_system import *
from src.model_part import Material, Section, TimoshenkoBeamElementModelPart, RodElementModelPart
from src.mesh_utils import *

import src.tests.utils.signal_proc as sp

from analytical_solutions.analytical_wave_prop import OneDimWavePropagation
from analytical_solutions.simple_supported import \
    SimpleSupportEulerNoDamping, \
    SimpleSupportEulerStatic, \
    SimpleSupportEulerWithDamping, \
    SimpleSupportTimoshenkoNoDamping
from analytical_solutions.cantilever_beam import PulseLoadNoDamping
import matplotlib.pyplot as plt

RENEW_BENCHMARKS = False
TEST_PATH = os.path.join('src', 'tests')
# TEST_PATH = '.'


class TestBenchmarkSet2:
    """
    In Benchmark set 2, benchmarks are verified with plots of the analytical solution, values are tested regressive
    """

    def test_dynamic_load_on_rod(self):
        """
        Test a wave propagation through a rod element.
        :return:
        """
        length_rod = 0.01
        n_beams = 101

        # Setup parameters rod element
        E = 20e5
        A = 1
        L = (n_beams - 1) * length_rod
        F = -1000
        rho = 3

        if RENEW_BENCHMARKS:
            # set and calculated analytical solution
            rod_analytical = OneDimWavePropagation(nb_cycles=10, nb_terms=100)
            rod_analytical.properties(rho, E, F, L, n_beams)
            rod_analytical.solution()
            rod_analytical.write_results()

        # load data from analytical solution
        with open(os.path.join(TEST_PATH, 'test_data', 'rod.json')) as rod_file:
            rod_analytical = json.load(rod_file)

        time = rod_analytical['time']

        # setup numerical model
        rod_nodes = [Node(i * length_rod, 0, 0) for i in range(n_beams)]
        rod_elements = [Element([rod_nodes[i], rod_nodes[i + 1]]) for i in range(n_beams - 1)]

        mesh = Mesh()
        mesh.add_unique_nodes_to_mesh(rod_nodes)
        mesh.add_unique_elements_to_mesh(rod_elements)

        rod = RodElementModelPart()
        rod.nodes = rod_nodes
        rod.elements = rod_elements

        rod.length_element = length_rod
        rod.stiffness = E * A / length_rod
        rod.mass = rho * A * length_rod

        foundation1 = ConstraintModelPart(normal_dof=False, y_disp_dof=False, z_rot_dof=False)
        foundation1.nodes = [rod_nodes[0]]

        foundation2 = ConstraintModelPart(normal_dof=True, y_disp_dof=False, z_rot_dof=False)
        foundation2.nodes = [rod_nodes[-1]]

        load = LoadCondition(normal_dof=True, y_disp_dof=False, z_rot_dof=False)
        load.normal_force = np.ones((1, len(time))) * F

        load.time = time
        load.nodes = [rod_nodes[-1]]

        model_parts = [rod, foundation1, foundation2, load]

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

        # get horizontal displacements and velocities
        hor_displacements = np.array([node.displacements[:, 0] for node in rod_nodes])
        hor_velocities = np.array([node.velocities[:, 0] for node in rod_nodes])

        if RENEW_BENCHMARKS:
            plt.plot(time, hor_velocities[-1, :], marker='o')
            plt.plot(time, np.array(rod_analytical['v'])[-1, :], marker='x')
            plt.show()

            # create dictionary for results
            result = {"time": time,
                      "u": hor_displacements.tolist(),
                      "v": hor_velocities.tolist()}

            # dump results
            with open(os.path.join(TEST_PATH, 'test_data', 'dynamic_load_on_rod_num.json'), "w") as f:
                json.dump(result, f, indent=2)

        # retrieve results from file
        with open(os.path.join(TEST_PATH, 'test_data', 'dynamic_load_on_rod_num.json')) as rod_file:
            rod_numerical = json.load(rod_file)

        # get expected displacement and velocity
        expected_displacement = np.array(rod_numerical['u'])
        expected_velocity = np.array(rod_numerical['v'])

        # assert
        np.testing.assert_array_almost_equal(expected_displacement, hor_displacements)
        np.testing.assert_array_almost_equal(expected_velocity, hor_velocities)

    def test_dynamic_normal_load_on_beam(self):
        """
        Test a wave propagation through an euler beam element following a normal pulse load.
        :return:
        """

        length_rod = 0.01
        n_beams = 101

        # Setup parameters euler beam
        E = 20e5
        I = 1
        A = 1
        L = (n_beams - 1) * length_rod
        F = -1000
        rho = 3

        # load data from analytical solution
        with open(os.path.join(TEST_PATH, 'test_data','rod.json')) as rod_file:
            rod_analytical = json.load(rod_file)

        time = rod_analytical['time']

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

        foundation1 = ConstraintModelPart(normal_dof=False, y_disp_dof=False, z_rot_dof=False)
        foundation1.nodes = [rod_nodes[0]]

        foundation2 = ConstraintModelPart(normal_dof=True, y_disp_dof=False, z_rot_dof=False)
        foundation2.nodes = [rod_nodes[-1]]

        load = LoadCondition(normal_dof=True, y_disp_dof=False, z_rot_dof=False)
        load.normal_force = np.ones((1, len(time))) * F

        load.time = time
        load.nodes = [rod_nodes[-1]]

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

        hor_displacements = np.array([node.displacements[:, 0] for node in rod_nodes])
        hor_velocities = np.array([node.velocities[:, 0] for node in rod_nodes])

        if RENEW_BENCHMARKS:
            plt.plot(time, hor_velocities[-1, :], marker='o')
            plt.plot(time, np.array(rod_analytical['v'])[-1, :], marker='x')
            plt.show()

            # create dictionary for results
            result = {"time": time,
                      "u": hor_displacements.tolist(),
                      "v": hor_velocities.tolist()}

            # dump results
            with open(os.path.join(TEST_PATH, 'test_data', 'dynamic_load_on_beam_num.json'), "w") as f:
                json.dump(result, f, indent=2)

        # retrieve results from file
        with open(os.path.join(TEST_PATH, 'test_data', 'dynamic_load_on_beam_num.json')) as rod_file:
            beam_numerical = json.load(rod_file)

        # get expected displacement and velocity
        expected_displacement = np.array(beam_numerical['u'])
        expected_velocity = np.array(beam_numerical['v'])

        # assert
        np.testing.assert_array_almost_equal(expected_displacement, hor_displacements)
        np.testing.assert_array_almost_equal(expected_velocity, hor_velocities)

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

        damping_ratio = 0.02
        omega1 = 2
        omega2 = 5000

        # calculate damping coefficients
        constant = (
            2
            * damping_ratio
            / (omega1 + omega2)
        )
        a0 = omega1 * omega2 * constant
        a1 = constant
        coefs = [a0, a1]

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
        vertical_velocities = np.array([node.velocities[:, 1] for node in beam_nodes])

        # recalculate analytical solution and compare with numerical solution
        if RENEW_BENCHMARKS:
            beam_analytical = SimpleSupportEulerWithDamping(n=150, ele_size=length_beam)
            beam_analytical.properties(E, I, rho, A, L, F, coefs, time)
            beam_analytical.compute()

            plt.plot(time, vertical_displacements[int((n_beams-1)/2), :])
            plt.plot(time, beam_analytical.u[int((n_beams-1)/2), :])
            plt.show()

            # create dictionary for results
            result = {"time": time.tolist(),
                      "u": vertical_displacements.tolist(),
                      "v": vertical_velocities.tolist()}

            # dump results
            with open(os.path.join(TEST_PATH, 'test_data', 'simply_supported_damped_euler_beam_num.json'), "w") as f:
                json.dump(result, f, indent=2)

        # retrieve results from file
        with open(os.path.join(TEST_PATH, 'test_data', 'simply_supported_damped_euler_beam_num.json')) as f:
            beam_numerical = json.load(f)

        # get expected displacement and velocity
        expected_displacement = np.array(beam_numerical['u'])
        expected_velocity = np.array(beam_numerical['v'])

        # assert
        np.testing.assert_array_almost_equal(expected_displacement, vertical_displacements)
        np.testing.assert_array_almost_equal(expected_velocity, vertical_velocities)

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

        # recalculate analytical solution and compare with numerical solution
        if RENEW_BENCHMARKS:
            beam_analytical = SimpleSupportTimoshenkoNoDamping(n=100, ele_size=length_beam)
            beam_analytical.properties(E, G, timoshenko_coef, I, rho, A, L, F, time)
            beam_analytical.compute()

            plt.plot(time, vertical_displacements[int((n_beams-1)/2), :],marker='x')
            plt.plot(time, beam_analytical.u[int((n_beams-1)/2), :])
            plt.show()

            # create dictionary for results
            result = {"time": time.tolist(),
                      "u": vertical_displacements.tolist(),
                      "v": vertical_velocities.tolist()}

            # dump results
            with open(os.path.join(TEST_PATH, 'test_data', 'simply_supported_timoshenko_beam_num.json'),
                      "w") as f:
                json.dump(result, f, indent=2)

        # retrieve results from file
        with open(os.path.join(TEST_PATH, 'test_data', 'simply_supported_timoshenko_beam_num.json')) as f:
            beam_numerical = json.load(f)

        # get expected displacement and velocity
        expected_displacement = np.array(beam_numerical['u'])
        expected_velocity = np.array(beam_numerical['v'])

        # assert
        np.testing.assert_array_almost_equal(expected_displacement, vertical_displacements)
        np.testing.assert_array_almost_equal(expected_velocity, vertical_velocities)
