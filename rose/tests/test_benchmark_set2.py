import json
import os.path
import pytest

from rose.base.global_system import *
from rose.base.model_part import Material, Section, TimoshenkoBeamElementModelPart, RodElementModelPart
from rose.utils.mesh_utils import *
from rose.utils.plot_utils import *

import rose.tests.utils.signal_proc as sp

from analytical_solutions.analytical_wave_prop import OneDimWavePropagation
from analytical_solutions.simple_supported import \
    SimpleSupportEulerNoDamping, \
    SimpleSupportEulerStatic, \
    SimpleSupportEulerWithDamping, \
    SimpleSupportTimoshenkoNoDamping
from analytical_solutions.cantilever_beam import PulseLoadNoDamping
from analytical_solutions.winkler import MovingLoad
import matplotlib.pyplot as plt

# if RENEW_BENCHMARKS is true, the analytical solutions will be recalculated, results will be plotted together with the
# numerical solution.
RENEW_BENCHMARKS = False
# TEST_PATH = os.path.join('rose', 'tests')
TEST_PATH = '.'


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

        calculation_time_steps = 100000

        # initialisation_time = np.linspace(0, 0.01, 10000)
        # calculation_time = np.linspace(initialisation_time[-1], initialisation_time[-1] + 0.1, calculation_time_steps)
        # time = np.concatenate((initialisation_time, calculation_time[1:]))
        # E = 20e3
        time = np.linspace(0, 100, 10001)
        # E = 20e3
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

        # load.y_force[0,:len(initialisation_time)] = np.linspace(0,F,len(initialisation_time))

        load.time = time
        load.nodes = [beam_nodes[int((n_beams-1)/2)]]

        model_parts = [beam, foundation1, foundation2, load]

        # set solver
        solver = NewmarkSolver()
        # solver = ZhaiSolver()

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

    def test_euler_beam_on_varying_foundation(self):
        """
        Tests a moving load on a beam with a winkler foundation. Where the stiffness of the winkler foundation changes
        halfway the track. Vertical deflection on each node of the rail is compared with the analytical solution.
        :return:
        """

        # Set parameters of beam and winkler foundation
        stiffness_spring = 400e3
        stiffness_spring_2 = stiffness_spring * 10
        distance_springs = 0.5
        winkler_mod_1 = stiffness_spring / distance_springs
        winkler_mod_2 = stiffness_spring_2 / distance_springs

        youngs_mod_beam = 1.28e7
        intertia_beam = 1
        rho = 120
        y_load = -18e3

        n_sleepers = 100

        # setup numerical model
        # set time in two stages
        calculation_time_steps = 500

        initialisation_time = np.linspace(0, 10, 100)
        # calculation_time = np.linspace(initialisation_time[-1], initialisation_time[-1] + 5, calculation_time_steps)
        calculation_time = np.linspace(initialisation_time[-1], initialisation_time[-1] + 10, calculation_time_steps)
        time = np.concatenate((initialisation_time, calculation_time[1:]))

        # set geometry
        # set left part of rail
        depth_soil = 0.9
        element_model_parts_1, mesh_1 = create_horizontal_track(
            n_sleepers, distance_springs, depth_soil
        )
        bottom_boundary_1 = add_no_displacement_boundary_to_bottom(
            element_model_parts_1["soil"]
        )

        # set right part of rail
        element_model_parts_2, mesh_2 = create_horizontal_track(
            n_sleepers, distance_springs, depth_soil
        )
        bottom_boundary_2 = add_no_displacement_boundary_to_bottom(
            element_model_parts_2["soil"]
        )

        # reset coordinates of left part of the rail
        for node in mesh_1.nodes:
            node.coordinates[0] = node.coordinates[0] - (n_sleepers) * distance_springs

        all_mesh = Mesh()

        # fill model parts
        rail_model_part = element_model_parts_1["rail"]
        rail_pad_model_part = element_model_parts_1["rail_pad"]
        sleeper_model_part = element_model_parts_1["sleeper"]
        soil_1 = element_model_parts_1["soil"]

        # combine left and right rail model part
        rail_model_part.elements = rail_model_part.elements + \
                                   [Element([element_model_parts_1["rail"].nodes[-1],
                                             element_model_parts_2["rail"].nodes[0]])] \
                                   + element_model_parts_2["rail"].elements
        rail_model_part.nodes = rail_model_part.nodes + element_model_parts_2["rail"].nodes

        # combine left and right sleeper model part
        sleeper_model_part.nodes = sleeper_model_part.nodes + element_model_parts_2["sleeper"].nodes
        sleeper_model_part.elements = sleeper_model_part.elements + element_model_parts_2["sleeper"].elements

        # add left and right soil and rail_pad model parts to the mesh
        soil_2 = element_model_parts_2["soil"]
        rail_pad_model_part_2 = element_model_parts_2["rail_pad"]

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
        material.density = rho  # 7860

        section = Section()
        section.area = 1
        section.sec_moment_of_inertia = intertia_beam
        section.shear_factor = 0

        rail_model_part.section = section
        rail_model_part.material = material
        rail_model_part.damping_ratio = 0.3
        rail_model_part.radial_frequency_one = 2
        rail_model_part.radial_frequency_two = 500

        rail_model_part.initialize()

        rail_pad_model_part.mass = 0.000001  # 5
        rail_pad_model_part.stiffness = stiffness_spring / 0.1
        rail_pad_model_part.damping = 0  # 12e3

        rail_pad_model_part_2.mass = 0.000001  # 5
        rail_pad_model_part_2.stiffness = stiffness_spring_2 / 0.1
        rail_pad_model_part_2.damping = 0  # 12e3

        sleeper_model_part.mass = 0.0000001  # 162.5
        sleeper_model_part.distance_between_sleepers = distance_springs

        soil_1.stiffness = stiffness_spring / depth_soil  # 300e6
        soil_1.damping = 0

        soil_2.stiffness = stiffness_spring_2 / depth_soil  # 300e6
        soil_2.damping = 0

        # set load
        position = np.array([node.coordinates[0] for node in rail_model_part.nodes])
        velocity = (position[-1] - position[0]) / (time[-1] - time[len(initialisation_time)])

        # set moving load on rail_model_part
        velocities = np.ones(len(time)) * velocity
        velocities[0:len(initialisation_time)] = 0
        load = add_moving_point_load_to_track(
            rail_model_part,
            time,
            len(initialisation_time),
            velocities,
            y_load=y_load,
        )

        # constraint rotation at the side boundaries
        side_boundaries = ConstraintModelPart(normal_dof=True, y_disp_dof=True, z_rot_dof=False)
        side_boundaries.nodes = [rail_model_part.nodes[0], rail_model_part.nodes[-1]]

        # set solver
        solver = NewmarkSolver()
        # solver = ZhaiSolver()

        # populate global system
        global_system = GlobalSystem()
        global_system.mesh = all_mesh
        global_system.time = time
        global_system.solver = solver

        # get all element model parts from dictionary
        model_parts = [[rail_model_part, rail_pad_model_part, rail_pad_model_part_2, sleeper_model_part, soil_1, soil_2,
                        side_boundaries],
                       list(bottom_boundary_1.values()), list(bottom_boundary_2.values()), list(load.values())]
        # get all element model parts from dictionary
        # model_parts = [[rail_model_part, rail_pad_model_part, rail_pad_model_part_2, sleeper_model_part, soil_1, soil_2
        #                 ],
        #                list(bottom_boundary_1.values()), list(bottom_boundary_2.values()), list(load.values())]

        global_system.model_parts = list(itertools.chain.from_iterable(model_parts))

        # calculate
        global_system.main()

        # get vertical displacement vertical rail
        vertical_displacements_rail = np.array(
            [node.displacements[:, 1] for node in global_system.model_parts[0].nodes])
        coords = np.array([node.coordinates[0] for node in global_system.model_parts[0].nodes])

        # recalculate analytical solution and compare with numerical solution
        # if True:
        if RENEW_BENCHMARKS:
            # calculate analytical solution
            position = np.linspace(rail_model_part.nodes[0].coordinates[0],
                                   rail_model_part.nodes[-1].coordinates[0], calculation_time_steps)
            p = MovingLoad(a=0.005)
            p.parameters(position, velocity, youngs_mod_beam, intertia_beam, rho, [winkler_mod_1, winkler_mod_2],
                         y_load)
            p.solve()

            # # todo check time discreatisation and force build-up
            plt.plot(coords,vertical_displacements_rail[:, int(len(initialisation_time) + len(calculation_time) * 1 / 4)], color="k")
            plt.plot(coords,vertical_displacements_rail[:, int(len(initialisation_time) + len(calculation_time) * 2 / 4)], color="k")
            plt.plot(coords, vertical_displacements_rail[:, int(len(initialisation_time) + len(calculation_time) * 3 / 4)], color="k")
            plt.plot(p.position, p.displacement[:, int(len(p.time) * 1/4)], color="r", marker='x')
            plt.plot(p.position, p.displacement[:, int(len(p.time) * 2/4)], color="r", marker='x')
            plt.plot(p.position, p.displacement[:, int(len(p.time) * 3/4)], color="r", marker='x')
            # plt.plot(p.qsi, p.displacement[:, int(n_sleepers*2)], color="k")
            plt.show()

            # create animation
            create_animation("beam_on_varying_winkler_foundation.html", (p.position, coords),
                             (np.transpose(p.displacement), vertical_displacements_rail[:,
                                                            len(initialisation_time) - 1:]))

            # create dictionary for results
            result = {"time": time.tolist(),
                      "u": vertical_displacements_rail.tolist()}

            # dump results
            with open(os.path.join(TEST_PATH, 'test_data', 'beam_on_varying_winkler_foundation.json'),
                      "w") as f:
                json.dump(result, f, indent=2)

        # calculate analytical solution
        position = np.linspace(rail_model_part.nodes[0].coordinates[0],
                               rail_model_part.nodes[-1].coordinates[0], calculation_time_steps)
        p = MovingLoad(a=0.005)
        p.parameters(position, velocity, youngs_mod_beam, intertia_beam, rho, [winkler_mod_1, winkler_mod_2],
                     y_load)
        p.solve()

        # # todo check time discreatisation and force build-up
        # plt.plot(coords,
        #          vertical_displacements_rail[:, int(len(initialisation_time) + len(calculation_time) * 1 / 4)],
        #          color="k")
        # plt.plot(coords,
        #          vertical_displacements_rail[:, int(len(initialisation_time) + len(calculation_time) * 2 / 4)],
        #          color="k")
        # plt.plot(coords,
        #          vertical_displacements_rail[:, int(len(initialisation_time) + len(calculation_time) * 3 / 4)],
        #          color="k")
        # plt.plot(p.position, p.displacement[:, int(len(p.time) * 1 / 4)], color="r", marker='x')
        # plt.plot(p.position, p.displacement[:, int(len(p.time) * 2 / 4)], color="r", marker='x')
        # plt.plot(p.position, p.displacement[:, int(len(p.time) * 3 / 4)], color="r", marker='x')
        # # plt.plot(p.qsi, p.displacement[:, int(n_sleepers*2)], color="k")
        # plt.show()


        # retrieve results from file
        with open(os.path.join(TEST_PATH, 'test_data', 'beam_on_varying_winkler_foundation.json')) as f:
            beam_numerical = json.load(f)

        # get expected displacement
        expected_displacement = np.array(beam_numerical['u'])

        # assert
        np.testing.assert_array_almost_equal(expected_displacement, vertical_displacements_rail)

