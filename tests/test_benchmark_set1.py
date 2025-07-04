import pytest

from rose.model.global_system import *
from rose.model.train_model import TrainModel
from rose.model.model_part import Material, Section, TimoshenkoBeamElementModelPart
from rose.model.boundary_conditions import MovingPointLoad
from rose.pre_process.mesh_utils import *
from rose.post_processing.plot_utils import *

from SignalProcessingTools.time_signal import TimeSignalProcessing

from solvers.newmark_solver import NewmarkExplicit

from analytical_solutions.simple_supported import \
    SimpleSupportEulerNoDamping, \
    SimpleSupportEulerStatic
from analytical_solutions.cantilever_beam import PulseLoadNoDamping
from analytical_solutions.beam_with_hinge import BeamWithHinge


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

        load = MovingPointLoad(x_disp_dof=rail_model_part.x_disp_dof, y_disp_dof=rail_model_part.y_disp_dof,
                               z_rot_dof=rail_model_part.z_rot_dof, start_distance=5)
        load.time = time
        load.contact_model_part = rail_model_part
        load.y_force = y_load
        load.velocities = velocities
        load.initialisation_time = initialisation_time
        load.elements = rail_model_part.elements
        load.nodes = rail_model_part.nodes


        # set solver
        solver = NewmarkExplicit()

        # populate global system
        global_system = GlobalSystem()
        global_system.mesh = mesh
        global_system.time = time
        global_system.solver = solver

        # get all element model parts from dictionary
        model_parts = [
            list(element_model_parts.values()),
            list(bottom_boundary.values()),
            [load],
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

        # set up hinge foundation
        foundation1 = ConstraintModelPart(x_disp_dof=False, y_disp_dof=False, z_rot_dof=True)
        foundation1.nodes = [beam_nodes[0]]

        foundation2 = ConstraintModelPart(x_disp_dof=True, y_disp_dof=False, z_rot_dof=True)
        foundation2.nodes = [beam_nodes[-1]]

        # set load on middle node
        load = LoadCondition(x_disp_dof=False, y_disp_dof=True, z_rot_dof=False)
        load.y_force_matrix = np.ones((1, len(time))) * F
        load.time = time
        load.nodes = [beam_nodes[int((n_beams-1)/2)]]

        model_parts = [beam, foundation1, foundation2, load]

        # set solver
        solver = NewmarkExplicit()

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
        signal_num = TimeSignalProcessing(time,vertical_displacements[int((n_beams-1)/2), :],int(1 / time[1]))
        signal_num.fft(2**14, half_representation=True)
        amplitude_num = signal_num.amplitude

        signal_analyt = TimeSignalProcessing(time, beam_analytical.u[int((n_beams-1)/2), :], int(1 / time[1]))
        signal_analyt.fft(2**14, half_representation=True)
        amplitude_signal_analyt = signal_analyt.amplitude

        # assert if signal amplitudes are approximately equal at eigen frequency
        assert amplitude_num[163] == pytest.approx(amplitude_signal_analyt[163], rel=1e-2)



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

        foundation1 = ConstraintModelPart(x_disp_dof=False, y_disp_dof=False, z_rot_dof=True)
        foundation1.nodes = [beam_nodes[0]]

        foundation2 = ConstraintModelPart(x_disp_dof=True, y_disp_dof=False, z_rot_dof=True)
        foundation2.nodes = [beam_nodes[-1]]

        load = LoadCondition(x_disp_dof=False, y_disp_dof=True, z_rot_dof=False)
        load.y_force_matrix = np.ones((1, len(time))) * F

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

        # setup cantilever foundation
        foundation1 = ConstraintModelPart(x_disp_dof=False, y_disp_dof=False, z_rot_dof=False)
        foundation1.nodes = [rod_nodes[0]]

        # set load at beam end
        load = LoadCondition(x_disp_dof=False, y_disp_dof=True, z_rot_dof=False)
        load.y_force_matrix = np.ones((1, len(time))) * F
        load.time = time
        load.nodes = [rod_nodes[-1]]

        model_parts = [beam, foundation1, load]

        # set solver
        solver = NewmarkExplicit()

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

        signal_num = TimeSignalProcessing(time,vert_velocities[-1, :],int(1 / time[1]))
        signal_num.fft(2**14, half_representation=True)
        amplitude_num = signal_num.amplitude
        freq_num = signal_num.frequency

        signal_analyt = TimeSignalProcessing(time, pulse_load.v[-1, :], int(1 / time[1]))
        signal_analyt.fft(2**14, half_representation=True)
        amplitude_ana = signal_analyt.amplitude

        # Check if first eigen frequency is as expected
        assert first_eig_freq == pytest.approx(freq_num[amplitude_num == np.max(amplitude_num)][0], rel=0.01)

        # Check if amplitude is as expected
        assert max(amplitude_ana) == pytest.approx(max(amplitude_num), rel=0.01)


    def test_point_load_on_beam_with_hinge(self):
        """
        Tests point load on beam with a hinge in the middle. This test tests the maximum displacement at the location of
        the hinge, for a moving point load, from left to right, across the hinge.
        :return:
        """

        # Set parameters of beam

        length_beam = 1
        youngs_mod_beam = 1.0e7
        intertia_beam = 1
        rho = 10
        y_load = -10e3

        # discretisation
        n_beams = 100
        hinge_location = 50
        tot_length_beam = length_beam*n_beams

        # setup numerical model
        # set time integration
        calculation_time_steps = 101
        time = np.linspace(0, 10, calculation_time_steps)

        # discretise the beam
        nodes_track = [Node(i * length_beam, 0.0, 0.0) for i in range(n_beams +1 )]
        elements_track = [
            Element([nodes_track[i], nodes_track[i + 1]]) for i in range(n_beams)
        ]

        all_mesh = Mesh()
        all_mesh.add_unique_nodes_to_mesh(nodes_track)
        all_mesh.add_unique_elements_to_mesh(elements_track)
        all_mesh.reorder_element_ids()
        all_mesh.reorder_node_ids()

        rail_model_part = Rail()
        rail_model_part.elements = elements_track
        rail_model_part.nodes = nodes_track
        rail_model_part.length_rail = length_beam

        # add hinge to rail
        hinge_model_part = Hinge()
        hinge_model_part.rotational_stiffness = 0

        rail_model_parts, all_mesh = add_semi_rigid_hinge_at_x(rail_model_part,hinge_model_part, hinge_location, all_mesh)

        # get rail nodes and elements and reorder node and element ids
        rail_nodes = [part.nodes for part in rail_model_parts]
        rail_nodes = list(itertools.chain.from_iterable(rail_nodes))
        rail_node_idxs = [node.index for node in rail_nodes]
        _, unique_idxs = np.unique(rail_node_idxs, return_index=True)
        rail_nodes = list(np.array(rail_nodes)[unique_idxs])

        rail_elements = [part.elements for part in rail_model_parts]
        rail_elements = list(itertools.chain.from_iterable(rail_elements))
        all_mesh.reorder_node_ids()
        all_mesh.reorder_element_ids()

        # set rail elements
        material = Material()
        material.youngs_modulus = youngs_mod_beam  # Pa
        material.poisson_ratio = 0.0
        material.density = rho  # 7860

        section = Section()
        section.area = 1
        section.sec_moment_of_inertia = intertia_beam
        section.shear_factor = 0

        for part in rail_model_parts:
            part.section = section
            part.material = material

        # set load
        position = np.array([node.coordinates[0] for node in rail_model_part.nodes])
        velocity = (position[1] - position[0]) / (time[1])

        # set moving load on rail_model_part
        velocities = np.ones(len(time)) * velocity

        # set moving load
        load = MovingPointLoad(x_disp_dof=rail_model_part.x_disp_dof, y_disp_dof=rail_model_part.y_disp_dof,
                               z_rot_dof=rail_model_part.z_rot_dof)
        load.time = time
        load.contact_model_part = rail_model_part
        load.contact_model_parts = rail_model_parts
        load.y_force = y_load
        load.velocities = velocities
        load.initialisation_time = []
        load.nodes = rail_nodes
        load.elements = rail_elements

        # constraint rotation at the side boundaries
        left_boundary = ConstraintModelPart(x_disp_dof=False, y_disp_dof=False, z_rot_dof=False)
        right_boundary = ConstraintModelPart(x_disp_dof=False, y_disp_dof=False, z_rot_dof=True)
        left_boundary.nodes = [rail_model_parts[0].nodes[0]]
        right_boundary.nodes =[rail_model_parts[-1].nodes[-1]]

        # set solver
        solver = StaticSolver()

        # populate global system
        global_system = GlobalSystem()
        global_system.mesh = all_mesh
        global_system.time = time
        global_system.solver = solver

        global_system.is_rayleigh_damping = True
        global_system.damping_ratio = 0.3
        global_system.radial_frequency_one = 2
        global_system.radial_frequency_two = 500

        # get all element model parts from dictionary
        model_parts = rail_model_parts + [left_boundary, right_boundary, load]

        global_system.model_parts = model_parts

        # calculate
        global_system.main()

        # get vertical displacement vertical rail
        vertical_displacements_rail = np.array(
            [node.displacements[:, 1] for node in rail_nodes])
        coords = np.array([node.coordinates[0] for node in rail_nodes])

        # get displacement at hinge
        vertical_displacement_at_hinge = vertical_displacements_rail[50,:]

        # calculate analytical solution max displacement middle node
        analytical_beam = BeamWithHinge(hinge_location, tot_length_beam-hinge_location, youngs_mod_beam*intertia_beam, y_load)
        max_displacements = np.array([analytical_beam.calculate_max_disp(coord) for coord in coords])

        # assert if numerical solution is equal to analytical solution
        np.testing.assert_allclose(vertical_displacement_at_hinge,max_displacements,atol=1e-7)


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