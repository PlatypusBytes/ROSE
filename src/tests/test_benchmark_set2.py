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


class TestBenchmarkSet2:
    """
    In Benchmark set 2, benchmarks are verified with plots of the analytical solution, values are tested regressive
    """


    @pytest.mark.workinprogress
    def test_dynamic_load_on_rod(self):
        length_rod = 0.01
        n_beams = 101

        # Setup parameters euler beam
        # time = np.linspace(0, 10, 1001)
        E = 20e5
        I = 1
        A = 1
        L = (n_beams - 1) * length_rod
        F = -1000
        rho = 3

        # set and calculated analytical solution
        # rod_analytical = OneDimWavePropagation(nb_cycles=10,nb_terms=100)
        # rod_analytical.properties(rho, E, F, L, n_beams)
        # rod_analytical.solution()

        # res = {"time": rod_analytical.time.tolist(),
        #        "v": rod_analytical.v.tolist(),
        #        "u": rod_analytical.u.tolist()}
        #
        # with open("./rod.json", "w") as f:
        #     json.dump(res, f, indent=2)

        with open('test_data/rod.json') as rod_file:
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
        # rod.stiffness = E *100
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

        hor_displacements = np.array([node.displacements[:, 0] for node in rod_nodes])
        hor_velocities = np.array([node.velocities[:, 0] for node in rod_nodes])

        plt.plot(time, hor_velocities[-1, :], marker='o')
        plt.plot(time, np.array(rod_analytical['v'])[-1, :], marker='x')

        plt.show()

        # assert displacement on each node
        # np.testing.assert_allclose(beam_analytical.u, hor_displacements[:, -1])

        @pytest.mark.workinprogress
        def test_dynamic_normal_load_on_beam(self):
            length_rod = 0.01
            n_beams = 101

            # Setup parameters euler beam
            # time = np.linspace(0, 10, 1001)
            E = 20e5
            I = 1
            A = 1
            L = (n_beams - 1) * length_rod
            F = -1000
            rho = 3

            with open('test_data/rod.json') as rod_file:
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

            plt.plot(time, hor_velocities[-1, :], marker='o')
            plt.plot(time, np.array(rod_analytical['v'])[-1, :], marker='x')

            plt.show()