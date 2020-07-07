from src.geometry import *
from src.track import *
from src.soil import *
from src.boundary_conditions import NoDispRotCondition, CauchyCondition
from src.model_part import ConditionModelPart, ConstraintModelPart
from one_dimensional.solver import Solver, NewmarkSolver

from src.global_system import GlobalSystem

# import src.global_system as gs

import matplotlib.pyplot as plt

from scipy import sparse

# set mesh
mesh = Mesh()
nodes_track = [Node(0.0, 0.0, 0.0), Node(1.0, 0.0, 0.0), Node(2.0, 0.0, 0.0)]
mesh.add_unique_nodes_to_mesh(nodes_track)

elements_track = [Element([nodes_track[0], nodes_track[1]]), Element([nodes_track[1], nodes_track[2]])]
mesh.add_unique_elements_to_mesh(elements_track)


points_rail_pad = [nodes_track[0], Node(0.0, -0.1, 0.0), nodes_track[1], Node(1.0, -0.1, 0.0),
                   nodes_track[2], Node(2.0, -0.1, 0.0)]
mesh.add_unique_nodes_to_mesh(points_rail_pad)


elements_rail_pad = [Element([points_rail_pad[0], points_rail_pad[1]]), Element([points_rail_pad[2], points_rail_pad[3]]),
                 Element([points_rail_pad[4], points_rail_pad[5]])]
mesh.add_unique_elements_to_mesh(elements_track)


nodes_sleeper = [points_rail_pad[1], points_rail_pad[3], points_rail_pad[5]]
mesh.add_unique_nodes_to_mesh(nodes_sleeper)


points_soil = [points_rail_pad[1], Node(0.0, -1, 0.0), points_rail_pad[3], Node(1.0, -1, 0.0),
                   points_rail_pad[5], Node(2.0, -1, 0.0)]
mesh.add_unique_nodes_to_mesh(points_soil)


elements_soil = [Element([points_soil[0], points_soil[1]]), Element([points_soil[2], points_soil[3]]),
                 Element([points_soil[4], points_soil[5]])]
mesh.add_unique_elements_to_mesh(elements_soil)


no_disp_boundary_condition_nodes = [points_soil[1], points_soil[3], points_soil[5]]
mesh.add_unique_nodes_to_mesh(no_disp_boundary_condition_nodes)


force_nodes = [nodes_track[1]]
mesh.add_unique_nodes_to_mesh(no_disp_boundary_condition_nodes)


mesh.reorder_element_ids()
mesh.reorder_node_ids()


# set elements
material = Material()
material.youngs_modulus = 210E9  # Pa
material.poisson_ratio = 0.3
material.density = 7860

section = Section()
section.area = 69.682e-4
section.sec_moment_of_inertia = 2337.9e-8
section.shear_factor = 0
section.n_rail_per_sleeper = 2

rail_model_part = Rail(3)
rail_model_part.elements = elements_track
rail_model_part.nodes = nodes_track

rail_model_part.section = section
rail_model_part.material = material

rail_model_part.calculate_length_rail(0.6)
rail_model_part.calculate_mass()
rail_model_part.calculate_n_dof()

rail_model_part.damping_ratio = 0.04
rail_model_part.radial_frequency_one = 2
rail_model_part.radial_frequency_two = 500

rail_pad_model_part = RailPad()
rail_pad_model_part.elements = elements_rail_pad
rail_pad_model_part.nodes = points_rail_pad

rail_pad_model_part.mass = 5
rail_pad_model_part.stiffness = 145e6
rail_pad_model_part.damping = 12e3

sleeper_model_part = Sleeper()
sleeper_model_part.nodes = nodes_sleeper

sleeper_model_part.mass = 162.5
sleeper_model_part.distance_between_sleepers = 0.6

soil = Soil()
soil.stiffness = 300e6
soil.damping = 0
soil.nodes = points_soil
soil.elements = elements_soil


# set time
time = np.linspace(0, 10, 1000)

# set conditions
no_disp_boundary_condition = ConstraintModelPart()
no_disp_boundary_condition.nodes = no_disp_boundary_condition_nodes

force = CauchyCondition(y_disp_dof=True)
force.nodes = force_nodes

force.y_force = sparse.csr_matrix((1, len(time)))
frequency = 1
force.y_force[0, :] = np.sin(frequency * time) * 15000

# set solver
solver = NewmarkSolver()

# populate global system
global_system = GlobalSystem()
global_system.mesh = mesh
global_system.time = time

global_system.model_parts = [rail_model_part, rail_pad_model_part, sleeper_model_part, soil,
                             no_disp_boundary_condition, force]

global_system.solver = solver

# calculate
global_system.main()


plt.plot(global_system.solver.time, solver.u[:, 1])
plt.plot(global_system.solver.time, solver.u[:, 4])
plt.plot(global_system.solver.time, solver.u[:, 7])
plt.plot(global_system.solver.time, solver.u[:, 9])
plt.plot(global_system.solver.time, solver.u[:, 10])
plt.plot(global_system.solver.time, solver.u[:, 11])

plt.legend(["1","4","7","9","10","11"])
plt.show()