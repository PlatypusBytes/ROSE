from src.geometry import *
from src.track import *
from src.soil import *

import matplotlib.pyplot as plt

from scipy import sparse
# global_system = GlobalSystem()

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



mesh.reorder_element_ids()
mesh.reorder_node_ids()



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


model_parts = [rail_model_part, rail_pad_model_part, sleeper_model_part, soil]
for model_part in model_parts:
    model_part.initialize()

    for node in model_part.nodes:
        node.rotation_dof = model_part.rotation_dof if model_part.rotation_dof else node.rotation_dof
        node.x_disp_dof = model_part.x_disp_dof if model_part.x_disp_dof else node.x_disp_dof
        node.y_disp_dof = model_part.y_disp_dof if model_part.y_disp_dof else node.y_disp_dof

ndof = 0
index_dof = 0
for node in mesh.nodes:
    node.index_dof[0] = index_dof
    index_dof += 1
    node.index_dof[1] = index_dof
    index_dof += 1
    node.index_dof[2] = index_dof
    index_dof += 1
    ndof = ndof + len(node.index_dof)


global_stiffness_matrix = sparse.csr_matrix((ndof, ndof))

for model_part in model_parts:
    if model_part.elements:
        n_nodes_element = len(model_part.elements[0].nodes)
    else:
        n_nodes_element = 1

    model_part.set_aux_stiffness_matrix()
    model_part.aux_stiffness_matrix = utils.reshape_aux_matrix(n_nodes_element, [model_part.x_disp_dof, model_part.y_disp_dof, model_part.rotation_dof],
                                    model_part.aux_stiffness_matrix)

    if model_part.elements:
        global_stiffness_matrix = utils.add_aux_matrix_to_global(
            global_stiffness_matrix, model_part.aux_stiffness_matrix, model_part.elements)
    else:
        global_stiffness_matrix = utils.add_aux_matrix_to_global(
            global_stiffness_matrix, model_part.aux_stiffness_matrix, model_part.elements, model_part.nodes)




plt.spy(global_stiffness_matrix)
plt.show()



#
#
# def set_up_material():
#     # Steel
#     material = Material()
#     material.youngs_modulus = 210E9 # Pa
#     material.poisson_ratio = 0.3
#     material.density = 7860
#     return material
#
#
#
# def set_up_section():
#     section = Section()
#     section.area = 69.682e-4
#     section.sec_moment_of_inertia = 2337.9e-8
#     section.shear_factor = 0
#     section.n_rail_per_sleeper = 2
#     return section
#
#
# def set_up_rail(set_up_material, set_up_section):
#     rail = Rail(3)
#     rail.section = set_up_section
#     rail.material = set_up_material
#
#     rail.calculate_length_rail(0.6)
#     rail.calculate_mass()
#     rail.calculate_n_dof()
#
#     rail.damping_ratio = 0.04
#     rail.radial_frequency_one = 2
#     rail.radial_frequency_two = 500
#     return rail
#
#
# def set_up_sleeper():
#     sleeper = Sleeper()
#     sleeper.mass = 162.5
#     sleeper.distance_between_sleepers = 0.6
#     return sleeper
#
# def set_up_rail_pad():
#     rail_pad = RailPad()
#     rail_pad.mass = 5
#     rail_pad.stiffness = 145e6
#     rail_pad.damping = 12e3
#     return  rail_pad
#
# def set_up_soil():
#     soil = Soil()
#     soil.stiffness = 300e6
#     soil.damping = 0
#     return soil