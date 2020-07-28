from src.geometry import Node, Element, Mesh

from src.track import Rail, RailPad, Sleeper
from src.soil import Soil
from src.model_part import ConstraintModelPart
from src.boundary_conditions import CauchyCondition
import itertools
import numpy as np
from scipy import interpolate
from scipy import sparse


def add_moving_point_load_to_track(rail_model_part, time, build_up_idxs, moment=None,x_load=None, y_load=None):

    ndim = 2
    rotation_dof = moment is not None
    x_disp_dof = x_load is not None
    y_disp_dof = y_load is not None

    force = CauchyCondition(rotation_dof=rotation_dof, x_disp_dof=x_disp_dof, y_disp_dof=y_disp_dof)
    force.nodes = rail_model_part.nodes
    force.elements = rail_model_part.elements

    # todo xload and moment

    moving_y_force = np.ones(len(time)) * y_load
    moving_y_force[0:build_up_idxs] = np.linspace(0, y_load, build_up_idxs)

    #2d
    if ndim == 2:
        force.y_force = sparse.csr_matrix((len(force.nodes), len(time)))
        f = interpolate.interp1d([node.coordinates[0] for node in force.nodes],
                                 [node.coordinates[1] for node in force.nodes])
        moving_x_coords = np.linspace(force.nodes[0].coordinates[0], force.nodes[-1].coordinates[0], len(time))
        moving_y_coords = f(moving_x_coords)
        moving_coords = np.array([Node(moving_x_coord, moving_y_coords[idx], 0.0) for idx, moving_x_coord in enumerate(moving_x_coords)])

        force.set_moving_point_load(moving_coords, time, y_force=moving_y_force)



    return {'moving_load': force}

def add_no_displacement_boundary_to_bottom(bottom_model_part):

    vert_idx = 1 # 2d %todo, make this ndim dependent
    elements = bottom_model_part.elements
    bot_nodes = [sorted(element.nodes, key=lambda x: x.coordinates[vert_idx])[0] for element in elements]

    no_disp_boundary_condition = ConstraintModelPart()
    no_disp_boundary_condition.nodes = bot_nodes

    return {'bottom_boundary': no_disp_boundary_condition}


def create_horizontal_track(n_sleepers, sleeper_distance, soil_depth):
    mesh = Mesh()

    # add track
    nodes_track =[Node(i*sleeper_distance, 0.0, 0.0) for i in range(n_sleepers)]
    mesh.add_unique_nodes_to_mesh(nodes_track)
    elements_track = [Element([nodes_track[i], nodes_track[i+1]]) for i in range(n_sleepers-1)]
    mesh.add_unique_elements_to_mesh(elements_track)

    # add railpad
    points_rail_pad = [[nodes_track[i], Node(i*sleeper_distance, -0.1, 0.0)] for i in range(n_sleepers)]
    points_rail_pad = list(itertools.chain.from_iterable(points_rail_pad))
    mesh.add_unique_nodes_to_mesh(points_rail_pad)

    elements_rail_pad = [Element([points_rail_pad[i*2], points_rail_pad[i*2+1]]) for i in range(n_sleepers)]
    mesh.add_unique_elements_to_mesh(elements_rail_pad)

    # add sleeper
    nodes_sleeper = [points_rail_pad[i*2+1] for i in range(n_sleepers)]
    mesh.add_unique_nodes_to_mesh(nodes_sleeper)

    # add soil
    points_soil = [[points_rail_pad[i*2+1], Node(points_rail_pad[i*2+1].coordinates[0], points_rail_pad[i*2+1].coordinates[1] - soil_depth, 0.0)]
                   for i in range(n_sleepers)]
    points_soil = list(itertools.chain.from_iterable(points_soil))
    mesh.add_unique_nodes_to_mesh(points_soil)

    elements_soil = [Element([points_soil[i*2], points_soil[i*2+1]]) for i in range(n_sleepers)]
    mesh.add_unique_elements_to_mesh(elements_soil)

    # add no displacement boundary condition
    no_disp_boundary_condition_nodes = [points_soil[i*2+1] for i in range(n_sleepers)]
    mesh.add_unique_nodes_to_mesh(no_disp_boundary_condition_nodes)

    mesh.reorder_element_ids()
    mesh.reorder_node_ids()

    rail_model_part = Rail()
    rail_model_part.elements = elements_track
    rail_model_part.nodes = nodes_track
    rail_model_part.length_rail = sleeper_distance

    rail_pad_model_part = RailPad()
    rail_pad_model_part.elements = elements_rail_pad
    rail_pad_model_part.nodes = points_rail_pad

    sleeper_model_part = Sleeper()
    sleeper_model_part.nodes = nodes_sleeper

    soil = Soil()
    soil.nodes = points_soil
    soil.elements = elements_soil


    return {'rail':rail_model_part,
            'rail_pad': rail_pad_model_part,
            'sleeper': sleeper_model_part,
            'soil': soil}, mesh
    # points_rail_pad = [nodes_track[0], Node(0.0, -0.1, 0.0), nodes_track[1], Node(1.0, -0.1, 0.0),
    #                    nodes_track[2], Node(2.0, -0.1, 0.0)]

    #
    # elements_rail_pad = [Element([points_rail_pad[0], points_rail_pad[1]]),
    #                      Element([points_rail_pad[2], points_rail_pad[3]]),
    #                      Element([points_rail_pad[4], points_rail_pad[5]])]
    # mesh.add_unique_elements_to_mesh(elements_rail_pad)
