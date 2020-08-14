from src.geometry import Node, Element, Mesh

from src.track import Rail, RailPad, Sleeper
from src.soil import Soil
from src.model_part import ConstraintModelPart, ElementModelPart
from src.boundary_conditions import LineLoadCondition
import itertools
import numpy as np
from scipy import interpolate
from scipy import sparse
from scipy.spatial.distance import cdist

import src.utils as utils
from shapely.geometry import Point
from scipy.spatial.distance import cdist
from scipy.spatial import KDTree
from scipy import sparse
from typing import Union

INTERSECTION_TOLERANCE = 1e-6


def filter_data_outside_range(
    data: np.array, locations: np.array, lower_limit: float, upper_limit: float
):
    # remove force if it is located outside of the mesh
    data[locations > upper_limit] = 0
    data[locations < lower_limit] = 0
    locations[locations > upper_limit] = upper_limit
    locations[locations < lower_limit] = lower_limit

    return data, locations


def interpolate_cumulative_distance_on_nodes(
    cum_distances_nodes, nodal_coordinates, cum_distances_force
):

    fx = interpolate.interp1d(cum_distances_nodes, nodal_coordinates[:, 0])
    fy = interpolate.interp1d(cum_distances_nodes, nodal_coordinates[:, 1])
    fz = interpolate.interp1d(cum_distances_nodes, nodal_coordinates[:, 2])

    moving_x_coords = fx(cum_distances_force)
    moving_y_coords = fy(cum_distances_force)
    moving_z_coords = fz(cum_distances_force)

    return np.array([moving_x_coords, moving_y_coords, moving_z_coords]).transpose()


def add_moving_point_load_to_track(
    rail_model_part: ElementModelPart,
    time: np.array,
    build_up_idxs: int,
    velocities: np.array,
    normal_force: float = None,
    z_moment: float = None,
    y_load: float = None,
    start_coords: np.array = None,
):
    """
    Adds a moving point load to line elements, where the line elements and nodes are required to be ordered based on
    occurrence.

    :param rail_model_part:
    :param time:
    :param build_up_idxs:
    :param velocities:
    :param normal_force:
    :param z_moment:
    :param y_load:
    :param start_coords:
    :return:
    """

    # set line load condition
    normal_dof = rail_model_part.normal_dof
    z_rot_dof = rail_model_part.z_rot_dof
    y_disp_dof = rail_model_part.y_disp_dof

    force = LineLoadCondition(
        normal_dof=normal_dof, z_rot_dof=z_rot_dof, y_disp_dof=y_disp_dof
    )
    force.nodes = rail_model_part.nodes
    force.elements = rail_model_part.elements
    force.time = time
    force.initialize_matrices()

    # if start coords are not given, set the first node as start coordinates
    if start_coords is None:
        start_coords = np.array(force.nodes[0].coordinates)

    # get numpy array of nodal coordinates
    nodal_coordinates = np.array([node.coordinates for node in force.nodes])

    # find element in which the start coordinates are located
    element_idx = utils.find_intersecting_point_element(force.elements, start_coords)

    # calculate cumulative distances between nodes and point load locations
    cum_distances_nodes = np.append(
        0,
        np.cumsum(
            utils.distance_np(
                nodal_coordinates[:-1, :], nodal_coordinates[1:, :], axis=1
            )
        ),
    )

    # calculate cumulative distance of the force location based on force velocity
    cum_distances_force = (
        velocities * (time - time[0])
        + utils.distance_np(
            start_coords, np.array(force.elements[element_idx].nodes[0].coordinates)
        )
        + force.nodes.index(force.elements[element_idx].nodes[0])
    )

    # get element idx where point load is located for each time step
    element_idxs = np.zeros(len(cum_distances_force))
    i = element_idx
    for idx, distance in enumerate(cum_distances_force):
        if i < len(cum_distances_nodes) - 2:
            if distance > cum_distances_nodes[i + 1]:
                i += 1
        element_idxs[idx] = i

    # set the load vector as a function of time
    moving_normal_force = None
    if normal_force is not None:
        moving_normal_force = np.ones(len(time)) * normal_force
        moving_normal_force[0:build_up_idxs] = np.linspace(
            0, normal_force, build_up_idxs
        )
        moving_normal_force, cum_distances_force = filter_data_outside_range(
            moving_normal_force,
            cum_distances_force,
            cum_distances_nodes[0],
            cum_distances_nodes[-1],
        )

    moving_y_force = None
    if y_load is not None:
        moving_y_force = np.ones(len(time)) * y_load
        moving_y_force[0:build_up_idxs] = np.linspace(0, y_load, build_up_idxs)
        moving_y_force, cum_distances_force = filter_data_outside_range(
            moving_y_force,
            cum_distances_force,
            cum_distances_nodes[0],
            cum_distances_nodes[-1],
        )
    moving_z_moment = None
    if z_moment is not None:
        moving_z_moment = np.ones(len(time)) * z_moment
        moving_z_moment[0:build_up_idxs] = np.linspace(0, z_moment, build_up_idxs)
        moving_z_moment, cum_distances_force = filter_data_outside_range(
            moving_z_moment,
            cum_distances_force,
            cum_distances_nodes[0],
            cum_distances_nodes[-1],
        )

    moving_coords = interpolate_cumulative_distance_on_nodes(
        cum_distances_nodes, nodal_coordinates, cum_distances_force
    )

    force.set_moving_point_load(
        rail_model_part,
        moving_coords,
        time,
        element_idxs=element_idxs,
        normal_force=moving_normal_force,
        y_force=moving_y_force,
        z_moment=moving_z_moment,
    )

    return {"moving_load": force}


def add_no_displacement_boundary_to_bottom(bottom_model_part):
    vert_idx = 1  # 2d %todo, make this ndim dependent
    elements = bottom_model_part.elements
    bot_nodes = [
        sorted(element.nodes, key=lambda x: x.coordinates[vert_idx])[0]
        for element in elements
    ]

    no_disp_boundary_condition = ConstraintModelPart()
    no_disp_boundary_condition.nodes = bot_nodes

    return {"bottom_boundary": no_disp_boundary_condition}


def create_horizontal_track(n_sleepers, sleeper_distance, soil_depth):
    mesh = Mesh()

    # add track
    nodes_track = [Node(i * sleeper_distance, 0.0, 0.0) for i in range(n_sleepers)]
    mesh.add_unique_nodes_to_mesh(nodes_track)
    elements_track = [
        Element([nodes_track[i], nodes_track[i + 1]]) for i in range(n_sleepers - 1)
    ]
    mesh.add_unique_elements_to_mesh(elements_track)

    # add railpad
    points_rail_pad = [
        [nodes_track[i], Node(i * sleeper_distance, -0.1, 0.0)]
        for i in range(n_sleepers)
    ]
    points_rail_pad = list(itertools.chain.from_iterable(points_rail_pad))
    mesh.add_unique_nodes_to_mesh(points_rail_pad)

    elements_rail_pad = [
        Element([points_rail_pad[i * 2], points_rail_pad[i * 2 + 1]])
        for i in range(n_sleepers)
    ]
    mesh.add_unique_elements_to_mesh(elements_rail_pad)

    # add sleeper
    nodes_sleeper = [points_rail_pad[i * 2 + 1] for i in range(n_sleepers)]
    mesh.add_unique_nodes_to_mesh(nodes_sleeper)

    # add soil
    points_soil = [
        [
            points_rail_pad[i * 2 + 1],
            Node(
                points_rail_pad[i * 2 + 1].coordinates[0],
                points_rail_pad[i * 2 + 1].coordinates[1] - soil_depth,
                0.0,
            ),
        ]
        for i in range(n_sleepers)
    ]
    points_soil = list(itertools.chain.from_iterable(points_soil))
    mesh.add_unique_nodes_to_mesh(points_soil)

    elements_soil = [
        Element([points_soil[i * 2], points_soil[i * 2 + 1]]) for i in range(n_sleepers)
    ]
    mesh.add_unique_elements_to_mesh(elements_soil)

    # add no displacement boundary condition
    no_disp_boundary_condition_nodes = [
        points_soil[i * 2 + 1] for i in range(n_sleepers)
    ]
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

    return (
        {
            "rail": rail_model_part,
            "rail_pad": rail_pad_model_part,
            "sleeper": sleeper_model_part,
            "soil": soil,
        },
        mesh,
    )
