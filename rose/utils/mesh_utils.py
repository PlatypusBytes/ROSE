from rose.base.geometry import Node, Element, Mesh

from rose.track.track import Rail, RailPad, Sleeper
from rose.soil.soil import Soil
from rose.base.model_part import ConstraintModelPart, ElementModelPart
from rose.base.boundary_conditions import LineLoadCondition
import rose.utils.utils as utils

import itertools
import numpy as np

from scipy import sparse
from scipy.spatial.distance import cdist


from shapely.geometry import Point
from scipy.spatial.distance import cdist
from scipy.spatial import KDTree
from scipy import sparse
from typing import Union

INTERSECTION_TOLERANCE = 1e-6



def set_load_vector_as_function_of_time(time, load, build_up_idxs, load_locations, lower_limit_location: float, upper_limit_location):
    # set normal force location as a function of time
    moving_load = np.ones(len(time)) * load
    #
    moving_load[0:build_up_idxs] = np.linspace(
        0, load, build_up_idxs
    )
    moving_load, cum_distances_force = utils.filter_data_outside_range(
        moving_load,
        load_locations,
        lower_limit_location,
        upper_limit_location,
    )

    return moving_load, cum_distances_force


def get_coordinates_node_array(nodes):
    # get numpy array of nodal coordinates
    nodal_coordinates = np.array([node.coordinates for node in nodes])
    return nodal_coordinates


def calculate_cum_distances_coordinate_array(coordinates):

    # calculate cumulative distances between nodes and point load locations
    cum_distances_coordinates = np.append(
        0,
        np.cumsum(
            utils.distance_np(
                coordinates[:-1, :], coordinates[1:, :], axis=1
            )
        ),
    )
    return cum_distances_coordinates

# def get_active_elements():



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
    cum_distances_nodes = calculate_cum_distances_coordinate_array(nodal_coordinates)

    # calculate cumulative distance of the force location based on force velocity

    # calculate distance from force
    cum_distances_force = utils.calculate_cum_distance_from_velocity(time, velocities)

    # add distance force to first node in first active element
    cum_distances_force += utils.distance_np(
            start_coords, np.array(force.elements[element_idx].nodes[0].coordinates)
        )
    # ???
    cum_distances_force += force.nodes.index(force.elements[element_idx].nodes[0])

    # cum_distances_force = (
    #         np.append(
    #             0,
    #             np.cumsum((time[1:] - time[:-1]) * velocities[:-1])
    #         )
    #     + utils.distance_np(
    #         start_coords, np.array(force.elements[element_idx].nodes[0].coordinates)
    #     )
    #     + force.nodes.index(force.elements[element_idx].nodes[0])
    # )

    # get element idx where point load is located for each time step
    # set_active_elements
    # element_idxs = np.zeros(len(cum_distances_force))
    force.active_elements = np.zeros((len(force.elements), len(cum_distances_force)))
    i = element_idx
    for idx, distance in enumerate(cum_distances_force):
        if i < len(cum_distances_nodes) - 2:
            if distance > cum_distances_nodes[i + 1]:
                i += 1
        # force.active_elements[i, idx] = i
        force.active_elements[i, idx] = True

    element_idxs = force.active_elements.nonzero()[0]

    # force.active_elements = np.zeros(len(cum_distances_force))

    # set the load vector as a function of time
    #todo check what happens when more than 1 for type is added, e.g. both normal force and y force

    # set normal force vector as a function of time
    moving_normal_force = None
    if normal_force is not None:
        moving_normal_force, cum_distances_force = set_load_vector_as_function_of_time(
            time, normal_force, build_up_idxs, cum_distances_force, cum_distances_nodes[0], cum_distances_nodes[-1])

    # set y force vector as a function of time
    moving_y_force = None
    if y_load is not None:
        moving_y_force, cum_distances_force = set_load_vector_as_function_of_time(
            time, y_load, build_up_idxs, cum_distances_force, cum_distances_nodes[0], cum_distances_nodes[-1])

    # set z moment vector as a function of time
    moving_z_moment = None
    if z_moment is not None:
        moving_z_moment, cum_distances_force = set_load_vector_as_function_of_time(
            time, z_moment, build_up_idxs, cum_distances_force, cum_distances_nodes[0], cum_distances_nodes[-1])

    # interpolate force distances on nodes
    moving_coords = utils.interpolate_cumulative_distance_on_nodes(
        cum_distances_nodes, nodal_coordinates, cum_distances_force
    )

    # set load
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
    """
    Adds constraint boundary condition to the bottom nodes of a model part
    :param bottom_model_part:
    :return:
    """
    # index of vertical dimension; 1 for 2D; 2 for 3D
    vert_idx = 1  # 2d %todo, make this ndim dependent
    elements = bottom_model_part.elements

    # get bottom nodes of model part
    bot_nodes = [
        sorted(element.nodes, key=lambda node: node.coordinates[vert_idx])[0]
        for element in elements
    ]

    # set constraint condition
    no_disp_boundary_condition = ConstraintModelPart()
    no_disp_boundary_condition.nodes = bot_nodes

    return {"bottom_boundary": no_disp_boundary_condition}


def create_horizontal_track(n_sleepers, sleeper_distance, soil_depth):
    """
    Creates mesh of an horizontal track. Where the top of the track lies at z=0; the sleeper thickness is 0.1m.

    :param n_sleepers:
    :param sleeper_distance:
    :param soil_depth:
    :return:
    """
    # define constants
    track_level = 0.0
    sleeper_thickness = 0.1

    # initialise mesh
    mesh = Mesh()

    # add track nodes and elements to mesh
    nodes_track = [Node(i * sleeper_distance, track_level, 0.0) for i in range(n_sleepers)]
    mesh.add_unique_nodes_to_mesh(nodes_track)
    elements_track = [
        Element([nodes_track[i], nodes_track[i + 1]]) for i in range(n_sleepers - 1)
    ]
    mesh.add_unique_elements_to_mesh(elements_track)

    # add railpad nodes and elements to mesh
    points_rail_pad = [
        [nodes_track[i], Node(i * sleeper_distance, -sleeper_thickness, 0.0)]
        for i in range(n_sleepers)
    ]
    points_rail_pad = list(itertools.chain.from_iterable(points_rail_pad))
    mesh.add_unique_nodes_to_mesh(points_rail_pad)

    elements_rail_pad = [
        Element([points_rail_pad[i * 2], points_rail_pad[i * 2 + 1]])
        for i in range(n_sleepers)
    ]
    mesh.add_unique_elements_to_mesh(elements_rail_pad)

    # add sleeper nodes to mesh
    nodes_sleeper = [points_rail_pad[i * 2 + 1] for i in range(n_sleepers)]
    mesh.add_unique_nodes_to_mesh(nodes_sleeper)

    # add soil nodes and elements to mesh
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

    # add no displacement boundary condition to mesh
    no_disp_boundary_condition_nodes = [
        points_soil[i * 2 + 1] for i in range(n_sleepers)
    ]
    mesh.add_unique_nodes_to_mesh(no_disp_boundary_condition_nodes)

    # reorder node and element indices
    mesh.reorder_element_ids()
    mesh.reorder_node_ids()

    # initialise rail model part
    rail_model_part = Rail()
    rail_model_part.elements = elements_track
    rail_model_part.nodes = nodes_track
    rail_model_part.length_rail = sleeper_distance

    # initialise railpad model part
    rail_pad_model_part = RailPad()
    rail_pad_model_part.elements = elements_rail_pad
    rail_pad_model_part.nodes = points_rail_pad

    # initialise sleeper model part
    sleeper_model_part = Sleeper()
    sleeper_model_part.nodes = nodes_sleeper

    # initialise soil model part
    soil = Soil()
    soil.nodes = points_soil
    soil.elements = elements_soil

    # return dictionary of model parts present in horizontal track
    return (
        {
            "rail": rail_model_part,
            "rail_pad": rail_pad_model_part,
            "sleeper": sleeper_model_part,
            "soil": soil,
        },
        mesh,
    )
