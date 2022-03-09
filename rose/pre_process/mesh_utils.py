from rose.model.geometry import Node, Element, Mesh
from rose.model.track import Rail, RailPad, Sleeper
from rose.model.soil import Soil
from rose.model.model_part import ConstraintModelPart, ElementModelPart
# from rose.base.boundary_conditions import LineLoadCondition, MovingPointLoad
import rose.model.utils as utils

from copy import deepcopy
import itertools
import numpy as np

from typing import Dict, List

INTERSECTION_TOLERANCE = 1e-6



def set_load_vector_as_function_of_time(time, load, build_up_idxs, load_locations, lower_limit_location: float, upper_limit_location):


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


def add_no_displacement_boundary_to_bottom(bottom_model_part: ElementModelPart):
    """
    Adds constraint boundary condition to the bottom nodes of a model part
    :param bottom_model_part: element model part where the constrain boundary condition is applied to.
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


def add_semi_rigid_hinge_at_x(rail_model_part, x_coordinate_hinge, hinge_stiffness, mesh):
    """
    Adds a semi rigid hinge to the rail at a certain x coordinate. As a result, the rail model part is split in 4 parts.
    The elements before the hinge, the two elements connected to the hinge, and the elements after the hinge.

    :param rail_model_part: original rail model part
    :param x_coordinate_hinge: x coordinate of the hinge
    :param hinge_stiffness: stiffness of the hinge
    :param mesh: sorted mesh of system
    :return:
    """

    # check if mesh is valid. Node and Element index should be equal to the index is the respective list
    for idx, element in enumerate(mesh.elements):
        assert element.index == idx
    for idx, node in enumerate(mesh.nodes):
        assert node.index == idx

    # find node which is located at the x coord.
    hinge_node = None
    removed_node_idx = None
    for idx, node in enumerate(rail_model_part.nodes):
        if np.isclose(node.coordinates[0], x_coordinate_hinge):
            removed_node_idx = idx
            hinge_node = node
            rail_model_part.nodes.remove(node)
            break
    if hinge_node is None:
        raise Exception(f"node at x-coord {x_coordinate_hinge} is not found")

    # find elements which use the hinge_node
    # reverse iterate such that removing elements is done correctly
    hinge_elements = []
    hinge_rail_model_parts = []
    removed_el_idx = None
    for idx, element in enumerate(reversed(rail_model_part.elements)):

        # element is hinge element if it is connected to the hinge node
        if hinge_node in element.nodes:
            removed_el_idx = idx
            hinge_elements.append(element)
            rail_model_part.elements.remove(element)

            # create new rail model part for the hinge and set element. nodes and length rail
            hinge_rail_model_part = Rail()
            hinge_rail_model_part.elements = [element]
            hinge_rail_model_part.nodes = element.nodes
            hinge_rail_model_part.length_rail = rail_model_part.length_rail

            # add hinge stiffness to hinge model part
            if hinge_node == element.nodes[0]:
                hinge_rail_model_part.spring_stiffness1 = hinge_stiffness
            if hinge_node == element.nodes[1]:
                hinge_rail_model_part.spring_stiffness2 = hinge_stiffness

            hinge_rail_model_parts.append(hinge_rail_model_part)

    # reverse list such that the elements are in order of x-coordinate
    hinge_rail_model_parts.reverse()

    # reverse removed element index, since element iteration was done in reverse
    removed_el_idx = len(rail_model_part.elements) - removed_el_idx + 1

    # copy and split rail model parts at hinge
    rail_part_1 = deepcopy(rail_model_part)
    rail_part_2 = deepcopy(rail_model_part)

    rail_part_1.elements = [rail_model_part.elements[i] for i in range(removed_el_idx)]
    rail_part_1.nodes = [rail_model_part.nodes[i] for i in range(removed_node_idx)]

    rail_part_2.elements = [rail_model_part.elements[i] for i in range(removed_el_idx,len(rail_model_part.elements))]
    rail_part_2.nodes = [rail_model_part.nodes[i] for i in range(removed_node_idx,len(rail_model_part.nodes))]

    # all_rail_model_parts = [rail_part_1] + hinge_rail_model_parts + [rail_part_2]
    #
    # # # correct pointers within mesh
    # # for model_part in all_rail_model_parts:
    # #     for element in model_part.elements:
    # #         mesh.elements[element.index] = element
    # #     for node in model_part.nodes:
    # #         mesh.nodes[node.index] = node

    # return list of spatially sorted rail model parts and updated mesh
    return [rail_part_1] + hinge_rail_model_parts + [rail_part_2], mesh

def create_horizontal_track(n_sleepers, sleeper_distance, soil_depth):
    """
    Creates mesh of an horizontal track. Where the top of the track lies at z=0; the sleeper thickness is 1.0m.

    :param n_sleepers: number of sleepers [-]
    :param sleeper_distance: distance between sleepers [m]
    :param soil_depth: depth of the soil [m]
    :return: Dictionary with: rail model part, rail pad model part, sleeper model part, soil model part. Mesh

    """
    # define constants
    track_level = 0.0
    sleeper_thickness = 1.0

    n_rail_per_sleeper = 1

    # initialise mesh
    mesh = Mesh()

    # add track nodes and elements to mesh
    nodes_track = [Node(i * sleeper_distance/n_rail_per_sleeper, track_level, 0.0) for i in range(n_sleepers * n_rail_per_sleeper)]
    mesh.add_unique_nodes_to_mesh(nodes_track)
    elements_track = [
        Element([nodes_track[i], nodes_track[i + 1]]) for i in range(n_sleepers * n_rail_per_sleeper - 1)
    ]
    mesh.add_unique_elements_to_mesh(elements_track)

    # add railpad nodes and elements to mesh
    points_rail_pad = [
        [nodes_track[i*n_rail_per_sleeper], Node(i * sleeper_distance, -sleeper_thickness, 0.0)]
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


def combine_horizontal_tracks(tracks: List[Dict[str, ElementModelPart]], meshes: List[Mesh]):
    """
    Combines multiple horizontal track parts. This function takes a list of tracks and connects the track parts to a
    model. The tracks in the list need to be ordered from left to right. Coordinates of the track parts
    are automatically recalculated.

    This function assumes horizontal tracks over the x-axis

    :param tracks: Ordered list of dictionaries of model parts belonging to a horizontal track part
    :param meshes: list of meshes belonging to the track parts
    :return:
    """

    global_mesh = Mesh()

    # loop over each mesh
    for idx, mesh in enumerate(meshes):
        if idx>0:
            # move coordinates to the right
            last_node = max(meshes[idx - 1].nodes, key=lambda item: item.coordinates[0])
            dx = utils.distance_np(np.array(meshes[idx - 1].nodes[-1].coordinates),
                                   np.array(meshes[idx - 1].nodes[-2].coordinates))

            for node in mesh.nodes:
                node.coordinates[0] += last_node.coordinates[0] + dx

        # add mesh to global mesh
        global_mesh.add_unique_nodes_to_mesh(mesh.nodes)
        global_mesh.add_unique_elements_to_mesh(mesh.elements)

    soil_model_parts = []
    for idx, track in enumerate(tracks):
        # add soil model parts to list
        soil_model_parts.append(track["soil"])
        if idx>0:
            # combine rail elements in one model part
            connecting_element = Element([tracks[idx-1]["rail"].nodes[-1],
                                          track["rail"].nodes[0]])

            track["rail"].elements = tracks[idx-1]["rail"].elements + [connecting_element] + track["rail"].elements
            track["rail"].nodes = tracks[idx-1]["rail"].nodes + track["rail"].nodes
            global_mesh.add_unique_elements_to_mesh([connecting_element])

            # combine sleeper elements in one model part
            track["sleeper"].nodes = tracks[idx-1]["sleeper"].nodes + track["sleeper"].nodes
            track["sleeper"].elements = tracks[idx-1]["sleeper"].elements + track["sleeper"].elements

            # combine sleeper elements in one model part
            track["rail_pad"].nodes = tracks[idx - 1]["rail_pad"].nodes + track["rail_pad"].nodes
            track["rail_pad"].elements = tracks[idx - 1]["rail_pad"].elements + track["rail_pad"].elements

    # get combined model parts
    rail_model_part = tracks[-1]["rail"]
    sleeper_model_part = tracks[-1]["sleeper"]
    rail_pad_model_part = tracks[-1]["rail_pad"]

    # reorder node and element ids
    global_mesh.reorder_node_ids()
    global_mesh.reorder_element_ids()

    return rail_model_part, sleeper_model_part, rail_pad_model_part, soil_model_parts, global_mesh


if __name__ == "__main__":
    tracks=[]
    mesh1= Mesh()
    mesh1.nodes =[Node(0,0,0),Node(1,0,0) ,Node(2,0,0) ]

    mesh2 = Mesh()
    mesh2.nodes = [Node(0, 0, 0), Node(1, 0, 0), Node(2, 0, 0)]

    combine_horizontal_tracks(tracks, [mesh1, mesh2])

