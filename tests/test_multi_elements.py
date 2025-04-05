import pytest
import numpy as np
from rose.pre_process.mesh_utils import create_horizontal_track, combine_horizontal_tracks
from rose.model.geometry import Mesh, Node, Element

"""Test cases for the multiple elements per sleeper feature"""

def test_create_horizontal_track_with_multi_elements():
    """Test that create_horizontal_track properly creates multiple elements between sleepers"""
    # Test parameters
    n_sleepers = 5
    sleeper_distance = 0.6
    soil_depth = 1.0
    n_elements_per_sleeper = 3  # 3 elements per sleeper distance

    # Create track with multiple elements per sleeper
    element_model_parts, mesh = create_horizontal_track(
        n_sleepers, sleeper_distance, soil_depth, n_elements_per_sleeper
    )

    # Verify the rail model part
    rail_model_part = element_model_parts["rail"]

    # Check that length_element is correctly calculated
    assert pytest.approx(rail_model_part.length_element) == sleeper_distance / n_elements_per_sleeper

    # Check total number of rail nodes: should be n_sleepers * n_elements_per_sleeper + 1
    expected_nodes = n_sleepers * n_elements_per_sleeper + 1
    assert len(rail_model_part.nodes) == expected_nodes

    # Check total number of rail elements: should be n_sleepers * n_elements_per_sleeper
    expected_elements = n_sleepers * n_elements_per_sleeper
    assert len(rail_model_part.elements) == expected_elements

    # Check that sleeper nodes are still created only at sleeper positions
    assert len(element_model_parts["sleeper"].nodes) == n_sleepers

    # Check that rail pads connect to sleepers correctly
    assert len(element_model_parts["rail_pad"].elements) == n_sleepers

def test_uniform_element_length():
    """Test that all elements have the same length when using multiple elements per sleeper"""
    # Test parameters
    n_sleepers = 5
    sleeper_distance = 0.6
    soil_depth = 1.0
    n_elements_per_sleeper = 4  # 4 elements per sleeper distance

    # Create track with multiple elements per sleeper
    element_model_parts, mesh = create_horizontal_track(
        n_sleepers, sleeper_distance, soil_depth, n_elements_per_sleeper
    )

    # Get the rail model part
    rail_model_part = element_model_parts["rail"]

    # Calculate the lengths of all elements
    element_lengths = []
    for element in rail_model_part.elements:
        node1 = element.nodes[0]
        node2 = element.nodes[1]
        dx = node2.coordinates[0] - node1.coordinates[0]
        dy = node2.coordinates[1] - node1.coordinates[1]
        dz = node2.coordinates[2] - node1.coordinates[2]
        length = np.sqrt(dx**2 + dy**2 + dz**2)
        element_lengths.append(length)

    # Check that all element lengths are the same (within floating point precision)
    expected_length = sleeper_distance / n_elements_per_sleeper
    for length in element_lengths:
        assert pytest.approx(length, abs=1e-10) == expected_length

def test_combine_tracks_with_multi_elements():
    """Test that tracks with multiple elements per sleeper can be combined correctly"""
    # Create two track segments with multiple elements per sleeper
    n_sleepers = 3
    sleeper_distance = 0.6
    soil_depth = 1.0
    n_elements_per_sleeper = 2

    # Create first track segment
    element_model_parts1, mesh1 = create_horizontal_track(
        n_sleepers, sleeper_distance, soil_depth, n_elements_per_sleeper
    )

    # Create second track segment
    element_model_parts2, mesh2 = create_horizontal_track(
        n_sleepers, sleeper_distance, soil_depth, n_elements_per_sleeper
    )

    # Combine the track segments
    rail_model_part, sleeper_model_part, rail_pad_model_part, soil_model_parts, all_mesh = \
        combine_horizontal_tracks([element_model_parts1, element_model_parts2], [mesh1, mesh2], sleeper_distance)

    # In the combine_horizontal_tracks function, nodes from both tracks are preserved
    # Each track has n_sleepers * n_elements_per_sleeper + 1 nodes
    # So the combined track will have the sum of both track's nodes
    expected_nodes = 2 * (n_sleepers * n_elements_per_sleeper + 1)

    # Elements: n_sleepers * n_elements_per_sleeper per track plus one connecting element
    expected_elements = 2 * n_sleepers * n_elements_per_sleeper + 1

    # Verify the combined model
    assert len(rail_model_part.nodes) == expected_nodes
    assert len(rail_model_part.elements) == expected_elements

    # Calculate the lengths of all elements
    element_lengths = []
    for element in rail_model_part.elements:
        node1 = element.nodes[0]
        node2 = element.nodes[1]
        dx = node2.coordinates[0] - node1.coordinates[0]
        dy = node2.coordinates[1] - node1.coordinates[1]
        dz = node2.coordinates[2] - node1.coordinates[2]
        length = np.sqrt(dx**2 + dy**2 + dz**2)
        element_lengths.append(length)

    # Check that all element lengths are the same (within floating point precision)
    expected_length = sleeper_distance / n_elements_per_sleeper
    for length in element_lengths:
        assert pytest.approx(length, abs=1e-10) == expected_length