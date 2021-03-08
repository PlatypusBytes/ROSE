# from src.track import *
from rose.base.model_part import Material, Section
from rose.solver.solver import NewmarkSolver
from rose.utils.mesh_utils import *
from rose.utils.plot_utils import *
from rose.base.global_system import GlobalSystem

import cProfile

import matplotlib.pyplot as plt


def main2():
    # calculate analytical solution
    stiffness_spring = 2.75e5
    y_load = -18e3

    distance_springs = 1
    n_sleepers = 50

    length_track = n_sleepers*distance_springs * 2

    velocity = 27.78

    # setup numerical model
    # set time in two stages
    initialisation_time = np.linspace(0, 0.01, 10)
    calculation_time = np.linspace(initialisation_time[-1], length_track/velocity, 500)
    time = np.concatenate((initialisation_time, calculation_time[1:]))

    # set geometry
    depth_soil = 0.9
    element_model_parts_1, mesh_1 = create_horizontal_track(
        n_sleepers, distance_springs, depth_soil
    )
    bottom_boundary_1 = add_no_displacement_boundary_to_bottom(
        element_model_parts_1["soil"]
    )

    element_model_parts_2, mesh_2 = create_horizontal_track(
        n_sleepers, distance_springs, depth_soil
    )
    bottom_boundary_2 = add_no_displacement_boundary_to_bottom(
        element_model_parts_2["soil"]
    )

    for node in mesh_2.nodes:
        node.coordinates[0] = node.coordinates[0] + (n_sleepers) * distance_springs

    all_mesh = Mesh()

    # fill model parts
    rail_model_part = element_model_parts_1["rail"]
    rail_pad_model_part = element_model_parts_1["rail_pad"]
    sleeper_model_part = element_model_parts_1["sleeper"]
    soil_1 = element_model_parts_1["soil"]


    rail_model_part.elements = rail_model_part.elements + \
                               [Element([element_model_parts_1["rail"].nodes[-1], element_model_parts_2["rail"].nodes[0]])] + \
                               element_model_parts_2["rail"].elements
    rail_model_part.nodes = rail_model_part.nodes + element_model_parts_2["rail"].nodes

    rail_pad_model_part.nodes = rail_pad_model_part.nodes + element_model_parts_2["rail_pad"].nodes
    rail_pad_model_part.elements = rail_pad_model_part.elements + element_model_parts_2["rail_pad"].elements

    sleeper_model_part.nodes = sleeper_model_part.nodes + element_model_parts_2["sleeper"].nodes
    sleeper_model_part.elements = sleeper_model_part.elements + element_model_parts_2["sleeper"].elements

    soil_2 = element_model_parts_2["soil"]

    all_mesh.add_unique_nodes_to_mesh(mesh_1.nodes)
    all_mesh.add_unique_nodes_to_mesh(mesh_2.nodes)

    all_mesh.add_unique_elements_to_mesh(mesh_1.elements)
    all_mesh.add_unique_elements_to_mesh(rail_model_part.elements)
    all_mesh.add_unique_elements_to_mesh(mesh_2.elements)

    all_mesh.reorder_node_ids()
    all_mesh.reorder_element_ids()

    # set elements
    material = Material()
    material.youngs_modulus = 2.059e11 # 4.41e05  # Pa
    material.poisson_ratio = 0.0
    material.density = 7860  # 7860

    section = Section()
    section.area = 0.0072
    section.sec_moment_of_inertia = 5.44893E-05
    section.shear_factor = 0

    rail_model_part.section = section
    rail_model_part.material = material
    rail_model_part.damping_ratio = 0.00
    rail_model_part.radial_frequency_one = 2
    rail_model_part.radial_frequency_two = 500

    rail_model_part.initialize()

    rail_pad_model_part.mass = 5 #0.000001  # 5
    rail_pad_model_part.stiffness = stiffness_spring / 0.1
    rail_pad_model_part.damping = 0  # 12e3

    sleeper_model_part.mass = 162.5 #0.0000001  # 162.5
    sleeper_model_part.distance_between_sleepers = distance_springs

    soil_1.stiffness = stiffness_spring / depth_soil  # 300e6
    soil_1.damping = 0# 12e3

    soil_2.stiffness = stiffness_spring / depth_soil * 10 # 300e6
    soil_2.damping = 0 #12e3

    # set load
    velocities = np.ones(len(time)) * velocity
    load = add_moving_point_load_to_track(
        rail_model_part,
        time,
        len(initialisation_time),
        velocities,
        y_load=y_load,
        start_coords=np.array([0, 0, 0]),
    )

    # set solver
    solver = NewmarkSolver()

    # populate global system
    global_system = GlobalSystem()
    global_system.mesh = all_mesh
    global_system.time = time
    global_system.solver = solver

    # get all element model parts from dictionary
    model_parts = [[rail_model_part, rail_pad_model_part, sleeper_model_part, soil_1, soil_2],
                   list(bottom_boundary_1.values()), list(bottom_boundary_2.values()), list(load.values())
                   ]
    global_system.model_parts = list(itertools.chain.from_iterable(model_parts))

    # calculate
    global_system.main()

    # get vertical displacement on rail model part
    coordinates_rail = np.array([node.coordinates[0] for node in global_system.model_parts[0].nodes])
    y_displacements_rail = np.array([node.displacements[:, 1] for node in global_system.model_parts[0].nodes])


    create_animation('../rose/test2.html', coordinates_rail, y_displacements_rail)


def main():
    # set initialisation time steps, i.e. the time in which the loads will be linearly increased from 0 to the prefered
    # values.
    initialisation_time = np.linspace(0, 0.1, 100)

    # set time steps of the main calculation
    calculation_time = np.linspace(initialisation_time[-1], 10, 5000)

    # combine all timesteps to np array
    time = np.concatenate((initialisation_time, calculation_time[1:]))

    # create mesh of a horizontal track with a rail, dampers, sleepers and soil
    element_model_parts, mesh = create_horizontal_track(100, 1.1, 0.9)

    # set bottom boundary of the soil elements as a no displacement boundary
    bottom_boundary = add_no_displacement_boundary_to_bottom(
        element_model_parts["soil"]
    )

    # get element model parts from mesh dictionary
    rail_model_part = element_model_parts["rail"]
    rail_pad_model_part = element_model_parts["rail_pad"]
    sleeper_model_part = element_model_parts["sleeper"]
    soil = element_model_parts["soil"]

    # set parameters of all element model parts
    # set rail material parameters
    material = Material()
    material.youngs_modulus = 441e3  # Pa
    material.poisson_ratio = 0.0
    material.density = 0.000001  # 7860

    # set rail section parameters
    section = Section()
    section.area = 1
    section.sec_moment_of_inertia = 1
    section.shear_factor = 0

    # set rail parameters
    rail_model_part.section = section
    rail_model_part.material = material
    rail_model_part.damping_ratio = 0.0000
    rail_model_part.radial_frequency_one = 2
    rail_model_part.radial_frequency_two = 500

    # set rail pad parameters
    rail_pad_model_part.mass = 0.000001  # 5
    rail_pad_model_part.stiffness = 275e4
    rail_pad_model_part.damping = 0  # 12e3

    # set sleeper parameters
    sleeper_model_part.mass = 0.0000001  # 162.5
    sleeper_model_part.distance_between_sleepers = 1.1

    # set soil parameters
    soil.stiffness = 275e3 / 0.9  # 300e6
    soil.damping = 0

    # Initialise rail model part; model part on which a load is location needs to be initialized before the load can be
    # added.
    rail_model_part.initialize()

    # set load velocity
    velocities = np.ones(len(time)) * 10

    # add moving load to rail model part
    load = add_moving_point_load_to_track(
        element_model_parts["rail"], time, len(initialisation_time), velocities, y_load=-18000
    )

    # set solver
    solver = NewmarkSolver()

    # populate global system
    global_system = GlobalSystem()
    global_system.mesh = mesh
    global_system.time = time

    model_parts = [
        list(element_model_parts.values()),
        list(bottom_boundary.values()),
        list(load.values()),
    ]

    # combine model parts from dictionary into list
    global_system.model_parts = list(itertools.chain.from_iterable(model_parts))

    # add solver
    global_system.solver = solver

    # calculate
    global_system.main()

    # fig = plot_2d_geometry(mesh.elements)



    plt.plot(global_system.time, global_system.displacements[:, 151], linestyle="-")

    # get vertical displacement on rail model part
    y_displacements_rail = np.array([node.displacements[:, 1] for node in global_system.model_parts[0].nodes])
    # create_animation('test.html',time, y_displacements_rail)

    plt.xlabel("time")
    plt.ylabel("displacement")
    plt.savefig("temp.pdf")


cProfile.run("main2()", filename=None, sort=2)
