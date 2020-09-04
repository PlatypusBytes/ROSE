from src.geometry import *
# from src.track import *
from src.soil import *
from src.boundary_conditions import NoDispRotCondition, LoadCondition
from src.model_part import ConditionModelPart, ConstraintModelPart, Material, Section
from one_dimensional.solver import Solver, NewmarkSolver
from src.plot_utils import plot_2d_geometry, create_animation
from src.mesh_utils import *

from src.global_system import GlobalSystem
import cProfile

import matplotlib.pyplot as plt


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


cProfile.run("main()", filename=None, sort=2)
