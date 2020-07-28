from src.geometry import *
from src.track import *
from src.soil import *
from src.boundary_conditions import NoDispRotCondition, CauchyCondition
from src.model_part import ConditionModelPart, ConstraintModelPart
from one_dimensional.solver import Solver, NewmarkSolver
from src.plot_utils import plot_2d_geometry
from src.mesh_utils import *

from src.global_system import GlobalSystem
import cProfile

# import src.global_system as gs

import matplotlib.pyplot as plt

from scipy import sparse

def main():
    time = np.linspace(0, 10, 1000)
    element_model_parts, mesh = create_horizontal_track(100, 2, 1)
    bottom_boundary = add_no_displacement_boundary_to_bottom(element_model_parts['soil'])
    load = add_moving_point_load_to_track(element_model_parts['rail'], time, 10, y_load=-15000)


    rail_model_part = element_model_parts['rail']
    rail_pad_model_part = element_model_parts['rail_pad']
    sleeper_model_part = element_model_parts['sleeper']
    soil = element_model_parts['soil']

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

    rail_model_part.section = section
    rail_model_part.material = material
    rail_model_part.damping_ratio = 0.04
    rail_model_part.radial_frequency_one = 2
    rail_model_part.radial_frequency_two = 500

    rail_pad_model_part.mass = 5
    rail_pad_model_part.stiffness = 145e6
    rail_pad_model_part.damping = 12e3

    sleeper_model_part.mass = 162.5
    sleeper_model_part.distance_between_sleepers = 0.6

    soil.stiffness = 300e6
    soil.damping = 0

    # set solver
    solver = NewmarkSolver()

    # populate global system
    global_system = GlobalSystem()
    global_system.mesh = mesh
    global_system.time = time

    model_parts = [list(element_model_parts.values()), list(bottom_boundary.values()), list(load.values())]
    # model_parts = list(itertools.chain.from_iterable(model_parts))
    global_system.model_parts = list(itertools.chain.from_iterable(model_parts))

    global_system.solver = solver

    # calculate
    global_system.main()

    # fig = plot_2d_geometry(mesh.elements)
    # fig.show()

    fig2 = plt.figure()

    plt.plot(global_system.time, global_system.displacements[:, 1], linestyle='-')
    plt.plot(global_system.time, global_system.displacements[:, 4], linestyle='--')
    plt.plot(global_system.time, global_system.displacements[:, 7], linestyle='-.')
    plt.plot(global_system.time, global_system.displacements[:, 9], linestyle=':')
    plt.plot(global_system.time, global_system.displacements[:, 10], marker='v')
    plt.plot(global_system.time, global_system.displacements[:, 11], marker='o')

    # plt.legend(["1","4","7","9","10","11"])
    plt.legend(["1", "2", "3", "4", "5", "6"])
    plt.xlabel('time')
    plt.ylabel('displacement')
    fig2.show()


cProfile.run('main()')

