import os
import pickle
# import ROSE packages
from run_rose.read_wolf import read_wolf
from rose.model.model_part import Material, Section
from rose.model.train_model import *
from rose.model.train_track_interaction import *
# import rose.model.solver as solver_c

from solvers.newmark_solver import NewmarkSolver

def assign_data_to_coupled_model(train_info, track_info, time_int, soil):
    # choose solver
    # solver = solver_c.NewmarkSolver()
    solver = NewmarkSolver()
    # solver = solver_c.ZhaiSolver()

    all_element_model_parts = []
    all_meshes = []
    # loop over number of segments
    for idx in range(track_info["geometry"]["n_segments"]):
        # set geometry of one segment
        element_model_parts, mesh = create_horizontal_track(track_info["geometry"]["n_sleepers"][idx],
                                                            track_info["geometry"]["sleeper_distance"],
                                                            track_info["geometry"]["depth_soil"][idx])
        # add segment model parts and mesh to list
        all_element_model_parts.append(element_model_parts)
        all_meshes.append(mesh)

    # Setup global mesh and combine model parts of all segments
    rail_model_part, sleeper_model_part, rail_pad_model_part, soil_model_parts, all_mesh = \
        combine_horizontal_tracks(all_element_model_parts, all_meshes)

    # Fixate the bottom boundary
    bottom_boundaries = [add_no_displacement_boundary_to_bottom(soil_model_part)["bottom_boundary"] for soil_model_part
                         in soil_model_parts]

    # set initialisation time
    initialisation_time = np.linspace(0, time_int["tot_ini_time"], time_int["n_t_ini"])
    # set calculation time
    calculation_time = np.linspace(initialisation_time[-1], initialisation_time[-1] + time_int["tot_calc_time"],
                                   time_int["n_t_calc"])
    # Combine all time steps in an array
    time = np.concatenate((initialisation_time, calculation_time[1:]))

    # set elements
    material = Material()
    material.youngs_modulus = track_info["materials"]["young_mod_beam"]
    material.poisson_ratio = track_info["materials"]["poisson_beam"]
    material.density = track_info["materials"]["rho"]

    section = Section()
    section.area = track_info["materials"]["rail_area"]
    section.sec_moment_of_inertia = track_info["materials"]["inertia_beam"]
    section.shear_factor = track_info["materials"]["shear_factor_rail"]

    rail_model_part.section = section
    rail_model_part.material = material

    rail_pad_model_part.mass = track_info["materials"]["mass_rail_pad"]
    rail_pad_model_part.stiffness = track_info["materials"]["stiffness_rail_pad"]
    rail_pad_model_part.damping = track_info["materials"]["damping_rail_pad"]

    sleeper_model_part.mass = track_info["materials"]["mass_sleeper"]

    for idx, soil_model_part in enumerate(soil_model_parts):
        soil_model_part.stiffness = soil["stiffness"][idx]
        soil_model_part.damping = soil["damping"][idx]

    # set velocity of train
    velocities = np.ones(len(time)) * train_info["velocity"]

    # prevent train from moving in initialisation phase
    velocities[0:len(initialisation_time)] = 0

    # constraint rotation at the side boundaries
    side_boundaries = ConstraintModelPart(x_disp_dof=False, y_disp_dof=True, z_rot_dof=True)
    side_boundaries.nodes = [rail_model_part.nodes[0], rail_model_part.nodes[-1]]

    # populate global system
    track = GlobalSystem()
    track.mesh = all_mesh
    track.time = time

    # collect all model parts track
    model_parts = [rail_model_part, rail_pad_model_part, sleeper_model_part, side_boundaries] \
                  + soil_model_parts + bottom_boundaries
    track.model_parts = model_parts

    # set up train
    train = TrainModel()
    train.time = time
    train.velocities = velocities

    # set up carts
    train.cart_distances = train_info["cart_distances"]
    train.carts = [Cart() for idx in range(len(train_info["cart_distances"]))]
    for cart in train.carts:
        cart.bogie_distances = train_info["bogie_distances"]
        cart.inertia = train_info["inertia_cart"]
        cart.mass = train_info["mass_cart"]
        cart.stiffness = train_info["sec_stiffness"]
        cart.damping = train_info["sec_damping"]
        cart.length = train_info["cart_length"]

        # setup bogies per cart
        cart.bogies = [Bogie() for idx in range(len(train_info["bogie_distances"]))]
        for bogie in cart.bogies:
            bogie.wheel_distances = train_info["wheel_distances"]
            bogie.mass = train_info["mass_bogie"]
            bogie.inertia = train_info["inertia_bogie"]
            bogie.stiffness = train_info["prim_stiffness"]
            bogie.damping = train_info["prim_damping"]
            bogie.length = train_info["bogie_length"]


            # setup wheels per bogie
            bogie.wheels = [Wheel() for idx in range(len(train_info["wheel_distances"]))]
            for wheel in bogie.wheels:
                wheel.mass = train_info["mass_wheel"]

    # setup coupled train track system
    coupled_model = CoupledTrainTrack()

    coupled_model.train = train
    coupled_model.track = track
    coupled_model.rail = rail_model_part
    coupled_model.time = time
    coupled_model.initialisation_time = initialisation_time

    coupled_model.hertzian_contact_coef = track_info["materials"]["hertzian_contact_coef"]
    coupled_model.hertzian_power = track_info["materials"]["hertzian_power"]

    coupled_model.solver = solver

    coupled_model.is_rayleigh_damping = True
    coupled_model.damping_ratio = track_info["materials"]["damping_ratio"]
    coupled_model.radial_frequency_one = track_info["materials"]["omega_one"]
    coupled_model.radial_frequency_two = track_info["materials"]["omega_two"]

    return coupled_model


def get_results_coupled_model(coupled_model, output_interval):

    top_nodes = [element.nodes[0] for element in coupled_model.track.model_parts[4].elements]
    vertical_displacements_soil = np.array(
        [node.displacements[0::output_interval, 1] for node in top_nodes])
    # vertical_displacements_soil = np.array(
    #     [node.displacements[0::output_interval, 1] for node in coupled_model.track.model_parts[4].nodes])
    vertical_force_soil = np.array(
        [element.force[0::output_interval, 0] for element in coupled_model.track.model_parts[4].elements])
    mid_idx_force =  int(vertical_force_soil.shape[0]/2)

    dynamic_stiffness_soil = vertical_force_soil/vertical_displacements_soil

    return vertical_displacements_soil, vertical_force_soil[mid_idx_force,:], dynamic_stiffness_soil