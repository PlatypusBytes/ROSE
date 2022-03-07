import os
import pickle
# import ROSE packages
from run_rose.read_wolf import read_wolf
from rose.model.model_part import Material, Section
from rose.model.train_model import *
from rose.model.train_track_interaction import *

import solvers.newmark_solver as solver_c
# import rose.model.solver as solver_c


def train_model():
    # set up train
    train = {}
    # set up bogie configuration
    train["wheel_distances"] = [-1.25, 1.25]  # wheel distances from the centre of the bogie [m]
    train["bogie_length"] = 2  # length of the bogie [m]

    # set up cart configuration
    train["bogie_distances"] = [-10, 10]  # bogie distances from the centre of the cart [m]
    train["cart_length"] = 28  # length of the cart [m]

    # set up train configuration
    train["cart_distances"] = [26.55 + 14]  # cart distances from the start of the track [m]

    # set train parameters
    train["mass_wheel"] = 1834  # mass of one wheel [kg]
    train["mass_bogie"] = 6e3  # mass of one bogie [kg]
    train["mass_cart"] = 75.5e3  # mass of one cart  [kg]

    train["inertia_bogie"] = 0.31e3  # mass inertia of one bogie   [kg.m2]
    train["inertia_cart"] = 128.8e3  # mass inertia of one cart   [kg.m2]

    train["prim_stiffness"] = 4800e3  # primary suspension: stiffness between wheels and bogie  [N/m]
    train["sec_stiffness"] = 2708e3  # secondary suspension: stiffness between bogies and cart  [N/m]

    train["prim_damping"] = 0.25e3  # primary suspension: damping between wheels and bogie  [N.s/m]
    train["sec_damping"] = 64e3  # secondary suspension: damping between bogies and cart  [N.s/m]

    return train


def geometry(nb_sleeper, fact=1):
    # Set geometry parameters
    geometry = {}
    geometry["n_segments"] = len(nb_sleeper)
    geometry["n_sleepers"] = [int(n / fact) for n in nb_sleeper]  # number of sleepers per segment
    geometry["sleeper_distance"] = 0.6 * fact  # distance between sleepers, equal for each segment
    geometry["depth_soil"] = [1, 1]  # depth of the soil [m] per segment

    return geometry


def materials():
    material = {}
    # set parameters of the rail
    material["young_mod_beam"] = 210e9  # young modulus rail
    material["poisson_beam"] = 0.0  # poison ration rail
    material["inertia_beam"] = 2.24E-05  # inertia of the rail
    material["rho"] = 7860  # density of the rail
    material["rail_area"] = 69.6e-2  # area of the rail
    material["shear_factor_rail"] = 0  # Timoshenko shear factor

    # Rayleigh damping system
    material["damping_ratio"] = 0.02  # damping
    material["omega_one"] = 6.283  # first radial_frequency
    material["omega_two"] = 125.66  # second radial_frequency

    # set parameters rail pad
    material["mass_rail_pad"] = 5  # mass of the rail pad [kg]
    material["stiffness_rail_pad"] = 750e6  # stiffness of the rail pad [N/m2]
    material["damping_rail_pad"] = 750e3  # damping of the rail pad [N/m2/s]

    # set parameters sleeper
    material["mass_sleeper"] = 140  # [kg]

    # set up contact parameters
    material["hertzian_contact_coef"] = 9.1e-7  # Hertzian contact coefficient
    material["hertzian_power"] = 3 / 2  # Hertzian power

    return material


def time_integration():
    time = {}
    # set time parameters in two stages
    time["tot_ini_time"] = 0.5  # total initalisation time  [s]
    time["n_t_ini"] = 5000  # number of time steps initialisation time  [-]

    time["tot_calc_time"] = 1.2  # total time during calculation phase   [s]
    time["n_t_calc"] = 8000  # number of time steps during calculation phase [-]

    return time


def soil_parameters(sleeper_distance, stiffness, damping):
    # Set soil parameters of each segment
    soil = {"stiffness_soils": [s * sleeper_distance for s in stiffness],
            "damping_soils": [d * sleeper_distance for d in damping]}

    return soil


def create_model(tr, geometry, mat, time_int, soil, velocity):
    # choose solver
    solver = solver_c.NewmarkSolver()
    # solver = solver_c.ZhaiSolver()

    all_element_model_parts = []
    all_meshes = []
    # loop over number of segments
    for idx in range(geometry["n_segments"]):
        # set geometry of one segment
        element_model_parts, mesh = create_horizontal_track(geometry["n_sleepers"][idx],
                                                            geometry["sleeper_distance"],
                                                            geometry["depth_soil"][idx])
        # add segment model parts and mesh to list
        all_element_model_parts.append(element_model_parts)
        all_meshes.append(mesh)

    # Setup global mesh and combine model parts of all segments
    rail_model_part, sleeper_model_part, rail_pad_model_part, soil_model_parts, all_mesh = \
        combine_horizontal_tracks(all_element_model_parts, all_meshes)

    hinge_rail_model_parts = add_semi_rigid_hinge_at_x(rail_model_part, 18.6, 2 / 3)

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
    material.youngs_modulus = mat["young_mod_beam"]
    material.poisson_ratio = mat["poisson_beam"]
    material.density = mat["rho"]

    section = Section()
    section.area = mat["rail_area"]
    section.sec_moment_of_inertia = mat["inertia_beam"]
    section.shear_factor = mat["shear_factor_rail"]

    rail_model_part.section = section
    rail_model_part.material = material

    for hinge_model_part in hinge_rail_model_parts:
        hinge_model_part.section = section
        hinge_model_part.material = material


    rail_pad_model_part.mass = mat["mass_rail_pad"]
    rail_pad_model_part.stiffness = mat["stiffness_rail_pad"]
    rail_pad_model_part.damping = mat["damping_rail_pad"]

    sleeper_model_part.mass = mat["mass_sleeper"]

    for idx, soil_model_part in enumerate(soil_model_parts):
        soil_model_part.stiffness = soil["stiffness_soils"][idx]
        soil_model_part.damping = soil["damping_soils"][idx]

    # set velocity of train
    velocities = np.ones(len(time)) * velocity

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
                  + soil_model_parts + bottom_boundaries + hinge_rail_model_parts
    track.model_parts = model_parts

    # set up train
    train = TrainModel()
    train.use_irregularities = True
    train.time = time
    train.velocities = velocities

    # set up carts
    train.cart_distances = tr["cart_distances"]
    train.carts = [Cart() for idx in range(len(tr["cart_distances"]))]
    for cart in train.carts:
        cart.bogie_distances = tr["bogie_distances"]
        cart.inertia = tr["inertia_cart"]
        cart.mass = tr["mass_cart"]
        cart.stiffness = tr["sec_stiffness"]
        cart.damping = tr["sec_damping"]
        cart.length = tr["cart_length"]
        # cart.calculate_total_n_dof()

        # setup bogies per cart
        cart.bogies = [Bogie() for idx in range(len(tr["bogie_distances"]))]
        for bogie in cart.bogies:
            bogie.wheel_distances = tr["wheel_distances"]
            bogie.mass = tr["mass_bogie"]
            bogie.inertia = tr["inertia_bogie"]
            bogie.stiffness = tr["prim_stiffness"]
            bogie.damping = tr["prim_damping"]
            bogie.length = tr["bogie_length"]
            # bogie.calculate_total_n_dof()

            # setup wheels per bogie
            bogie.wheels = [Wheel() for idx in range(len(tr["wheel_distances"]))]
            for wheel in bogie.wheels:
                wheel.mass = tr["mass_wheel"]

    # setup coupled train track system
    coupled_model = CoupledTrainTrack()

    coupled_model.train = train
    coupled_model.track = track
    coupled_model.rail = rail_model_part
    coupled_model.time = time
    coupled_model.initialisation_time = initialisation_time

    coupled_model.hertzian_contact_coef = mat["hertzian_contact_coef"]
    coupled_model.hertzian_power = mat["hertzian_power"]

    coupled_model.solver = solver

    coupled_model.is_rayleigh_damping = True
    coupled_model.damping_ratio = mat["damping_ratio"]
    coupled_model.radial_frequency_one = mat["omega_one"]
    coupled_model.radial_frequency_two = mat["omega_two"]

    return coupled_model


def write_results(coupled_model: CoupledTrainTrack, segment_id: str, output_dir: str,
                  output_interval: int = 10):
    """
    Writes dynamic results of a couple model

    :param coupled_model: current coupled model
    :param segment_id: id of the current segment
    :param output_dir: output directory
    :param output_interval: interval of how many timesteps should be written in output
    :return:
    """

    # check if output folder exists
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    # collect results
    vertical_displacements_rail = np.array(
        [node.displacements[0::output_interval, 1] for node in coupled_model.track.model_parts[0].nodes])
    vertical_force_rail = np.array(
        [element.force[0::output_interval, 1] for element in coupled_model.track.model_parts[0].elements])
    coords_rail = np.array([node.coordinates[0] for node in coupled_model.track.model_parts[0].nodes])

    vertical_displacements_rail_pad = np.array(
        [node.displacements[0::output_interval, 1] for node in coupled_model.track.model_parts[1].nodes])
    vertical_force_rail_pad = np.array(
        [element.force[0::output_interval, 1] for element in coupled_model.track.model_parts[1].elements])
    coords_rail_pad = np.array([node.coordinates[0] for node in coupled_model.track.model_parts[1].nodes])

    vertical_displacements_sleeper = np.array(
        [node.displacements[0::output_interval, 1] for node in coupled_model.track.model_parts[2].nodes])
    # vertical_force_sleeper = np.array(
    #     [node.force[0::output_interval, 1] for node in coupled_model.track.model_parts[2].nodes])
    # coords_sleeper = np.array([node.coordinates[0] for node in coupled_model.track.model_parts[2].nodes])

    vertical_displacements_soil = np.array(
        [node.displacements[0::output_interval, 1] for node in coupled_model.track.model_parts[4].nodes])
    vertical_force_soil = np.array(
        [element.force[0::output_interval, 0] for element in coupled_model.track.model_parts[4].elements])
    coords_soil = np.array([node.coordinates[0] for node in coupled_model.track.model_parts[4].nodes])

    vertical_displacements_train = np.array(
        [node.displacements[0::output_interval, 1] for node in coupled_model.train.nodes])
    vertical_force_train = np.array([node.force[0::output_interval, 1] for node in coupled_model.train.nodes])

    result_track = {"name": segment_id,
                    # "omega": omega,
                    "time": coupled_model.time[0::output_interval].tolist(),
                    "velocity": coupled_model.train.velocities[0::output_interval].tolist(),
                    "vert_disp_rail": vertical_displacements_rail.tolist(),
                    "vert_force_rail": vertical_force_rail.tolist(),
                    "coords_rail": coords_rail.tolist(),
                    "vertical_displacements_rail_pad": vertical_displacements_rail_pad.tolist(),
                    "vertical_force_rail_pad": vertical_force_rail_pad.tolist(),
                    "coords_rail_pad": coords_rail_pad.tolist(),
                    "vertical_displacements_sleeper": vertical_displacements_sleeper.tolist(),
                    # "vertical_force_sleeper": vertical_force_sleeper.tolist(),
                    # "coords_sleeper": coords_sleeper.tolist(),
                    "vertical_displacements_soil": vertical_displacements_soil.tolist(),
                    "vertical_force_soil": vertical_force_soil.tolist(),
                    "coords_soil": coords_soil.tolist(),
                    "vertical_displacements_train": vertical_displacements_train.tolist(),
                    "vertical_force_train": vertical_force_train.tolist(),
                    }

    # filename
    file_name = f'res_{segment_id}.pickle'
    # dump pickle
    with open(os.path.join(output_dir, file_name), "wb") as f:
        pickle.dump(result_track, f)

    return


def main():
    nb_sleepers = [100, 100]
    stiffness = [158e6, 180e6]
    damping = [30e3, 20e3]
    speed = 100 / 3.6
    output_dir = "./res"

    # create train
    tr = train_model()
    # create geometry
    geom = geometry(nb_sleepers)
    # materials
    mat = materials()
    # time integration
    tim = time_integration()
    # soil parameters
    soil = soil_parameters(geom["sleeper_distance"], stiffness, damping)
    # define train-track mode model
    coupled_model = create_model(tr, geom, mat, tim, soil, speed)
    # calculate
    coupled_model.main()
    # write results
    for sce in range(len(nb_sleepers)):
        write_results(coupled_model, sce, output_dir, output_interval=10)


if __name__ == "__main__":
    import cProfile

    cProfile.run('main()','profiler')
    # main()
