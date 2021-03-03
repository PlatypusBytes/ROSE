


import os
import pickle

from run_rose.read_wolf import read_wolf
from rose.base.model_part import Material, Section
from rose.train_model.train_model import *
from rose.base.train_track_interaction import *
import rose.solver.solver as solver_c

# from pympler import tracker
# tr = tracker.SummaryTracker()

import gc


def set_base_model():
    # Set geometry parameters
    n_segments = 2

    fact = 1
    n_sleepers = [int(100/fact), int(100/fact)]    # number of sleepers per segment
    sleeper_distance = 0.6 * fact # distance between sleepers, equal for each segment
    # sleeper_distance = 0.5 * fact
    depth_soil = [1,1]      # depth of the soil [m] per segment

    # Set soil parameters of each segment
    stiffness_soils = [180e6 * sleeper_distance] # will be overwritten
    stiffness_soils = [180e7 * sleeper_distance, 18e6 * sleeper_distance] # will be overwritten
    damping_soils = [1500e3 * sleeper_distance, 150e3 * sleeper_distance]
    # damping_soils = [0 * sleeper_distance] # will be overwritten
    # set parameters of the rail
    youngs_mod_beam = 210e9     # youngs modulus rail
    # youngs_mod_beam = 1.28e7     # youngs modulus rail
    poison_beam = 0.0           # poison ration rail
    intertia_beam = 2.24E-05    # inertia of the rail
    # intertia_beam = 1    # inertia of the rail
    rho = 7860                  # density of the rail
    # rho = 120
    rail_area = 69.6e-2
    # rail_area = 1
    shear_factor_rail = 0
    damping_ratio_rail = 0.02
    # damping_ratio_rail = 0.3
    omega_one_rail = 6.283          # first radial_frequency rail
    omega_two_rail = 125.66        # second radial_frequency rail

    # set parameters rail pad
    mass_rail_pad = 5           # mass of the rail pad [kg]
    stiffness_rail_pad = 750e6  # stiffness of the rail pad [N/m2]
    damping_rail_pad = 750e3

    # set parameters sleeper
    mass_sleeper = 140        # [kg]

    # set up train
    # set up bogie configuration
    wheel_distances = [-1.25, 1.25]  # wheel distances from the centre of the bogie [m]
    bogie_length = 2                 # length of the bogie [m]

    # wheel_distances = [0]  # wheel distances from the centre of the bogie [m]
    # bogie_length = 0                 # length of the bogie [m]

    # set up cart configuration
    bogie_distances = [-10, 10]        # bogie distances from the centre of the cart [m]
    cart_length = 28                 # length of the cart [m]

    # bogie_distances = [0]        # bogie distances from the centre of the cart [m]
    # cart_length = 0                # length of the cart [m]

    # set up train configuration
    cart_distances = [26.55 + 14]    # cart distances from the start of the track [m]
    # cart_distances = [26.55 + 0]    # cart distances from the start of the track [m]

    # set train parameters
    mass_wheel = 1834         # mass of one wheel [kg]
    mass_bogie = 6e3         # mass of one bogie [kg]
    mass_cart = 75.5e3           # mass of one cart  [kg]

    inertia_bogie = 0.31e3           # mass interia of one bogie   [?]
    inertia_cart = 128.8e3            # mass interia of one cart   [?]

    prim_stiffness = 4800e3        # stiffness between wheels and bogie
    sec_stiffness = 2708e3         # stiffness between bogies and cart

    prim_damping = 0.25e3         # damping between wheels and bogie
    sec_damping = 64e3          # damping between bogies and cart

    # prim_damping = 0      # damping between wheels and bogie
    # sec_damping = 0         # damping between bogies and cart

    # prim_stiffness = 4800e3        # stiffness between wheels and bogie
    # sec_stiffness = 2708e3         # stiffness between bogies and cart
    #
    # prim_damping = 0.25e3         # damping between wheels and bogie
    # sec_damping = 64e3          # damping between bogies and cart

    # set up velocity
    velocity = 38.9             # constant velocity of the train [m/s]

    # set up contact parameters
    herzian_contact_coef = 9.1e-7   # herzian contact coefficient
    herzian_power = 3 / 2          # herzian power

    # set time parameters in two stages
    tot_ini_time = 0.5      # total initalisation time  [s]
    n_t_ini = 5000          # number of time steps initialisation time  [-]

    tot_calc_time = 1.2      # total time during calculation phase   [s]
    n_t_calc = 8000        # number of time steps during calculation phase [-]

    # choose solver
    solver = solver_c.NewmarkSolver()
    # solver = solver_c.ZhaiSolver()

    all_element_model_parts = []
    all_meshes = []
    # loop over number of segments
    for idx in range(n_segments):

        # set geometry of one segment
        element_model_parts, mesh = create_horizontal_track(
            n_sleepers[idx], sleeper_distance, depth_soil[idx]
        )
        # add segment model parts and mesh to list
        all_element_model_parts.append(element_model_parts)
        all_meshes.append(mesh)

    # Setup global mesh and combine model parts of all segments
    rail_model_part, sleeper_model_part, rail_pad_model_part, soil_model_parts, all_mesh = \
        combine_horizontal_tracks(all_element_model_parts, all_meshes)

    # Fixate the bottom boundary
    bottom_boundaries = [add_no_displacement_boundary_to_bottom(soil_model_part)["bottom_boundary"] for soil_model_part in soil_model_parts]

    # set initialisation time
    initialisation_time = np.linspace(0, tot_ini_time, n_t_ini)
    # set calculation time
    calculation_time = np.linspace(initialisation_time[-1], initialisation_time[-1] + tot_calc_time, n_t_calc)
    # Combine all time steps in an array
    time = np.concatenate((initialisation_time, calculation_time[1:]))

    # set elements
    material = Material()
    material.youngs_modulus = youngs_mod_beam  # Pa
    material.poisson_ratio = poison_beam
    material.density = rho

    section = Section()
    section.area = rail_area
    section.sec_moment_of_inertia = intertia_beam
    section.shear_factor = shear_factor_rail

    rail_model_part.section = section
    rail_model_part.material = material
    rail_model_part.damping_ratio = damping_ratio_rail
    rail_model_part.radial_frequency_one = omega_one_rail
    rail_model_part.radial_frequency_two = omega_two_rail

    rail_pad_model_part.mass = mass_rail_pad  # 5
    rail_pad_model_part.stiffness = stiffness_rail_pad
    rail_pad_model_part.damping = damping_rail_pad #12e3  # 12e3

    sleeper_model_part.mass = mass_sleeper   # 162.5
    sleeper_model_part.distance_between_sleepers = sleeper_distance

    for idx, soil_model_part in enumerate(soil_model_parts):
        soil_model_part.stiffness = stiffness_soils[idx]
        soil_model_part.damping = damping_soils[idx]

    # set velocity of train
    velocities = np.ones(len(time)) * velocity

    # prevent train from moving in initialisation phase
    velocities[0:len(initialisation_time)] = 0

    # constraint rotation at the side boundaries
    side_boundaries = ConstraintModelPart(normal_dof=False, y_disp_dof=True, z_rot_dof=True)
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
    train.cart_distances = cart_distances
    train.carts = [Cart() for idx in range(len(cart_distances))]
    for cart in train.carts:
        cart.bogie_distances = bogie_distances
        cart.inertia = inertia_cart
        cart.mass = mass_cart
        cart.stiffness = sec_stiffness
        cart.damping = sec_damping
        cart.length = cart_length
        cart.calculate_total_n_dof()

        # setup bogies per cart
        cart.bogies = [Bogie() for idx in range(len(bogie_distances))]
        for bogie in cart.bogies:
            bogie.wheel_distances = wheel_distances
            bogie.mass = mass_bogie
            bogie.inertia = inertia_bogie
            bogie.stiffness = prim_stiffness
            bogie.damping = prim_damping
            bogie.length = bogie_length
            bogie.calculate_total_n_dof()

            # setup wheels per bogie
            bogie.wheels = [Wheel() for idx in range(len(wheel_distances))]
            for wheel in bogie.wheels:
                wheel.mass = mass_wheel

    # setup coupled train track system
    coupled_model = CoupledTrainTrack()

    coupled_model.train = train
    coupled_model.track = track
    coupled_model.rail = rail_model_part
    coupled_model.time = time
    coupled_model.initialisation_time = initialisation_time

    coupled_model.herzian_contact_coef = herzian_contact_coef
    coupled_model.herzian_power = herzian_power

    coupled_model.solver = solver

    coupled_model.is_rayleigh_damping = True
    coupled_model.damping_ratio = damping_ratio_rail
    coupled_model.radial_frequency_one = omega_one_rail
    coupled_model.radial_frequency_two = omega_two_rail

    return coupled_model

def calculate(soil_stiffness, soil_damping, coupled_model):
    # soil = coupled_model.track.model_parts[4]
    # soil.stiffness = soil_stiffness
    # soil.damping = soil_damping

    # soil = coupled_model.track.model_parts[4]
    # soil.stiffness = 400e3
    # soil.damping = 400e2

    # calculate
    coupled_model.main()

def write_results(coupled_model, segment_id, omega, output_dir, output_interval=10):

    # collect results
    vertical_displacements_rail = np.array(
        [node.displacements[0::output_interval, 1] for node in coupled_model.track.model_parts[0].nodes])
    vertical_force_rail = np.array(
        [node.force[0::output_interval, 1] for node in coupled_model.track.model_parts[0].nodes])
    coords_rail = np.array([node.coordinates[0] for node in coupled_model.track.model_parts[0].nodes])

    vertical_displacements_rail_pad = np.array(
        [node.displacements[0::output_interval, 1] for node in coupled_model.track.model_parts[1].nodes])
    vertical_force_rail_pad = np.array(
        [node.force[0::output_interval, 1] for node in coupled_model.track.model_parts[1].nodes])
    coords_rail_pad = np.array([node.coordinates[0] for node in coupled_model.track.model_parts[1].nodes])

    vertical_displacements_sleeper = np.array(
        [node.displacements[0::output_interval, 1] for node in coupled_model.track.model_parts[2].nodes])
    vertical_force_sleeper = np.array(
        [node.force[0::output_interval, 1] for node in coupled_model.track.model_parts[2].nodes])
    coords_sleeper = np.array([node.coordinates[0] for node in coupled_model.track.model_parts[2].nodes])

    vertical_displacements_soil = np.array(
        [node.displacements[0::output_interval, 1] for node in coupled_model.track.model_parts[4].nodes])
    vertical_force_soil = np.array(
        [node.force[0::output_interval, 1] for node in coupled_model.track.model_parts[4].nodes])
    coords_soil = np.array([node.coordinates[0] for node in coupled_model.track.model_parts[4].nodes])

    vertical_displacements_train = np.array([node.displacements[0::output_interval, 1] for node in coupled_model.train.nodes])
    vertical_force_train = np.array([node.force[0::output_interval, 1] for node in coupled_model.train.nodes])

    rot_train = np.array([node.displacements[0::output_interval, 2] for node in coupled_model.train.nodes])

    result_track = {"name": segment_id,
                    "omega": omega,
                    "time": coupled_model.time[0::output_interval].tolist(),
                    "velocity": coupled_model.train.velocities[0::output_interval].tolist(),
                    "vert_disp_rail": vertical_displacements_rail.tolist(),
                    "vert_force_rail": vertical_force_rail.tolist(),
                    "coords_rail": coords_rail.tolist(),
                    "vertical_displacements_rail_pad": vertical_displacements_rail_pad.tolist(),
                    "vertical_force_rail_pad": vertical_force_rail_pad.tolist(),
                    "coords_rail_pad": coords_rail_pad.tolist(),
                    "vertical_displacements_sleeper": vertical_displacements_sleeper.tolist(),
                    "vertical_force_sleeper": vertical_force_sleeper.tolist(),
                    "coords_sleeper": coords_sleeper.tolist(),
                    "vertical_displacements_soil": vertical_displacements_soil.tolist(),
                    "vertical_force_soil": vertical_force_soil.tolist(),
                    "coords_soil": coords_soil.tolist(),
                    "vertical_displacements_train": vertical_displacements_train.tolist(),
                    "vertical_force_train": vertical_force_train.tolist(),
                    "rot_train": rot_train.tolist()
                    }


    file_name = f'res_{segment_id}_tmp3.pickle'

    with open(os.path.join(output_dir, file_name), "wb") as f:
        pickle.dump(result_track, f)


def main():

    coupled_model = set_base_model()

    cd = os.getcwd()

    output_dir = os.path.join(cd, "../rose/batch_results")


    wolf_res_path = r'../rose/wolf/dyn_stiffness'
    # wolf_files = [os.path.join(wolf_res_path, f) for f in os.listdir(wolf_res_path) if isfile(join(wolf_res_path, f))]

    wolf_files = [os.path.join(wolf_res_path,"KDyn_Segment 1090_scenario 1.json")]
    # wolf_files = [os.path.join(wolf_res_path,"KDyn_Segment 1090_scenario 1.json"),
    #               os.path.join(wolf_res_path,"KDyn_Segment 1001_scenario 1.json")]

    results = read_wolf(wolf_files)

    segment_ids = [res['name'] for res in results]
    omegas = [res['omega'] for res in results]
    stiffnesses = [res['stiffness'] * 0.6 for res in results]
    dampings = [res['damping'] * 0.6 for res in results]

    max_stiffness, min_stiffness = max(stiffnesses), min(stiffnesses)
    max_damping, min_damping = max(dampings), min(dampings)

    # check limits
    # stiffnesses =[max_stiffness, min_stiffness]
    # dampings =[max_damping, min_damping]

    # new_coupled_model = copy.deepcopy(coupled_model)
    # calculate(stiffness, damping, new_coupled_model)

    for stiffness, damping, segment_id, omega in zip(stiffnesses, dampings, segment_ids, omegas):
        new_coupled_model = copy.deepcopy(coupled_model)
        calculate(stiffness, damping, new_coupled_model)
        write_results(new_coupled_model, segment_id, omega, output_dir, output_interval=10)

        # allObjects = muppy.get_objects()
        # sum = summary.summarize(allObjects)
        # summary.print_(sum)
        # tr.print_diff()
        del new_coupled_model
        gc.collect()

if __name__ == "__main__":
    main()




