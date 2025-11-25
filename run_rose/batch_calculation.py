
import gc
import os
from os.path import isfile, join
import pickle

from run_rose.read_wolf import read_wolf
from rose.model.model_part import Material, Section
from rose.pre_process.default_trains import TrainType, set_train
from rose.model.train_track_interaction import *
from solvers.newmark_solver import NewmarkSolver

# Set geometry parameters
n_segments = 1

fact = 1
n_sleepers = [int(200/fact)]    # number of sleepers per segment
sleeper_distance = 0.6 * fact # distance between sleepers, equal for each segment
depth_soil = [1]      # depth of the soil [m] per segment
n_rail_per_sleeper = 1 # number of rail elements between two sleepers [-]

# Set soil parameters of each segment
stiffness_soils = [180e6 * sleeper_distance] # will be overwritten
# damping_soils = [1500e3 * sleeper_distance, 1500e3 * sleeper_distance]
damping_soils = [0 * sleeper_distance] # will be overwritten
# set parameters of the rail
youngs_mod_beam = 210e9     # youngs modulus rail
poison_beam = 0.0           # poison ratio rail
intertia_beam = 2.24E-05    # inertia of the rail
rho = 7860                  # density of the rail
rail_area = 69.6e-2
shear_factor_rail = 0
damping_ratio_rail = 0.02
omega_one_rail = 6.283          # first radial_frequency rail
omega_two_rail = 125.66        # second radial_frequency rail

# set parameters rail pad
mass_rail_pad = 5           # mass of the rail pad [kg]
stiffness_rail_pad = 750e6  # stiffness of the rail pad [N/m2]
damping_rail_pad = 750e3

# set parameters sleeper
mass_sleeper = 140        # [kg]

# set up train
starting_distance_mid_cart = 40.55
train_type = TrainType.INTERCITY

# set up velocity
velocity = 38.9/1              # constant velocity of the train [m/s]

# set up contact parameters
herzian_contact_coef = 9.1e-7   # herzian contact coefficient
herzian_power = 3 / 2          # herzian power

# set time parameters in two stages

# set initialisation time, during this time, the train does not move
tot_ini_time = 0.5      # total initalisation time  [s]
n_t_ini = 5000          # number of time steps initialisation time  [-]

# set calculation time, during this time, the train does move
tot_calc_time = 1.2       # total time during calculation phase   [s]
n_t_calc = 8000        # number of time steps during calculation phase [-]

# choose solver
solver = NewmarkSolver()


def set_base_model():
    """
    Sets the base coupled model with the predefined parameters
    :return:
    """

    all_element_model_parts = []
    all_meshes = []
    # loop over number of segments
    for idx in range(n_segments):

        # set geometry of one segment
        element_model_parts, mesh = create_horizontal_track(
            n_sleepers[idx], sleeper_distance, depth_soil[idx], n_rail_per_sleeper
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
    # set material rail
    material = Material()
    material.youngs_modulus = youngs_mod_beam  # Pa
    material.poisson_ratio = poison_beam
    material.density = rho

    # set section rail
    section = Section()
    section.area = rail_area
    section.sec_moment_of_inertia = intertia_beam
    section.shear_factor = shear_factor_rail

    # set rail
    rail_model_part.section = section
    rail_model_part.material = material
    rail_model_part.damping_ratio = damping_ratio_rail
    rail_model_part.radial_frequency_one = omega_one_rail
    rail_model_part.radial_frequency_two = omega_two_rail

    # set rail pad
    rail_pad_model_part.mass = mass_rail_pad
    rail_pad_model_part.stiffness = stiffness_rail_pad
    rail_pad_model_part.damping = damping_rail_pad

    # set sleeper
    sleeper_model_part.mass = mass_sleeper
    sleeper_model_part.distance_between_sleepers = sleeper_distance

    for idx, soil_model_part in enumerate(soil_model_parts):
        soil_model_part.stiffness = stiffness_soils[idx]
        soil_model_part.damping = damping_soils[idx]

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
                  + soil_model_parts + bottom_boundaries
    track.model_parts = model_parts

    # set up train
    train = set_train(time, velocities, starting_distance_mid_cart, train_type)

    # setup coupled train track system
    coupled_model = CoupledTrainTrack()

    coupled_model.train = train
    coupled_model.track = track
    coupled_model.rail = rail_model_part
    coupled_model.time = time
    coupled_model.initialisation_time = initialisation_time

    coupled_model.hertzian_contact_coef = herzian_contact_coef
    coupled_model.hertzian_power = herzian_power

    coupled_model.solver = solver

    coupled_model.is_rayleigh_damping = True
    coupled_model.damping_ratio = damping_ratio_rail
    coupled_model.radial_frequency_one = omega_one_rail
    coupled_model.radial_frequency_two = omega_two_rail

    return coupled_model

def calculate(soil_stiffness: float, soil_damping: float, coupled_model: CoupledTrainTrack, velocity: velocity):
    """
    Calculates the coupled model

    :param soil_stiffness: different soil stiffness
    :param soil_damping:  different soil damping
    :param coupled_model:  current coupled model
    :param velocity: velocity at each time step
    :return:
    """
    soil = coupled_model.track.model_parts[4]
    soil.stiffness = soil_stiffness
    soil.damping = soil_damping

    train_vel = coupled_model.train.velocities
    train_vel[train_vel>1e-10] = velocity

    # calculate
    coupled_model.main()

def write_results(coupled_model: CoupledTrainTrack, segment_id: str, omega: float, output_dir: str,
                  output_interval: int =10):
    """
    Writes dynamic results of a couple model

    :param coupled_model: current coupled model
    :param segment_id: id of the current segment
    :param omega:
    :param output_dir: output directory
    :param output_interval: interval of how many timesteps should be written in output
    :return:
    """

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

    vertical_displacements_train = np.array([node.displacements[0::output_interval, 1] for node in coupled_model.train.nodes])
    vertical_force_train = np.array([node.force[0::output_interval, 1] for node in coupled_model.train.nodes])

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
                    # "vertical_force_sleeper": vertical_force_sleeper.tolist(),
                    # "coords_sleeper": coords_sleeper.tolist(),
                    "vertical_displacements_soil": vertical_displacements_soil.tolist(),
                    "vertical_force_soil": vertical_force_soil.tolist(),
                    "coords_soil": coords_soil.tolist(),
                    "vertical_displacements_train": vertical_displacements_train.tolist(),
                    "vertical_force_train": vertical_force_train.tolist(),
                    }


    file_name = f'res_{segment_id}.pickle'
    # file_name = f'res_{int(coupled_model.train.velocities[-1])}.pickle'

    with open(os.path.join(output_dir, file_name), "wb") as f:
        pickle.dump(result_track, f)


def main():

    # set base coupled model
    coupled_model = set_base_model()

    # set output directory
    cd = os.getcwd()
    output_dir = os.path.join(cd, "../rose/batch_results")

    # set wolf results directory
    wolf_res_path = r'../rose/utils/dyn_stiffness'
    wolf_files = [os.path.join(wolf_res_path, f) for f in os.listdir(wolf_res_path) if isfile(join(wolf_res_path, f))]

    # read wolf files
    results = read_wolf(wolf_files)

    # get all segment ids
    segment_ids = [res['name'] for res in results]

    # get all omgeas
    omegas = [res['omega'] for res in results]

    # get all stiffnesses
    stiffnesses = [res['stiffness'] * sleeper_distance for res in results]
    dampings = [res['damping'] * sleeper_distance for res in results]

    # loop over all segments and calculate
    for stiffness, damping, segment_id, omega in zip(stiffnesses, dampings, segment_ids, omegas):
        new_coupled_model = copy.deepcopy(coupled_model)
        calculate(stiffness, damping, new_coupled_model, velocity)
        write_results(new_coupled_model, segment_id, omega, output_dir, output_interval=10)

        # delete garbage
        del new_coupled_model
        gc.collect()

if __name__ == "__main__":
    main()




