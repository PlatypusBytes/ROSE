import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import scipy

# Import ROSE packages
from rose.model.model_part import Material, Section
from rose.model.train_track_interaction import *
from rose.pre_process.default_trains import TrainType, set_train
import solvers.newmark_solver as solver_c

SHOW_RESULTS = False

def geometry(nb_sleeper, n_elements_per_sleeper=1, fact=1):
    # Set geometry parameters
    geometry = {}
    geometry["n_segments"] = len(nb_sleeper)  # number of segments
    geometry["n_sleepers"] = [int(n / fact) for n in nb_sleeper]  # number of sleepers per segment
    geometry["sleeper_distance"] = 0.6 * fact  # distance between sleepers, equal for each segment
    geometry["depth_soil"] = [1] * len(nb_sleeper)  # depth of the soil [m] per segment
    geometry["n_elements_per_sleeper"] = n_elements_per_sleeper  # number of elements between sleepers

    return geometry

def materials():
    material = {}
    # set parameters of the rail
    material["young_mod_beam"] = 210e9  # young modulus rail
    material["poisson_beam"] = 0.0  # poison ration rail
    material["inertia_beam"] = 2.24E-05  # inertia of the rail
    material["rho"] = 7860  # density of the rail
    material["rail_area"] = 69.6e-4  # area of the rail
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

def time_integration(running_time, time_step):
    time = {}
    # set time parameters in two stages
    time["tot_ini_time"] = 0.0001  # total initialization time [s]
    time["n_t_ini"] = 5  # number of time steps initialization time [-]

    time["tot_calc_time"] = running_time  # total time during calculation phase [s]
    time["n_t_calc"] = int(running_time / time_step)  # number of time steps during calculation phase [-]

    return time

def soil_parameters(sleeper_distance, stiffness, damping):
    # Set soil parameters of each segment
    soil = {"stiffness_soils": [s * sleeper_distance for s in stiffness],
            "damping_soils": [d * sleeper_distance for d in damping]}

    return soil

def create_model(train_type, train_start_coord, geometry, mat, time_int, soil, velocity,
                use_irregularities, output_interval):
    # Code adapted from demo.py
    # choose solver
    solver = solver_c.NewmarkImplicitForce()
    solver.output_interval = output_interval

    all_element_model_parts = []
    all_meshes = []
    # loop over number of segments
    for idx in range(geometry["n_segments"]):
        # set geometry of one segment
        element_model_parts, mesh = create_horizontal_track(
            geometry["n_sleepers"][idx],
            geometry["sleeper_distance"],
            geometry["depth_soil"][idx],
            n_rail_per_sleeper=geometry["n_elements_per_sleeper"]
        )
        # add segment model parts and mesh to list
        all_element_model_parts.append(element_model_parts)
        all_meshes.append(mesh)

    # Setup global mesh and combine model parts of all segments
    rail_model_part, sleeper_model_part, rail_pad_model_part, soil_model_parts, all_mesh = \
        combine_horizontal_tracks(all_element_model_parts, all_meshes, geometry["sleeper_distance"], geometry["n_elements_per_sleeper"])

    # Fixate the bottom boundary
    bottom_boundaries = [
        add_no_displacement_boundary_to_bottom(soil_model_part)["bottom_boundary"]
        for soil_model_part in soil_model_parts
    ]

    # set initialization time
    initialisation_time = np.linspace(0, time_int["tot_ini_time"], time_int["n_t_ini"])
    # set calculation time
    calculation_time = np.linspace(
        initialisation_time[-1],
        initialisation_time[-1] + time_int["tot_calc_time"],
        time_int["n_t_calc"]
    )
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

    rail_pad_model_part.mass = mat["mass_rail_pad"]
    rail_pad_model_part.stiffness = mat["stiffness_rail_pad"]
    rail_pad_model_part.damping = mat["damping_rail_pad"]

    sleeper_model_part.mass = mat["mass_sleeper"]

    for idx, soil_model_part in enumerate(soil_model_parts):
        soil_model_part.stiffness = soil["stiffness_soils"][idx]
        soil_model_part.damping = soil["damping_soils"][idx]

    # set velocity of train
    velocities = np.ones(len(time)) * velocity

    # prevent train from moving in initialization phase
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

    # create train
    train = set_train(time, velocities, train_start_coord, train_type)
    train.use_irregularities = use_irregularities
    train.irregularity_parameters = {"Av": 0.00002095, "seed": 14}
    train.time = time
    train.velocities = velocities

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

def write_results(coupled_model: CoupledTrainTrack, output_dir: str, file_name: str, geometry: dict):
    """
    Writes dynamic results of a coupled model

    :param coupled_model: current coupled model
    :param output_dir: output directory
    :param file_name: name of the output file
    :return:
    """
    # check if output folder exists
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    # collect results
    vertical_displacements_rail = np.array(
        [node.displacements[:, 1] for node in coupled_model.track.model_parts[0].nodes])
    vertical_force_rail = np.array(
        [element.force[:, 1] for element in coupled_model.track.model_parts[0].elements])
    coords_rail = np.array([node.coordinates[0] for node in coupled_model.track.model_parts[0].nodes])

    vertical_displacements_rail_pad = np.array(
        [node.displacements[:, 1] for node in coupled_model.track.model_parts[1].nodes])
    vertical_force_rail_pad = np.array(
        [element.force[:, 1] for element in coupled_model.track.model_parts[1].elements])
    coords_rail_pad = np.array([node.coordinates[0] for node in coupled_model.track.model_parts[1].nodes])

    vertical_displacements_sleeper = np.array(
        [node.displacements[:, 1] for node in coupled_model.track.model_parts[2].nodes])

    # collect results for soil
    soil_nodes = []
    soil_elements = []

    # get soil
    idx_soil = [i for i, cm in enumerate(coupled_model.track.model_parts) if type(cm).__name__ == "Soil"]

    for i in idx_soil:
        soil_nodes.append(coupled_model.track.model_parts[i].nodes)
        soil_elements.append(coupled_model.track.model_parts[i].elements)
    soil_nodes = list(itertools.chain.from_iterable(soil_nodes))
    soil_elements = list(itertools.chain.from_iterable(soil_elements))

    vertical_displacements_soil = np.array(
        [node.displacements[:, 1] for node in soil_nodes])

    id_s = 0
    soil_id = []
    for i, p in enumerate(coupled_model.track.model_parts):
        if isinstance(p, Soil):
            soil_id.extend(len(coupled_model.track.model_parts[i].elements) * [id_s])
            id_s += 1

    vertical_force_soil = np.array(
        [element.force[:, 0] for element in soil_elements])
    coords_soil = np.array([node.coordinates[0] for node in soil_nodes])

    vertical_displacements_train = np.array(
        [node.displacements[:, 1] for node in coupled_model.train.nodes])
    vertical_force_train = np.array([node.force[:, 1] for node in coupled_model.train.nodes])

    solver_output_indices = coupled_model.solver.output_time_indices

    # collect stiffness and damping of the soil
    soil_stiff = scipy.sparse.lil_matrix(coupled_model.track.global_stiffness_matrix.shape)
    soil_damp = scipy.sparse.lil_matrix(coupled_model.track.global_stiffness_matrix.shape)
    idx_matrix = np.full((coupled_model.track.global_stiffness_matrix.shape[0]), np.nan)
    n = 0
    for j in idx_soil:
        for i, _ in enumerate(coupled_model.track.model_parts[j].elements):
            idx = coupled_model.track.model_parts[j].elements[i].nodes[0].index_dof[1]
            soil_stiff[idx, idx] = soil_stiff[idx, idx] + coupled_model.track.model_parts[j].elements[i].model_parts[0].stiffness
            soil_damp[idx, idx] = soil_damp[idx, idx] + coupled_model.track.model_parts[j].elements[i].model_parts[0].damping
            idx_matrix[n] = idx
            n += 1

    result_track = {"name": file_name,
                    "time": coupled_model.time_out[:].tolist(),
                    "velocity": coupled_model.train.velocities[solver_output_indices].tolist(),
                    "vert_disp_rail": vertical_displacements_rail.tolist(),
                    "vert_force_rail": vertical_force_rail.tolist(),
                    "coords_rail": coords_rail.tolist(),
                    "vertical_displacements_rail_pad": vertical_displacements_rail_pad.tolist(),
                    "vertical_force_rail_pad": vertical_force_rail_pad.tolist(),
                    "coords_rail_pad": coords_rail_pad.tolist(),
                    "vertical_displacements_sleeper": vertical_displacements_sleeper.tolist(),
                    "vertical_displacements_soil": vertical_displacements_soil.tolist(),
                    "vertical_force_soil": vertical_force_soil.tolist(),
                    "coords_soil": coords_soil.tolist(),
                    "vertical_displacements_train": vertical_displacements_train.tolist(),
                    "vertical_force_train": vertical_force_train.tolist(),
                    "soil_ID": soil_id,
                    "soil_stiffness": soil_stiff,
                    "soil_damping": soil_damp,
                    "idx_matrix": list(set(idx_matrix[np.isnan(idx_matrix) == False])),
                    "global_stiffness": coupled_model.track.global_stiffness_matrix,
                    "global_damping": coupled_model.track.global_damping_matrix,
                    "global_mass": coupled_model.track.global_mass_matrix,
                    "n_elements_per_sleeper": geometry["n_elements_per_sleeper"],
                    }

    # dump pickle
    pickle_path = os.path.join(output_dir, f'{file_name}.pickle')
    with open(pickle_path, "wb") as f:
        pickle.dump(result_track, f)

    return pickle_path

def run_benchmark():
    """Run benchmark models with different numbers of elements per sleeper"""
    # Define parameters
    nb_sleepers = [150, 150]  # Reduced number for faster benchmarking
    stiffness = [132e6, 132e7]
    damping = [30e3, 30e3]

    # Define multiple element configurations to test
    element_configs = [1, 2, 4, 8]

    # Train parameters
    train_type = TrainType.DOUBLEDEKKER
    train_start_coord = 20
    train_speed = 140/3.6
    running_time = 3  # Reduced for faster benchmarking
    time_step = 0.00025
    use_irregularities = True
    output_time_interval = 7
    output_dir = "./results_element_benchmark"

    # Store file paths for result files
    result_files = []

    # Run simulations for each element configuration
    for n_elements in element_configs:
        print(f"Running simulation with {n_elements} elements per sleeper...")

        # Create geometry with specified number of elements per sleeper
        geom = geometry(nb_sleepers, n_elements_per_sleeper=n_elements)

        # Materials remain the same for all configurations
        mat = materials()

        # Time integration parameters
        tim = time_integration(running_time, time_step)

        # Soil parameters
        soil = soil_parameters(geom["sleeper_distance"], stiffness, damping)

        # Create the model
        coupled_model = create_model(
            train_type, train_start_coord, geom, mat, tim, soil,
            train_speed, use_irregularities, output_time_interval
        )

        # Run the simulation
        coupled_model.main()

        # Save results
        file_name = f"DOUBLEDEKKER_{n_elements}_elements"
        result_file = write_results(coupled_model, output_dir, file_name, geom)
        result_files.append(result_file)

        print(f"Completed simulation with {n_elements} elements per sleeper")

    return result_files

def plot_results(result_files):
    """
    Plot results from different element configurations
    """

    results = []
    for file_path in result_files:
        with open(file_path, "rb") as f:
            results.append(pickle.load(f))

    # Set up plots
    plt.figure(figsize=(15, 10))

    # Plot 1: Vertical rail displacements at specific node indices
    plt.subplot(2, 2, 1)
    for i, result in enumerate(results):
        n_elements = result["n_elements_per_sleeper"]
        # Select a node approximately at the same position for each model
        # For models with multiple elements, adjust the index to get same position
        node_idx = 50 * n_elements // results[0]["n_elements_per_sleeper"]
        plt.plot(result["vert_disp_rail"][node_idx], label=f"{n_elements} elements")

    plt.title("Vertical Rail Displacement (Node 50)")
    plt.xlabel("Time Step")
    plt.ylabel("Displacement [m]")
    plt.legend()
    plt.grid(True)

    # Plot 2: Vertical rail displacements at another node
    plt.subplot(2, 2, 2)
    for i, result in enumerate(results):
        n_elements = result["n_elements_per_sleeper"]
        # Select a node approximately at the same position for each model
        node_idx = 100 * n_elements // results[0]["n_elements_per_sleeper"]
        plt.plot(result["vert_disp_rail"][node_idx], label=f"{n_elements} elements")

    plt.title("Vertical Rail Displacement (Node 100)")
    plt.xlabel("Time Step")
    plt.ylabel("Displacement [m]")
    plt.legend()
    plt.grid(True)

    # Plot 3: Soil displacements at specific indices
    plt.subplot(2, 2, 3)
    for i, result in enumerate(results):
        n_elements = result["n_elements_per_sleeper"]
        plt.plot(result["vertical_displacements_soil"][50], label=f"{n_elements} elements")

    plt.title("Vertical Soil Displacement (Node 50)")
    plt.xlabel("Time Step")
    plt.ylabel("Displacement [m]")
    plt.legend()
    plt.grid(True)

    # Plot 4: Rail displacement profiles at specific time
    plt.subplot(2, 2, 4)
    time_idx = 500  # Choose a relevant time index
    for i, result in enumerate(results):
        n_elements = result["n_elements_per_sleeper"]
        # Extract displacement at the specified time for all nodes
        disp_profile = [disp[time_idx] for disp in result["vert_disp_rail"]]
        # Plot against spatial coordinates
        plt.plot(result["coords_rail"], disp_profile, label=f"{n_elements} elements")

    plt.title(f"Rail Displacement Profile (Time Step {time_idx})")
    plt.xlabel("Position [m]")
    plt.ylabel("Displacement [m]")
    plt.legend()
    plt.grid(True)

    # Adjust layout and save figure
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(result_files[0]), "element_size_comparison.png"))
    plt.show()

def compare_results(result_files):
    """Compare and plot results from different element configurations"""
    # Load result files


    results = []
    for file_path in result_files:
        with open(file_path, "rb") as f:
            results.append(pickle.load(f))

    # Quantitative comparison: Calculate differences between models
    base_result = results[0]  # 1 element per sleeper as baseline

    print("Quantitative comparison with baseline (1 element per sleeper):")
    for i, result in enumerate(results[1:], 1):
        n_elements = result["n_elements_per_sleeper"]

        # Compare at equivalent positions
        node_idx_base = 50
        node_idx_multi = 50 * n_elements

        # Extract displacements at these nodes
        disp_base = base_result["vert_disp_rail"][node_idx_base]
        disp_multi = result["vert_disp_rail"][node_idx_multi]

        # Calculate maximum absolute difference and relative error
        max_diff = np.max(np.abs(np.array(disp_base) - np.array(disp_multi)))
        max_rel_error = max_diff / np.max(np.abs(disp_base)) * 100

        print(f"{n_elements} elements per sleeper:")
        print(f"  Max absolute difference: {max_diff:.6e} m")
        print(f"  Max relative error: {max_rel_error:.2f}%")

        # Check if results are within acceptable tolerance (e.g., 1%)
        assert max_rel_error < 1.0, f"Results for {n_elements} elements per sleeper exceed 1% tolerance."



def test_benchmark_multi_elements():
    """Test function to run the benchmark and compare results"""
    result_files = run_benchmark()

    if SHOW_RESULTS:
        plot_results(result_files)

    compare_results(result_files)
