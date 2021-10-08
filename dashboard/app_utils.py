import copy
import os
import pickle
import json

from pyproj import Transformer
# import ROSE packages
from rose.model.model_part import Material, Section
from rose.model.train_model import *
from rose.model.train_track_interaction import *
from rose.model import Varandas
from solvers.newmark_solver import NewmarkSolver


def transform_rd_to_lat_lon(rd_x, rd_y):
    """
    Converts rd coordinates to lat lon coordinates
    :param rd_x: rd x coordinates
    :param rd_y: rd y coordinates
    :return:
    """
    transformer = Transformer.from_crs("epsg:28992", "epsg:4326")
    x, y = transformer.transform(rd_x, rd_y)
    return x,y

def calculate_weighted_mean_and_std(values: np.ndarray, weights: np.ndarray) -> [np.ndarray, np.ndarray]:
    """
    Return the weighted average and standard deviation.

    @param values: data values
    @param weights: weights of the data values
    @return: mean, standard deviation
    """

    average = np.average(values, weights=weights, axis=0)
    variance = np.average((values-average) ** 2, weights=weights, axis=0)  # Fast and numerically precise

    return average, np.sqrt(variance)

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
    """
    Gets vertical force in soil at the middle of the coupled model
    :param coupled_model:
    :param output_interval:
    :return:
    """

    # get vertical force in soil
    vertical_force_soil = np.array(
        [element.force[0::output_interval, 0] for element in coupled_model.track.model_parts[4].elements])

    # get middle soil index
    mid_idx_force = int(vertical_force_soil.shape[0]/2)
    time = coupled_model.time[0::output_interval]

    # return time and mid vertical soil force
    return time, vertical_force_soil[mid_idx_force,:]

def write_geo_json(features, filename: str):
    """
    Creates and writes geojson dict to json file
    :param features: all features in the geojson
    :param filename: output filename
    :return:
    """

    # find limits of dynamic stiffness and cumulative settlement, and find all train types
    all_train_types = set()
    min_sett, max_sett = 1e10, -1e10
    min_stiff, max_stiff = 1e10, -1e10
    for feature in features:
        min_sett = min(min_sett, min(feature["properties"]["cumulative_settlement_mean"]))
        max_sett = max(max_sett, max(feature["properties"]["cumulative_settlement_mean"]))
        min_stiff = min(min_stiff, min(min(feature["properties"]["mean_dyn_stiffness"])))
        max_stiff = max(max_stiff, max(max(feature["properties"]["mean_dyn_stiffness"])))
        all_train_types.update(feature["properties"]["train_names"])

    all_train_types = list(all_train_types)

    # set limit per colour code
    colours = ["#691aff", "#b81010", "#ffdb1a", "#6d1046", "#000066"]
    cumulative_settlement_limits = np.linspace(min_sett,max_sett, len(colours))
    dyn_stiffness_limits = np.linspace(min_stiff,max_stiff, len(colours))

    # generate geojson dict
    geo_json = {"colours": colours,
                "cumulative_sett_limits": np.around(cumulative_settlement_limits, decimals=2).tolist(),
                "dyn_stiff_limits": np.around(dyn_stiffness_limits, decimals=2).tolist(),
                "all_train_types": all_train_types,
                "geojson": {"type": "FeatureCollection",
                            "features": features}
                }

    # write geojson
    with open(filename, "w") as json_file:
        json.dump(geo_json, json_file, indent=2)


def add_feature_to_geo_json(coordinates, time, mean_dyn_stiffness,std_dyn_stiffness,train_names, mean_cum_settlement,
                            std_cum_settlement):
    """
    Adds a feature of a single SOS segment to a geojson feature dict. The feature includes lat-lon coordinates;
    mean and std dynamic stiffness; mean and std cumulative settlement; train types; time steps
    :param coordinates:
    :param time:
    :param mean_dyn_stiffness:
    :param std_dyn_stiffness:
    :param train_names:
    :param mean_cum_settlement:
    :param std_cum_settlement:
    :return:
    """

    # convert coordinates to lat lon coordinates
    np_coords = np.array(coordinates)
    lat, lon = transform_rd_to_lat_lon(np_coords[:,0], np_coords[:,1])
    new_coordinates = np.array([lat, lon]).T.tolist()

    # create feature dict
    feature = {"type": "Feature",
                     "geometry": {
                         "type": "LineString",
                         "coordinates": new_coordinates
                     },
                     "properties":{
                         "mean_dyn_stiffness": np.around(mean_dyn_stiffness, decimals=2).tolist(),
                         "std_dyn_stiffness": np.around(std_dyn_stiffness, decimals=2).tolist(),
                         "train_names": train_names,
                         "time": np.around(time,decimals=2).tolist(),
                         "cumulative_settlement_mean": np.around(mean_cum_settlement,decimals=2).tolist(),
                         "cumulative_settlement_std": np.around(std_cum_settlement,decimals=2).tolist()
                     }}
    return feature


def runner(json_input, path_results, calculation_time=50):
    """
    Runs the complete ROSE calculation: Firstly stiffness and damping of the soil are determined with wolf; secondly
    the coupled dynamic train track interaction model is ran; lastly the cumulative settlement model is ran.

    :param json_input:
    :param path_results:
    :param calculation_time: time of the cumulative settlement calculation [days]
    :return:
    """
    from run_rose.run_wolf import create_layering_for_wolf, run_wolf_on_layering

    # retrieve data from input file
    with open(json_input,'r') as f:
        input_data = json.load(f)

    # check if path to save results exist
    print(os.path.join(path_results, input_data["project_name"]))
    if not os.path.isdir(os.path.join(path_results, input_data["project_name"])):
        os.makedirs(os.path.join(path_results, input_data["project_name"]))

    sos_data = input_data["sos_data"]
    traffic_data =input_data["traffic_data"]

    # set embankment data
    E = 100e6
    v = 0.2
    emb = ["embankment", E / (2 * (1 + v)), v, 2000, 0.05, 1]

    features = []
    # loop over segments
    for k, v in sos_data.items():
        # loop over trains
        forces = []
        mean_dynamic_stiffnesses = []
        std_dynamic_stiffnesses = []
        train_types = []
        train_dicts = {}
        scenario_probabilities = [scenario["probability"] for scenario in v["scenarios"].values()]
        # loop over train types
        for train in traffic_data.values():
            train_types.append(train["type"])
            vertical_force_soil_segment = []
            dyn_stiffnesses = []
            # loop over scenarios
            for k2, v2 in v["scenarios"].items():

                # get wolf data
                layering = create_layering_for_wolf(v2["soil_layers"], emb)
                omega = 2*np.pi * train["velocity"]/ train["cart_length"]
                wolf_data = run_wolf_on_layering(layering,np.array([omega]))

                # determine dynamic soil stiffness and damping
                sleeper_dist = input_data["track_info"]["geometry"]["sleeper_distance"]
                dyn_stiffness = np.real(wolf_data.K_dyn) * sleeper_dist
                damping = np.imag(wolf_data.K_dyn) / omega * sleeper_dist
                soil = {"stiffness": dyn_stiffness,
                        "damping": damping}

                dyn_stiffnesses.append(dyn_stiffness)

                # assign data to coupled model
                coupled_model = assign_data_to_coupled_model(train,input_data["track_info"], input_data["time_integration"], soil)

                # run coupled model
                coupled_model.main()

                # get results from coupled model
                time,vertical_force_soil_scenario = get_results_coupled_model(coupled_model, 10)
                vertical_force_soil_segment.append(vertical_force_soil_scenario)

            train_dicts[train["type"]] = train["traffic"]

            #todo change with cumulative settlement

            # calculate mean and std of force of current train
            vertical_force_soil_segment = np.array(vertical_force_soil_segment)
            forces.append(vertical_force_soil_segment)

            mean_dynamic_stiffness, std_dynamic_stiffness = calculate_weighted_mean_and_std(np.array(dyn_stiffnesses), np.array(scenario_probabilities))
            mean_dynamic_stiffnesses.append(list(mean_dynamic_stiffness))
            std_dynamic_stiffnesses.append(list(std_dynamic_stiffness))

        # calculate cumulative settlements per scenario
        cumulative_settlements = []
        for i, scenario in enumerate(v["scenarios"].values()):

            # get forces in soil all trains at current scenario
            for j, train in enumerate(train_dicts.values()):
                train["forces"] = forces[j][i,:][None,:]

            # calculate cumulative settlement
            sett = Varandas.AccumulationModel()
            sett.read_traffic(train_dicts, calculation_time)
            sett.settlement(idx=[0])

            # calculate output step size (1 outout value per day + last value
            n_steps = len(sett.results["time"])
            n_days = sett.results["time"][-1] - sett.results["time"][0]
            step_size = n_steps/n_days

            # get output indices
            indices = [int(day*step_size) for day in range(int(n_days))]
            # add last index
            indices.append(n_steps-1)

            # get cumulative settlement result
            cumulative_time = np.array(sett.results["time"])[indices]
            cumulative_settlements.append(np.array(sett.results["settlement"]['0'])[indices])

        # calculate mean and std cumulative settlement in mm
        m_to_mm = 1e3
        cumulative_settlement_mean,cumulative_settlement_std =  calculate_weighted_mean_and_std(
            np.array(cumulative_settlements)*m_to_mm, np.array(scenario_probabilities))

        # add feature to geo json
        feature = add_feature_to_geo_json(v["coordinates"], cumulative_time, mean_dynamic_stiffnesses,
                                          std_dynamic_stiffnesses, train_types, cumulative_settlement_mean,
                                          cumulative_settlement_std)

        features.append(feature)

    # write geo_json
    write_geo_json(features, os.path.join(path_results, input_data["project_name"], "data.json"))
