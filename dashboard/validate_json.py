from schema import Schema, And, Use, SchemaError


def definitions():
    """
    Defines the configuration schema for the validation of the input json file

    @return: Schema configuration file
    """

    # sos data dictionary
    conf_schema_sos = Schema({
        "coordinates": And([[float], [float]]),
        "scenarios": {And(str): {"probability": And(Use(float)),
                                 "soil_layers": {"soil_name": And([str]),
                                                 "top_level": And([float, int]),
                                                 "shear_modulus": And([float, int]),
                                                 "formation": And([str]),
                                                 "gamma_dry": And([float, int]),
                                                 "Su": And([float, int]),
                                                 "Young_modulus": And([float, int]),
                                                 "poisson": And([float, int]),
                                                 "c": And([float, int]),
                                                 "damping": And([float, int]),
                                                 "m": And([float, int]),
                                                 "gamma_wet": And([float, int]),
                                                 "a": And([float, int]),
                                                 "friction_angle": And([float, int]),
                                                 "b": And([float, int]),
                                                 "POP": And([float, int]),
                                                 "cohesion": And([float, int]),
                                                 }
                                 },
                      },
    })

    # main scheme
    conf_schema = Schema({
                          "sos_data": {And(str): conf_schema_sos},
                          "traffic_data": {And(str): {
                                                      "wheel_distances": Schema([float, int]),
                                                      "bogie_length": And(Use(float)),
                                                      "bogie_distances": Schema([float, int]),
                                                      "cart_length": And(Use(float)),
                                                      "cart_distances": Schema([float, int]),
                                                      "mass_wheel": And(Use(float)),
                                                      "mass_bogie": And(Use(float)),
                                                      "mass_cart": And(Use(float)),
                                                      "inertia_bogie": And(Use(float)),
                                                      "inertia_cart": And(Use(float)),
                                                      "prim_stiffness": And(Use(float)),
                                                      "sec_stiffness": And(Use(float)),
                                                      "prim_damping": And(Use(float)),
                                                      "sec_damping": And(Use(float)),
                                                      "velocity": And(Use(float)),
                                                      "type": And(Use(str))
                                                      },
                          },
        'track_info': {"geometry": {"n_segments": And(Use(int)),
                                    "n_sleepers": Schema([int]),
                                    "sleeper_distance": And(Use(float)),
                                    "depth_soil": Schema([float, int]),
                                    },
                       "materials": {"young_mod_beam": And(Use(float)),
                                     "poisson_beam": And(Use(float)),
                                     "inertia_beam": And(Use(float)),
                                     "rho": And(Use(float)),
                                     "rail_area": And(Use(float)),
                                     "shear_factor_rail": And(Use(float)),
                                     "damping_ratio": And(Use(float)),
                                     "omega_one": And(Use(float)),
                                     "omega_two": And(Use(float)),
                                     "mass_rail_pad": And(Use(float)),
                                     "stiffness_rail_pad": And(Use(float)),
                                     "damping_rail_pad": And(Use(float)),
                                     "mass_sleeper": And(Use(float)),
                                     "hertzian_contact_coef": And(Use(float)),
                                     "hertzian_power": And(Use(float))
                                     }
                       },
        "time_integration": {"tot_ini_time": And(Use(float)),
                             "n_t_ini": And(Use(int)),
                             "tot_calc_time": And(Use(float)),
                             "n_t_calc": And(Use(int)),
                             }
    })

    return conf_schema


def check_json(input_json):
    """
    Check if the input json file is correct

    @param input_json: input json file
    @return: True / False
    """
    conf_schema = definitions()
    try:
        conf_schema.validate(input_json)
        return True
    except SchemaError:
        return False


if __name__ == "__main__":
    # Example run
    import json
    with open("../run_rose/example_rose_input.json", "r") as fi:
        input = json.load(fi)

    print(check_json(input))
