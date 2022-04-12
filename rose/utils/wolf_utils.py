# import wolf package
from WolfStiffness.wolfStiffness import WolfStiffness


def run_wolf_on_layering(layering, omega):

    wolf = WolfStiffness(omega, output_folder="./")
    wolf.layers = layering
    wolf.compute()
    # wolf.write(plot=plots, freq=True)
    return wolf.data


def create_layering_for_wolf(sos_layering_data, first_layer):
    """
    Creates layering for Wolf

    :param sos_layering_data: SOS layering data for 1 scenario
    :param first_layer:
    :return:
    """
    force = ['Force', '-', '-', '-', '-', '-', '1', 'V']
    aux = [force, first_layer]
    for j in range(len(sos_layering_data["soil_name"]) - 1):
        aux.append([sos_layering_data["soil_name"][j],
                    str(sos_layering_data["shear_modulus"][j] * 1e6),
                    str(sos_layering_data["poisson"][j]),
                    str(sos_layering_data["gamma_wet"][j] * 1000 / 9.81),
                    str(sos_layering_data["damping"][j]),
                    str(sos_layering_data["top_level"][j] -
                        sos_layering_data["top_level"][j + 1]),
                    "-", "-"])

    aux.append([sos_layering_data["soil_name"][-1],
                str(sos_layering_data["shear_modulus"][-1] * 1e6),
                str(sos_layering_data["poisson"][-1]),
                str(sos_layering_data["gamma_wet"][-1] * 1000 / 9.81),
                str(sos_layering_data["damping"][-1]),
                "inf", "-", "-"])  # add infinite

    return aux