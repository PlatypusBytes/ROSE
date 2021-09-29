import numpy as np
import json
# import rose packages
from rose.utils import LayeredHalfSpace

def run_wolf_on_layering(layering, omega):

    data = LayeredHalfSpace.Layers(layering)
    data.assign_properties()
    data.correction_incompressible()
    data.static_cone()
    data.dynamic_stiffness(omega)

    return data

def run_wolf(layers_file, omega, output="./", freq=False, plots=True):
    r"""
    Dynamic stiffness according to Wolf and Deeks (2004)

    Infinite layered soil solution
    Only considers the translational cones. The rotational cones are not considered.
    """

    for layers in layers_file:
        print(layers[0])
        data = LayeredHalfSpace.Layers(layers[1])

        data.assign_properties()
        data.correction_incompressible()
        data.static_cone()
        data.dynamic_stiffness(omega)

        LayeredHalfSpace.write_output(output, layers[0], data, omega, freq, plots=plots)

    return data


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


def read_file(file, first_layer):

    force = ['Force', '-', '-', '-', '-', '-', '1', 'V']

    with open(file, "r") as f:
        data = json.load(f)

    layers = []
    # create layers for wolf
    for segment in data:
        for scenario in data[segment]:
            aux = [force, first_layer]
            for j in range(len(data[segment][scenario]["soil_layers"]["soil_name"]) - 1):
                aux.append([data[segment][scenario]["soil_layers"]["soil_name"][j],
                            str(data[segment][scenario]["soil_layers"]["shear_modulus"][j] * 1e6),
                            str(data[segment][scenario]["soil_layers"]["poisson"][j]),
                            str(data[segment][scenario]["soil_layers"]["gamma_wet"][j] * 1000 / 9.81),
                            str(data[segment][scenario]["soil_layers"]["damping"][j]),
                            str(data[segment][scenario]["soil_layers"]["top_level"][j] - data[segment][scenario]["soil_layers"]["top_level"][j + 1]),
                            "-", "-"])

            aux.append([data[segment][scenario]["soil_layers"]["soil_name"][-1],
                        str(data[segment][scenario]["soil_layers"]["shear_modulus"][-1] * 1e6),
                        str(data[segment][scenario]["soil_layers"]["poisson"][-1]),
                        str(data[segment][scenario]["soil_layers"]["gamma_wet"][-1] * 1000 / 9.81),
                        str(data[segment][scenario]["soil_layers"]["damping"][-1]),
                        "inf", "-", "-"])  # add infinite
            layers.append([f"{segment}_{scenario}", aux])
    return layers


if __name__ == "__main__":
    E = 100e6
    v = 0.2
    emb = ["embankment", E / (2 * (1 + v)), v, 2000, 0.05, 1]
    layers = read_file(r"./SOS/SOS.json", emb)
    import time
    t_ini = time.time()
    run_wolf(layers, np.array([8.729139587]), output=r"./wolf/dyn_stiffness", plots=False)
    print(f"Time: {time.time() - t_ini}")
