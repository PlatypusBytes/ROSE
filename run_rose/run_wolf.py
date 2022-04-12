import numpy as np
import json
# import wolf package
from WolfStiffness.wolfStiffness import WolfStiffness


def run_wolf(layers_file, omega, output="./", plots=True):
    r"""
    Dynamic stiffness according to Wolf and Deeks (2004)

    Infinite layered soil solution
    Only considers the translational cones. The rotational cones are not considered.
    """

    for layers in layers_file:
        print(layers[0])

        wolf = WolfStiffness(omega, output_folder=output)
        wolf.name = layers[0]
        wolf.layers = layers[1]
        wolf.compute()
        wolf.write(plot=plots, freq=True)


def read_file(file, first_layer):
    """
    reads json SOS file and add first layer.

    Returns a data structure that can be used in WolfStiffness
    """
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
    layers = read_file(r"../data_proc/SOS.json", emb)
    import time
    t_ini = time.time()
    run_wolf(layers, np.array([8.729139587]), output=r"./wolf/dyn_stiffness", plots=False)
    print(f"Time: {time.time() - t_ini}")
