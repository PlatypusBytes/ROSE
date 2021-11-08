import os
import sys
import copy
import json
import numpy as np
import matplotlib.pylab as plt
import shapefile


class ReadSosScenarios:
    def __init__(self, profile_filename: str, soil_props: str, segment_filename: str, shape_filename: str, depth_ref: float = -20.) -> None:
        """
        Reads the SOS csv files. Plots and creates json file

        :param profile_filename: CSV file with profile
        :param soil_props: CSV file with soil properties
        :param segment_filename: CSV file with segments
        :param shape_filename: shapefile
        :param depth_ref: reference depth to plot the segments (default -20.)
        """

        # open profile csv file
        with open(profile_filename, "r") as f:
            data = f.read().splitlines()
        self.profiles = [d.split(";") for d in data[1:]]

        # open segment csv file
        with open(segment_filename, "r") as f:
            data = f.read().splitlines()
        self.segments = [d.split(";") for d in data[1:]]

        # read shapefile
        self.shape_coord = []
        self.shape_segment = []
        self.read_shape_file(shape_filename)

        # read materials properties
        self.materials = {}
        self.read_materials(soil_props)

        self.SOS = {}  # SOS dictionary

        self.depth_ref = depth_ref  # depth reference for SOS plot

        return

    def read_shape_file(self, file):
        """
        Read shape file for the SOS segments and create index for coordinates

        :param file: shape file
        """

        sf = shapefile.Reader(file)

        for s in sf.shapeRecords():
            self.shape_segment.append(s.record.Segment)
            self.shape_coord.append(np.array(s.shape.points).tolist())
        return

    def read_materials(self, file: str) -> None:
        """
        Reads the CSV file with the materials parameters

        :param file: csv file
        """
        # open props csv file
        with open(file, "r") as f:
            data = f.read().splitlines()
        props = [d.split(";") for d in data[2:]]

        for p in props:

            new_p = []
            for x in p:
                try:
                    new_p.append(float(x))
                except ValueError:
                    new_p.append(x)
                if x == "NaN":
                    new_p[-1] = "NaN"

            self.materials.update({p[0]: {"formation": new_p[1],
                                          "gamma_dry": new_p[3],
                                          "gamma_wet": new_p[4],
                                          "cohesion": new_p[5],
                                          "friction_angle": new_p[6],
                                          "Su": new_p[7],
                                          "m": new_p[9],
                                          "POP": new_p[10],
                                          "shear_modulus": new_p[12],
                                          "Young_modulus": new_p[13],
                                          "poisson": new_p[14],
                                          "a": new_p[15],
                                          "b": new_p[16],
                                          "c": new_p[17],
                                          "damping": new_p[19],
                                          }})
        return

    def create_segments(self):
        """
        Creates the dictionary with the information about the SOS segments
        """
        # collect unique segments
        unique_segments = sorted(set([i[0] for i in self.segments]))

        # for each segment
        for seg in unique_segments:

            # add key to dictionary
            self.SOS.update({f"Segment {seg}": {}})

            # get scenarios for segment
            scenarios = [j for j, x in enumerate(self.segments) if x[0] == seg]

            # find coordinates in shape file
            idx_coord = self.shape_segment.index(int(seg[1:]))
            coords = self.shape_coord[idx_coord]

            materials_keys = []
            for k in self.materials:
                materials_keys.extend(self.materials[k].keys())
            materials_keys = set(materials_keys)

            # for each scenario:
            for i, idx in enumerate(scenarios):
                # collect segment info
                idx_segment = [j for j, x in enumerate(self.profiles) if x[0] == self.segments[idx][1]]

                # soil layers dict dict:
                aux = {"soil_name": [],
                       "top_level": [],
                       }
                for k in materials_keys:
                    aux.update({k: []})

                for j in idx_segment:
                    name = self.profiles[j][2]
                    # ignore this layer
                    if name == "H_Aa_ht_ophoging":
                        continue

                    aux["soil_name"].append(name)
                    aux["top_level"].append(float(self.profiles[j][1]))

                    for k in materials_keys:
                        aux[k].append(self.materials[name][k])

                # update scenario into segment
                self.SOS[f"Segment {seg}"].update({f"scenario {int(i + 1)}":  {"probability": float(self.segments[idx][2]),
                                                                               # "top_level": [float(self.profiles[j][1]) for j in idx_segment],
                                                                               "soil_layers": aux,#[self.profiles[j][2] for j in idx_segment],
                                                                               "coordinates": coords,
                                                                               }
                                                   })
        return

    def dump(self, output_file: str) -> None:
        """
        Dumps result dictionary into a json file

        :param output_file: json output full filename
        """

        # check if output folder exists. creates output if does not exist
        if not os.path.isdir(os.path.split(output_file)[0]):
            os.makedirs(os.path.split(output_file)[0])

        # check if file extension is json
        if not os.path.split(output_file)[1].endswith(".json"):
            sys.exit(f"Error: file name {output_file} must be json file")

        # dump json
        with open(output_file, "w") as f:
            json.dump(self.SOS, f, indent=2)

        return

    def plot_sos(self, output_folder: str = "./results",
                 colour_file: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "SOS_colour_code.json")) -> None:
        """
        Creates the plot for each SOS

        :param output_folder: path to the location of the file (optional "./results")
        :param colour_file: path to the colour_file scheme of each soil layer (default "./SOS_colour_code.json")
        """

        # create the results folder if does not exist
        if not os.path.isdir(output_folder):
            os.makedirs(output_folder)

        # open json file for colour SOS
        with open(colour_file, "r") as f:
            colour = json.load(f)

        # for each segment
        for seg in self.SOS:
            # number of scenarios
            nb_sce = len(self.SOS[seg])

            # create figure
            fig, ax = plt.subplots(1, 2, figsize=(8, 6))
            plt.rcParams.update({'font.size': 10})
            ax[0].set_position([0.1, 0.22, 0.5, 0.7])
            ax[1].set_position([0.65, 0.14, 0.3, 0.8])

            nam = []
            probs = []
            for i, sce in enumerate(self.SOS[seg]):
                # get depths
                depth = copy.deepcopy(self.SOS[seg][sce]["soil_layers"]["top_level"])
                depth.append(self.depth_ref)
                name_layer = self.SOS[seg][sce]["soil_layers"]["soil_name"]

                probs.append(f'{i + 1}\nProb: {self.SOS[seg][sce]["probability"]}')

                for j in range(len(self.SOS[seg][sce]["soil_layers"]['top_level'])):
                    # collect soil type
                    nam.append(self.SOS[seg][sce]["soil_layers"]['soil_name'][j])
                    ax[0].fill_betweenx([depth[j], depth[j + 1]],
                                        np.zeros(2) + i * 1 + 0.75,
                                        np.zeros(2) + i * 1 + 1.25,
                                        facecolor=list(map(lambda x: x / 255, colour[name_layer[j]])))

            # define unique for plotting of the legend
            nam = list(set(nam))

            # for the legend of the colors
            for i in range(len(nam)):
                ax[1].fill_betweenx([1 + (i + 1), 1.5 + (i + 1)],
                                    0,
                                    0.75,
                                    facecolor=list(map(lambda x: x / 255, colour[nam[i]])))
                ax[1].text(1., 1.05 + (i + 1), nam[i], fontsize=14)

            # settings for the colors
            ax[1].set_xlim(0, 3)
            ax[1].set_ylim(0, 20)
            ax[1].axis("off")

            # settings for the soil
            ax[0].grid()
            ax[0].set_ylabel("Depth NAP [m]", fontsize=14)
            ax[0].set_xlabel("Scenario [%]", fontsize=14)
            ax[0].set_xlim(0, nb_sce + 1)
            ax[0].set_ylim(bottom=self.depth_ref)
            ax[0].set_xticks(range(1, nb_sce + 1))
            ax[0].set_xticklabels(probs, rotation=45, fontsize=12)
            plt.savefig(os.path.join(output_folder, f"{seg}.png"))
            plt.close()

        return

    @staticmethod
    def plot_highlighted_sos(sos_dict, highlighted_segment_name, fig=None,position=111):
        """
        Plots the coordinates of all the SOS segments and highlights the chosen SOS segment

        :param sos_dict: dictionary of all the sos data
        :param highlighted_segment_name: name of the sos segment to be highlighted
        :param fig:  optional existing figure
        :param position: position of subplot, default: 111
        :return:
        """

        # initialise figure if it is not an input
        if fig is None:
            fig = plt.figure()
        ax = fig.add_subplot(position)

        # initialise highlighted coordinates
        highlighted_coordinates = np.array([])

        # loop over sos segments
        for name, segment in sos_dict.items():

            if name == highlighted_segment_name:
                # get coordinates of the to be highlighted sos segments
                highlighted_coordinates = np.array(list(segment.values())[0]['coordinates'])
            else:

                # get and plot coordinates of the additional sos segments
                coordinates = np.array(list(segment.values())[0]['coordinates'])
                ax.plot(coordinates[:, 0], coordinates[:, 1], color='k')

        # plot highlighted coordinates, if coordinates are found. This plot is added last, such that it is plotted on
        # top.
        if highlighted_coordinates.size > 0:
            ax.plot(highlighted_coordinates[:, 0], highlighted_coordinates[:, 1], color='r', linewidth=5)

        # set labels
        ax.set_xlabel("x-coordinates [m]")
        ax.set_ylabel("y-coordinates [m]")

        return fig, ax


if __name__ == "__main__":
    sos = ReadSosScenarios(
        r"N:\Projects\11203500\11203592\B. Measurements and calculations\TKI Data\Output\soilprofiles.csv",
        r"N:\Projects\11203500\11203592\B. Measurements and calculations\TKI Data\Properties\20201102_Prorail_parameters_SOS.csv",
        r"N:\Projects\11203500\11203592\B. Measurements and calculations\TKI Data\Output\segments.csv",
        r"N:\Projects\11204500\11204953\B. Measurements and calculations\SOS\TKI Data\Segmenten\Segments_TKI_v2.shp")
    sos.create_segments()
    sos.dump("./SOS.json")
    sos.plot_sos(output_folder="./SOS/results")
