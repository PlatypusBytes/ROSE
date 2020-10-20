import os
import sys
import copy
import json
import numpy as np
import matplotlib.pylab as plt
import shapefile


class ReadSosScenarios:
    def __init__(self, profile_filename: str, segment_filename: str, shape_filename: str, depth_ref: float = -20.) -> None:
        """
        Reads the SOS csv files. Plots and creates json file

        :param profile_filename: CSV file with profile
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

            # for each scenario:
            for i, idx in enumerate(scenarios):
                # collect segment info
                idx_segment = [j for j, x in enumerate(self.profiles) if x[0] == self.segments[idx][1]]

                # update scenario into segment
                self.SOS[f"Segment {seg}"].update({f"scenario {int(i + 1)}":  {"probability": float(self.segments[idx][2]),
                                                                               "top_level": [float(self.profiles[j][1]) for j in idx_segment],
                                                                               "soil_name": [self.profiles[j][2] for j in idx_segment],
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
                 colour_file: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "./SOS_colour_code.json")) -> None:
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
            fig, ax = plt.subplots(1, 2, figsize=(10, 6))
            plt.rcParams.update({'font.size': 10})
            ax[0].set_position([0.1, 0.22, 0.5, 0.7])
            ax[1].set_position([0.65, 0.14, 0.3, 0.8])

            nam = []
            probs = []
            for i, sce in enumerate(self.SOS[seg]):
                # get depths
                depth = copy.deepcopy(self.SOS[seg][sce]["top_level"])
                depth.append(self.depth_ref)
                name_layer = self.SOS[seg][sce]["soil_name"]

                probs.append(f'{i + 1}\nProb: {self.SOS[seg][sce]["probability"]}')

                for j in range(len(self.SOS[seg][sce]['top_level'])):
                    # collect soil type
                    nam.append(self.SOS[seg][sce]['soil_name'][j])
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
