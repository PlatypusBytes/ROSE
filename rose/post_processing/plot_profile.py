import os
import json
from typing import List
import numpy as np
import matplotlib.pylab as plt


# settings for the plot
color = ["k", "b", "r"]
style = ["-", ":", "--"]
width = [1, 2, 1.5]
color_shad = [[0.5, 0.5, 0.5], "b", "r"]
color_alpha = [0.25, 0.15, 0.15]
font_size = 10

def compute_distance(coord_1, coord_2):
    return np.sqrt(np.power(coord_1[0]-coord_2[0],2) + np.power(coord_1[1]-coord_2[1],2))

def compute_cumulative_distance(coords):
    """
    computes cumulative distance of the track

    :param coords: array of sorted coordinates
    :return:
    """
    np_coords = np.array(coords)
    diffs = np.diff(np_coords,axis=0)
    distance = np.cumsum(np.sqrt(np.power(diffs[:,0],2) + np.power(diffs[:,1],2)))

    distance = np.concatenate([[0],distance])

    return distance

def weighted_avg_and_std(values: np.ndarray, weights: np.ndarray) -> List[np.ndarray]:
    """
    Return the weighted average and standard deviation.

    :param values: data values
    :param weights: weights of the data values
    :return: mean, standard deviation
    """

    average = np.average(values, weights=weights)
    variance = np.average((values-average) ** 2, weights=weights)  # Fast and numerically precise

    return average, np.sqrt(variance)


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx], idx

def plot_profile(data_list: list, time: float, data_type: list,
                 fct: float = 1.96,
                 xlabel: str = "Distance [km]", ylabel: str = "Value [-]",
                 xlim: list = None, ylim: list = None,
                 output_file: str = "./result.png", are_coords_inverted: bool = True) -> None:
    """

    :param label:
    :param data_list:
    :param time:
    :param data_type:
    :param fct:
    :param xlabel:
    :param ylabel:
    :param xlim:
    :param ylim:
    :param output_file:
    :return:
    """

    # check if folder exists if not create
    if not os.path.isdir(os.path.dirname(output_file)):
        os.makedirs(os.path.dirname(output_file))

    # create figure
    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    ax.set_position([0.15, 0.15, 0.8, 0.8])
    plt.rcParams.update({'font.size': 10})

    # iterate over all the data lists
    for idx, data in enumerate(data_list):
        # find time index
        index_time = np.argmin(np.abs(np.array(data["time"]) - time))

        seg_mean = []
        seg_std = []
        coords = []

        end_coords =[]

        for seg in data["data"]:
            mean_val = []
            prob_val = []
            for sce in data["data"][seg]:
                mean_val.append(data["data"][seg][sce]["data"][index_time])
                prob_val.append(data["data"][seg][sce]["probability"])

            res = weighted_avg_and_std(np.array(mean_val), np.array(prob_val))

            # append to results mean, std and coordinates
            seg_mean.extend(np.ones(len(data["coordinates"][seg])) * res[0])
            seg_std.extend(np.ones(len(data["coordinates"][seg])) * res[1])

            if coords:
                end_coords = [coords[0], coords[-1]]

            seg_coords = np.array(data["coordinates"][seg])

            # sort coordinate arrays such that the coordinates are connected
            closest_idx = 0
            if end_coords:
                dist_first_first = compute_distance(seg_coords[0], end_coords[0])
                dist_first_last = compute_distance(seg_coords[0], end_coords[1])
                dist_last_first = compute_distance(seg_coords[1], end_coords[0])
                dist_last_last = compute_distance(seg_coords[1], end_coords[1])

                closest_idx = np.array([dist_first_first,dist_first_last,dist_last_first,dist_last_last]).argmin()

            if closest_idx ==0:
                coords = list(np.flip(coords,axis=0))
                coords.extend(data["coordinates"][seg])
            elif closest_idx == 1:
                coords.extend(data["coordinates"][seg])
            elif closest_idx ==2:
                coords = list(np.flip(coords,axis=0))
                coords.extend(list(np.flip(data["coordinates"][seg],axis=0)))
            else:
                coords.extend(list(np.flip(data["coordinates"][seg],axis=0)))


        # compute distance
        dist = compute_cumulative_distance(coords)

        # plot for data list
        ax.plot(dist, seg_mean, color=color[idx], linewidth=width[idx], linestyle=style[idx], label=data_type[idx])
        ax.fill_between(dist,
                        np.array(seg_mean) - fct * np.array(seg_std),
                        np.array(seg_mean) + fct * np.array(seg_std),
                        color=color_shad[idx], alpha=color_alpha[idx])
    ax.grid()
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel(xlabel, fontsize=font_size)
    ax.set_ylabel(ylabel, fontsize=font_size)
    leg = ax.legend(loc=0, fontsize=font_size - 1)
    leg.get_frame().set_edgecolor("k")
    leg.get_frame().set_linewidth(0.25)

    plt.savefig(output_file)
    plt.close()

    return


if __name__ == "__main__":

    time = [0, 1, 2, 3, 4, 5]
    # data = {"time": time,
    #         "data": {"Segment 1": {"Scenario 1": {"data": np.random.normal(5, 1, len(time)).tolist(),
    #                                               "probability": 0.5},
    #                                "Scenario 2": {"data": np.random.normal(6, .5, len(time)).tolist(),
    #                                               "probability": 0.3},
    #                                "Scenario 3": {"data": np.random.normal(4, .3, len(time)).tolist(),
    #                                               "probability": 0.2},
    #                                },
    #                  "Segment 2": {"Scenario 1": {"data": np.random.normal(10, 1, len(time)).tolist(),
    #                                               "probability": 0.75},
    #                                "Scenario 2": {"data": np.random.normal(12, 1, len(time)).tolist(),
    #                                               "probability": 0.25},
    #                                },
    #                  "Segment 3": {"Scenario 1": {"data": np.random.normal(1, 1, len(time)).tolist(),
    #                                               "probability": 0.8},
    #                                "Scenario 2": {"data": np.random.normal(2, 1, len(time)).tolist(),
    #                                               "probability": 0.15},
    #                                "Scenario 3": {"data": np.random.normal(3, 1, len(time)).tolist(),
    #                                               "probability": 0.05},
    #                                },
    #                  },
    #         "coordinates": {"Segment 1": [[0, 0], [1, 1], [2, 2]],
    #                         "Segment 2": [[2, 2], [3, 3], [5, 5]],
    #                         "Segment 3": [[5, 5], [10, 10]],
    #                         }}

    with open(r"../batch_results/intercity/dyn_stiffness_profile.json", "r") as f:
        data = json.load(f)


    new_data = [data, data]

    # plot_profile([data], 3, ["case 1"], xlabel="Distance [m]", ylabel="Dynamic stiffness", xlim=[0, 20], ylim=[0, 15], output_file="./folder/results.png")
    # plot_profile(new_data, 3, ["case 1", "case 2"], xlabel="Distance [m]", ylabel="Dynamic stiffness", xlim=[0, 20], ylim=[0, 15], output_file="./folder/results.png")

    plot_profile([data], 3, ["case 1"], xlabel="Distance [m]", ylabel="Dynamic stiffness", output_file="./folder/results_tmp.png")
