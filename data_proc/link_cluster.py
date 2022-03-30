import os
import json

import numpy as np
import matplotlib.pylab as plt
from scipy.spatial import KDTree

from data_proc.ricardo import load_inframon_data, get_data_within_bounds
from data_proc.fugro import load_rila_data, interpolate_coordinates, calculate_d_values
from SignalProcessing import window, signal_tools


CLR = {0: {"color": "b",
           "thickness": 2},
       1: {"color": "r",
           "thickness": 2},
       2: {"color": "tab:olive",
           "thickness": 1},
       3: {"color": "lightblue",
           "thickness": 1},
       4: {"color": "navy",
           "thickness": 2},
       5: {"color": "y",
           "thickness": 2},
       6: {"color": "0.2",
           "thickness": 2},
       7: {"color": "0.4",
           "thickness": 2},
       8: {"color": "0.6",
           "thickness": 2},
       9: {"color": "0.8",
           "thickness": 1},
       }


# settings_filter = {"FS": 250,
#                    "cut-off": 120,
#                    "n": 10,
#                    "wavelength": 10}

FS = 250

def plot_seg(freq, ampl, freq_tot, ampl_tot, scenario, name, output_path):

    # if output path does not exist -> creates
    if not os.path.isdir(output_path):
        os.mkdir(output_path)

    fig, ax = plt.subplots(2, 1, figsize=(8, 5))
    ax[0].set_position([0.12, 0.55, 0.7, 0.40])
    ax[1].set_position([0.12, 0.10, 0.7, 0.40])

    for j in range(len(scenario)):
        ax[0].plot(freq[j], np.mean(np.abs(ampl[j]), axis=1),
                   color=CLR[j]["color"], linewidth=CLR[j]["thickness"], label=f"Sce {str(j)}")

        ax[0].fill_between(freq[j],
                           np.mean(np.abs(ampl[j]), axis=1) + np.std(np.abs(ampl[j]), axis=1),
                           np.mean(np.abs(ampl[j]), axis=1) - np.std(np.abs(ampl[j]), axis=1),
                           color=CLR[j]["color"], alpha=0.25, linewidth=0)

    ax[1].plot(freq_tot, ampl_tot, color="b", linewidth=CLR[j]["thickness"])

    fig.suptitle(f"{name}")

    # sort labels
    ax[0].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax[0].grid()
    ax[1].grid()
    ax[0].set_xlim(0, 120)
    ax[0].set_ylim(bottom=0)
    ax[1].set_xlim(0, 120)
    ax[1].set_ylim(bottom=0)
    ax[0].set_xticklabels([])
    # ax[0].set_xlabel("Frequency")
    ax[1].set_xlabel("Frequency")
    ax[0].set_ylabel("Amplitude")
    ax[1].set_ylabel("Amplitude")
    plt.savefig(os.path.join(output_path, f"fft_{name}.png"))
    plt.close()

    return


def fug(sos_path, cluster_path, rila_path, output_path):

    with open(sos_path, 'r') as f:
        sos_data = json.load(f)

    with open(cluster_path, "r") as fi:
        clusters = json.load(fi)

    fugro_data = load_rila_data(rila_path)

    for name, segment in sos_data.items():

        # skip segments where sensar data does not exists
        if not all([int(name.split(" ")[1]) >= 1029, int(name.split(" ")[1]) <= 1080]):
            continue

        # get coordinates of current segments
        coordinates = np.array(list(segment.values())[0]['coordinates'])

        # get coordinate limits
        xlim = [min(coordinates[:, 0]), max(coordinates[:, 0])]
        ylim = [min(coordinates[:, 1]), max(coordinates[:, 1])]

        # get rila data
        dates, coordinate_data, interpolated_heights = interpolate_coordinates(fugro_data, xlim, ylim)
        settlement = np.subtract(interpolated_heights, interpolated_heights[0, :]) * 1000

        # ToDo
        d1, d2, d3 = [], [], []
        for i in range(len(fugro_data)):
            d1_, d2_, d3_ = calculate_d_values(fugro_data.data[i], coordinate_data[i])
            d1.append(d1_)
            d2.append(d2_)
            d3.append(d3_)

        # cluster coordinates
        tree = KDTree(clusters[name]["coordinates"])
        _, idx = tree.query(coordinate_data[0])

        fugro_cluster = []
        for j in range(len(dates)):
            fugro_cluster.append([clusters[name]["scenario"][i] for i in idx])

        uniq_cluster = sorted(set([j for i in fugro_cluster for j in i]))

        plot_fugro(dates, settlement, fugro_cluster, uniq_cluster, f"set_{name}", output_path)
        plot_fugro(dates, d1, fugro_cluster, uniq_cluster, f"d1_{name}", output_path)
        plot_fugro(dates, d2, fugro_cluster, uniq_cluster, f"d2_{name}", output_path)
        plot_fugro(dates, d3, fugro_cluster, uniq_cluster, f"d3_{name}", output_path)

    return


def plot_fugro(date, set, cluster, scenario, name, output_path):

    # if output path does not exist -> creates
    if not os.path.isdir(output_path):
        os.mkdir(output_path)

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.set_position([0.12, 0.12, 0.8, 0.8])

    # for each scenario
    for j in range(len(scenario)):
        mean_set = []
        std_set = []
        for i in range(len(date)):
            idx = np.where(np.array(cluster[i]) == scenario[j])[0]
            settlement = set[i][idx]
            ax.plot([date[i]]*len(settlement), settlement,
                    color=CLR[j]["color"], linewidth=0, marker="o", markerfacecolor=CLR[j]["color"], markersize=2, alpha=0.5)
            mean_set.append(np.nanmean(settlement))
            std_set.append(np.nanstd(settlement))

        ax.plot(date, mean_set,
                color=CLR[j]["color"], linewidth=CLR[j]["thickness"], label=str(j))
        ax.plot(date, np.array(mean_set) + np.array(std_set),
                color=CLR[j]["color"], linewidth=CLR[j]["thickness"] - 1, linestyle=':')
        ax.plot(date, np.array(mean_set) - np.array(std_set),
                color=CLR[j]["color"], linewidth=CLR[j]["thickness"] - 1, linestyle=':')

    fig.suptitle(f"{name}")

    # sort labels
    ax.legend()
    ax.grid()
    # ax.set_xlim(0, 125)
    ax.set_xlabel("Date")
    ax.set_ylabel("Settlement [mm]")
    plt.savefig(os.path.join(output_path, f"{name}.png"))
    plt.close()

    return


def ric(sos_path, cluster_path, inframon_path, output_path):

    with open(sos_path, 'r') as f:
        sos_data = json.load(f)

    with open(cluster_path, "r") as fi:
        clusters = json.load(fi)

    ricardo_data = load_inframon_data(inframon_path)

    for name, segment in sos_data.items():

        # skip segments where sensar data does not exists
        if not all([int(name.split(" ")[1]) >= 1029, int(name.split(" ")[1]) <= 1080]):
            continue

        # get coordinates of current segments
        coordinates = np.array(list(segment.values())[0]['coordinates'])

        # get coordinate limits
        xlim = [min(coordinates[:, 0]), max(coordinates[:, 0])]
        ylim = [min(coordinates[:, 1]), max(coordinates[:, 1])]

        # get ricardo data
        ricardo_data_within_bounds = get_data_within_bounds(ricardo_data["Jan"], xlim, ylim)

        # cluster coordinates
        tree = KDTree(clusters[name]["coordinates"])
        _, idx = tree.query(ricardo_data_within_bounds["coordinates"])

        ricardo_cluster = [clusters[name]["scenario"][i] for i in idx]

        uniq_cluster = sorted(set(ricardo_cluster))

        frequency = []
        spectogram = []
        for un in uniq_cluster:
            idx = np.where(np.array(ricardo_cluster) == un)[0]
            acc = ((ricardo_data_within_bounds["acc_side_1"] + ricardo_data_within_bounds["acc_side_2"]) / 2)[idx]
            tim = ricardo_data_within_bounds["time"][idx]

            window_length = 128
            if len(acc) < window_length:
                if len(acc) % 2 != 0:
                    window_length = len(acc) - 1
                else:
                    window_length = len(acc)

            w = window.Window(tim, acc, M=window_length, FS=FS)
            w.fft_w()
            w.plot_spectrogram(output_folder=output_path, name=f"{name}_{un}")
            # s = signal_tools.Signal(tim, acc, FS=FS)
            # s.fft()
            # plt.plot(w.frequency, w.amplitude)
            # plt.plot(s.frequency, s.amplitude)
            # plt.show()

            frequency.append(w.frequency)
            spectogram.append(w.spectrogram)

        tim = ricardo_data_within_bounds["time"]
        acc = ((ricardo_data_within_bounds["acc_side_1"] + ricardo_data_within_bounds["acc_side_2"]) / 2)
        s = signal_tools.Signal(tim, acc)
        s.fft()

        plot_seg(frequency, spectogram, s.frequency, s.amplitude, uniq_cluster, name, output_path)
    return


if __name__ == "__main__":
    ric("./SOS.json", ".\clustering_sensar/segments_cluster.json", "./inframon.pickle", "clustering_inframon")
    # fug("./SOS.json", "./clustering_sensar/segments_cluster.json", r"./updated_rila_data.pickle", "clustering_rila")
