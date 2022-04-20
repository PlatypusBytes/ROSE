import os
import json
import pickle
import numpy as np
import matplotlib.pylab as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from datetime import datetime
import warnings
warnings.simplefilter('ignore', np.RankWarning)

# rose packages
from data_proc.sensar import load_sensar_data, get_all_items_within_bounds, map_settlement_at_starting_date


CLR = {0: {"color": "b",
           "thickness": 3},
       1: {"color": "r",
           "thickness": 3},
       2: {"color": "tab:olive",
           "thickness": 2},
       3: {"color": "lightblue",
           "thickness": 2},
       4: {"color": "navy",
           "thickness": 3},
       5: {"color": "y",
           "thickness": 3},
       6: {"color": "0.2",
           "thickness": 3},
       7: {"color": "0.4",
           "thickness": 2},
       8: {"color": "0.6",
           "thickness": 3},
       9: {"color": "0.8",
           "thickness": 2},
       }

RANGES = [
          datetime(2018, 1, 1),
          datetime(2019, 1, 1),
          datetime(2020, 1, 1),
          datetime(2020, 11, 20),
]


def main(path_sensar, path_SOS, path_results, degree=1, coord=False, plot=False):

    if not os.path.isdir(path_results):
        os.makedirs(path_results)

    # read sensar
    sensar_data = load_sensar_data(path_sensar)

    # read sos
    with open(path_SOS, 'r') as f:
        sos_data = json.load(f)

    # loop over sos segments
    score_km = []
    results = {}
    for name, segment in sos_data.items():

        # skip segments where sensar data does not exists
        if not all([int(name.split(" ")[1]) >= 1029, int(name.split(" ")[1]) <= 1080]):
            continue

        # get SOS coordinates of current segment
        coordinates = np.array(list(segment.values())[0]['coordinates'])

        # get SOS coordinate limits
        xlim = [min(coordinates[:, 0]), max(coordinates[:, 0])]
        ylim = [min(coordinates[:, 1]), max(coordinates[:, 1])]

        # collect sensar data in the SOS segment
        sensar_items_within_bounds = get_all_items_within_bounds(sensar_data, xlim, ylim)

        # compute rate / point
        features = []
        coordinates = []
        aux = 0
        idx = []

        all_settlements = []
        all_dates = []
        idx_to_delete = []
        # for every sensar point in the SOS segment
        for i in range(len(sensar_items_within_bounds)):

            # convert dates to timestamps
            dates = np.array([d.timestamp() for d in sensar_items_within_bounds[i]['dates']])
            settlements = np.array(sensar_items_within_bounds[i]['settlements'])

            if settlements.size > 0 and dates.size > 0:
                all_settlements.append(settlements)
                all_dates.append(dates)
            else:
                idx_to_delete.append(i)

        # maps all settlements at starting date
        all_dates, all_settlements = map_settlement_at_starting_date(all_dates, all_settlements)
        # delete empty indexes
        for i in idx_to_delete[::-1]:
            sensar_items_within_bounds.pop(i)

        for i in range(len(sensar_items_within_bounds)):
            fit = []  # fit

            # check if ranges in sensar dataset are good (are within RANGES)
            # If not skips the sensar point
            if not all([any(np.array(sensar_items_within_bounds[i]["dates"]) >= RANGES[-1]),
                        any(np.array(sensar_items_within_bounds[i]["dates"]) < RANGES[0])]):
                aux += 1
                continue

            idx.append(i)
            # for all data ranges -> compute fit
            for dat in range(1, len(RANGES)):
                # initial index of sensar data
                id_ini = np.where(np.array(sensar_items_within_bounds[i]["dates"]) >= RANGES[dat - 1])[0][0]
                # final index of sensar data
                id_end = np.where(np.array(sensar_items_within_bounds[i]["dates"]) >= RANGES[dat])[0][0]
                tim = [all_dates[i][k] for k in range(id_ini, id_end)]
                val = [all_settlements[i][k] for k in range(id_ini, id_end)]

                fit.extend(np.polyfit(tim, val, deg=degree))

            x = [min(np.array(sensar_items_within_bounds[i]["coordinates"])[:, 0]),
                 max(np.array(sensar_items_within_bounds[i]["coordinates"])[:, 0])]
            y = [min(np.array(sensar_items_within_bounds[i]["coordinates"])[:, 1]),
                 max(np.array(sensar_items_within_bounds[i]["coordinates"])[:, 1])]

            # add
            coordinates.append([x[0], y[0], x[1], y[1]])
            if coord:
                vars_aux = [np.mean([coordinates[-1][0], coordinates[-1][2]]),
                            np.mean([coordinates[-1][1], coordinates[-1][3]])]
                vars_aux = [np.sqrt(vars_aux[0] ** 2 + vars_aux[1] ** 2)]
                vars_aux.extend(fit)
                features.append(vars_aux)
            else:
                features.append(fit)

        # print how many invalid sensar points are in the segment
        print(f"{name}: skyped {aux} / {len(sensar_items_within_bounds)}")

        # K-means
        nb_clusters = len(sos_data[name])
        if nb_clusters != 1:
            kmeans = KMeans(n_clusters=nb_clusters, random_state=0).fit(np.array(features))
            cls = kmeans.predict(np.array(features))
            score_km.append(silhouette_score(np.array(features), cls))
        else:
            cls = np.zeros(len(features))
            score_km.append(1)

        perc = {}
        # compute percentage of the K-mean classes
        for j in set(cls):
            perc.update({int(j): round(len(np.where(cls == j)[0]) / len(cls) * 100, 1)})

        # sensar data used for the clustering
        full_data = [sensar_items_within_bounds[j] for j in idx]

        # add to output dict
        coord_list = [[(coordinates[i][0] + coordinates[i][2]) / 2, (coordinates[i][1] + coordinates[i][3])/ 2] for i in range(len(coordinates))]
        results.update({name: {"coordinates": coord_list,
                               "scenario": list(map(int, cls)),
                               "time": [aux["dates"] for aux in full_data],
                               "settlements": [aux["settlements"] for aux in full_data],
                               "percentage": perc}})

        if plot:

            # percentages clustering
            percentages_cluster = [perc[i] for i in perc]

            # read sos percentages
            aux_perc = [sos_data[name][sce]["probability"] for sce in sos_data[name]]

            sos_perc = np.zeros(len(percentages_cluster))
            sos_perc[np.argsort(percentages_cluster)] = np.array(aux_perc)[np.argsort(aux_perc)]

            # plot figure
            plot_segment(coordinates, cls, full_data, perc, sos_perc, name, path_results)

    with open(os.path.join(path_results, "segments_cluster.pickle"), "wb") as fo:
        pickle.dump(results, fo)

    print(f"Mean score {np.mean(np.array(score_km))}")
    print(f"Maximum {np.max(np.array(score_km))}")
    print(f"Minimum {np.min(np.array(score_km))}")

    return


def plot_segment(coordinates, data_class, data_features, percentage, sos_percentage, name, output_path):

    # if output path does not exist -> creates
    if not os.path.isdir(output_path):
        os.mkdir(output_path)

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].set_position([0.08, 0.12, 0.4, 0.8])
    ax[1].set_position([0.56, 0.12, 0.4, 0.8])

    for j in range(len(data_class)):
        ax[0].plot([coordinates[j][0], coordinates[j][2]], [coordinates[j][1], coordinates[j][3]],
                   color=CLR[data_class[j]]["color"], linewidth=CLR[data_class[j]]["thickness"], label=data_class[j])
        ax[1].plot(data_features[j]["dates"], data_features[j]["settlements"],
                   color=CLR[data_class[j]]["color"], linewidth=0, marker="x", markersize=2, alpha=0.25, label=data_class[j])

    # plot mean
    uniq_class = set(data_class)
    for un in uniq_class:
        idx = np.where(data_class == un)[0]
        sett = []
        dat = []
        for j in idx:
            sett.append(data_features[j]["settlements"])
            dat.append(data_features[j]["dates"])

        # mean and std for stiffness
        mean_v = []
        std_v = []

        # get unique dates
        uniq_dates_sorted = sorted(set([val for sublist in dat for val in sublist]))

        for d in uniq_dates_sorted:
            # get indexes for the unique date
            idx = [np.where(np.array(val) == d)[0] for i, val in enumerate(dat)]
            # get existing data at this date
            data = [sett[i][val[0]] for i, val in enumerate(idx) if len(val) != 0]
            aver_val = np.mean(np.array(data))
            std_val = np.sqrt(np.average((np.array(data) - aver_val) ** 2))
            mean_v.append(aver_val)
            std_v.append(std_val)

        ax[1].plot(uniq_dates_sorted, mean_v,
                   color=CLR[un]["color"], linewidth=0, marker="o", markersize=3)
        ax[1].plot(uniq_dates_sorted, np.array(mean_v) + 1.0 * np.array(std_v),
                   color=CLR[un]["color"], linewidth=0.0, marker="_", markersize=3)
        ax[1].plot(uniq_dates_sorted, np.array(mean_v) - 1.0 * np.array(std_v),
                   color=CLR[un]["color"], linewidth=0.0, marker="_", markersize=3)

    fig.suptitle(f"{name}")

    # sort labels
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    keys_sorted = sorted(by_label)
    values_sorted = [by_label[k] for k in keys_sorted]
    keys_sorted = [f"Sce {k} Prob: {percentage[k]}% ({sos_percentage[i]}%)" for i, k in enumerate(keys_sorted)]
    ax[0].legend(values_sorted, keys_sorted)

    ax[0].grid()
    ax[0].set_xlabel("X coordinate")
    ax[0].set_ylabel("Y coordinate")
    ax[1].grid()
    ax[1].set_xlabel("Time")
    ax[1].set_ylabel("Settlement")
    plt.savefig(os.path.join(output_path, f"{name}.png"))
    plt.close()

    return


if __name__ == "__main__":
    main("../data/Sensar/processed/filtered_processed_settlements_2.pickle",
         "../data_proc/SOS.json",
         "clustering_sensar",
         degree=1, coord=True, plot=False)
