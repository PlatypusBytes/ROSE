
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

import sensar
import fugro
import ricardo
import SoS
from rose.utils import signal_proc

settings_filter = {"FS": 250,
                   "cut-off": 30,
                   "n": 10}

def update_figure(ax,old_fig,new_fig,position):
    ax.remove()
    ax.figure=new_fig

    new_fig.axes[0] = ax

    new_fig.axes.append(ax)
    new_fig.add_axes(ax)

    ax.set_position(new_fig.axes[0].get_position())

    plt.close(old_fig)


def plot_data_on_sos_segment(sos_dict, sensar_dict, fugro_dict, ricardo_dict):
    """
    Plots data from sensar, fugro and ricardo within each SOS segment in a separate subplot.

    :param sos_dict: Sos data
    :param sensar_dict: Sensar data
    :param fugro_dict: Fugro rila data
    :param ricardo_dict: Ricardo data
    :return:
    """

    # get date limits
    # sensar_dates = sensar_dict.values()

    sensar_dates = list(sensar_dict.values())[0]["dates"]
    fugro_dates = fugro_dict["dates"]

    min_date = min([min(sensar_dates), min(fugro_dates)])
    max_date = max([max(sensar_dates), max(fugro_dates)])

    date_lim = [min_date, max_date]

    # loop over segments
    for name, segment in sos_dict.items():

        # initialise figure
        fig = plt.figure(figsize=(20,10))
        plt.tight_layout()

        # get coordinates of current segments
        coordinates = np.array(list(segment.values())[0]['coordinates'])

        # get coordinate limits
        xlim = [min(coordinates[:,0]), max(coordinates[:,0])]
        ylim = [min(coordinates[:,1]), max(coordinates[:,1])]

        # add plot of highlighted sos segments
        _, _ = SoS.ReadSosScenarios.plot_highlighted_sos(sos_data, name, fig=fig, position=325)

        # add plot of settlement within the current segment measured by the fugro rila system
        _, _ = fugro.plot_settlement_in_range_vs_date(fugro_dict, xlim, ylim, date_lim=date_lim, fig=fig, position=321)

        # add plot of Sensar settlement measurements within the current segment
        sensar_items_within_bounds = sensar.get_all_items_within_bounds(sensar_dict, xlim, ylim)
        if sensar_items_within_bounds:
            _, _ = sensar.plot_settlements_from_item_list_over_time(sensar_items_within_bounds,date_lim=date_lim, fig=fig, position=323)

        # get ricardo data
        ricardo_data_within_bounds =  ricardo.get_data_within_bounds(ricardo_dict["Jan"], xlim, ylim)
        if ricardo_data_within_bounds["acc_side_1"].size>0:
            # filter Ricardo measurements
            acc = signal_proc.filter_sig(ricardo_data_within_bounds["acc_side_1"],
                                         settings_filter["FS"], settings_filter["n"],
                                         settings_filter["cut-off"]).tolist()

            # add plot of train velocity and aspot measurements by Ricardo
            ricardo.plot_train_velocity(ricardo_data_within_bounds, fig=fig, position=322)
            ricardo.plot_acceleration(ricardo_data_within_bounds["time"],acc, fig=fig, position=324)


        fig.suptitle(name)
        fig.savefig(Path("tmp", name))

        plt.close(fig)

if __name__ == '__main__':



    # plt.subplot(221)
    #
    # # equivalent but more general
    # ax1 = plt.subplot(2, 2, 1)
    #
    # # add a subplot with no frame
    # ax2 = plt.subplot(222, frameon=False)
    #
    # # add a polar subplot
    # plt.subplot(223, projection='polar')
    #
    # # add a red subplot that shares the x-axis with ax1
    # plt.subplot(224, sharex=ax1, facecolor='red')
    #
    # # delete ax2 from the figure
    # plt.delaxes(ax2)
    #
    # # add ax2 to the figure again
    # plt.subplot(ax2)




    sos_fn = "../data_proc/SOS.json"
    with open(sos_fn, 'r') as f:
        sos_data = json.load(f)

    # SoS.ReadSosScenarios.plot_highlighted_sos(sos_data,)
    # plot_highlighted_sos(sos_dict, highlighted_segment_name)

    sensar_data = sensar.load_sensar_data("../data/Sensar/processed/processed_settlements.pickle")

    # fugro_file_dir = r"D:\software_development\ROSE\data\Fugro\Amsterdam_Eindhoven\Deltares_AmsterdamEindhovenKRDZ"

    # fugro_file_dir = r"D:\software_development\ROSE\data\Fugro\Amsterdam-Eindhoven TKI Project"
    # fugro_data = fugro.get_data_at_location(fugro_file_dir, location="all",filetype="csv")
    fugro_data = fugro.load_rila_data(r"../data/Fugro/rila_data.pickle")

    fugro_data = fugro.merge_data(fugro_data)

    ricardo_data = ricardo.load_inframon_data("./inframon.pickle")

    plot_data_on_sos_segment(sos_data, sensar_data, fugro_data, ricardo_data)

    # fig.show()

    pass