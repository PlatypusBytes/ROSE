
import json
from pathlib import Path
from typing import List, Dict

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

import sensar
import fugro
import ricardo
import SoS
import data_discontinuities as dd
import smooth
from rose.utils import signal_proc

settings_filter = {"FS": 250,
                   "cut-off": 120,
                   "n": 10,
                   "wavelength": 10}

def update_figure(ax,old_fig,new_fig,position):
    ax.remove()
    ax.figure=new_fig

    new_fig.axes[0] = ax

    new_fig.axes.append(ax)
    new_fig.add_axes(ax)

    ax.set_position(new_fig.axes[0].get_position())

    plt.close(old_fig)


def filter_data_sets(sensar_dict: Dict, fugro_dict: Dict ,ricardo_dict: Dict,data_discontinuities_fns: List[str]):
    """
    Filters Sensar data, Fugro data and Ricardo data on data discontinuity locations

    :param sensar_dict: Sensar data dictionary
    :param fugro_dict:  Fugro data dictionary
    :param ricardo_dict: Ricardo data dictionary
    :param data_discontinuities_fns:  data discontinuity file names
    :return:
    """

    #todo make more general, current only ricardo data from January is considered
    ricardo_dict = ricardo_dict["Jan"]

    # loop over data discontinuity files
    for fn in data_discontinuities_fns:

        # get coordinates from data discontinuity file
        filter_coordinates = dd.get_coordinates_from_json(fn)

        # if data in data discontinuity file are point coordinates, filter Sensar, Fugro and Ricardo data on point coordinates
        if isinstance(dd.get_coordinates_from_json(fn)[0][0], float):
            search_radius = 1
            sensar_dict = sensar.filter_data_at_point_coordinates(sensar_dict,filter_coordinates)
            fugro_dict = fugro.filter_data_at_point_coordinates(fugro_dict, filter_coordinates, search_radius)
            ricardo_dict = ricardo.filter_data_at_point_coordinates(ricardo_dict, filter_coordinates, search_radius)

        # else if data in data discontinuity file are line coordinates, get boundaries of corresponding lines and filter
        # Sensar, Fugro and Ricardo data within the boundaries
        else:
            x_bounds, y_bounds = dd.get_bounds_of_lines(filter_coordinates)
            sensar_dict = sensar.filter_data_within_bounds(x_bounds, y_bounds,sensar_dict)
            fugro.filter_data_within_bounds(x_bounds, y_bounds, fugro_dict)
            ricardo_dict = ricardo.filter_data_within_bounds(x_bounds, y_bounds, ricardo_dict)

    # return filter data dictionaries
    return sensar_dict, fugro_dict, ricardo_dict


def plot_data_on_sos_segment(sos_dict, sensar_dict, fugro_dict, ricardo_dict, options):
    """
    Plots data from sensar, fugro and ricardo within each SOS segment in a separate subplot.

    :param sos_dict: Sos data
    :param sensar_dict: Sensar data
    :param fugro_dict: Fugro rila data
    :param ricardo_dict: Ricardo data
    :return:
    """

    # get date limits from sensar data and fugro data
    sensar_dates = list(sensar_dict.values())[0]["dates"]
    fugro_dates = fugro_dict["dates"]

    min_date = min([min(sensar_dates), min(fugro_dates)])
    max_date = max([max(sensar_dates), max(fugro_dates)])

    date_lim = [min_date, max_date]

    # loop over sos segments
    for name, segment in sos_dict.items():

        if name == "Segment 1083":
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
            plt.grid()
            # add plot of settlement within the current segment measured by the fugro rila system
            _, _ = fugro.plot_settlement_in_range_vs_date(fugro_dict, xlim, ylim, date_lim=date_lim, fig=fig, position=321)
            plt.grid()

            # add plot of Sensar settlement measurements within the current segment
            sensar_items_within_bounds = sensar.get_all_items_within_bounds(sensar_dict, xlim, ylim)
            if sensar_items_within_bounds:
                _, _ = sensar.plot_settlements_from_item_list_over_time(sensar_items_within_bounds,date_lim=date_lim, fig=fig, position=323)
                plt.grid()

            # get ricardo data
            ricardo_data_within_bounds = ricardo.get_data_within_bounds(ricardo_dict, xlim, ylim)
            # ricardo_data_within_bounds = ricardo.get_data_within_bounds(ricardo_dict["Jan"], xlim, ylim)
            if ricardo_data_within_bounds["acc_side_1"].size>0:

                # filter Ricardo measurements
                acc = signal_proc.filter_sig(ricardo_data_within_bounds["acc_side_1"],
                                             settings_filter["FS"], settings_filter["cut-off"], settings_filter["n"],
                                             ).tolist()
                acc = signal_proc.filter_sig(acc, settings_filter["FS"], 40, 10, type="highpass")

                # plot train velocity
                ricardo.plot_train_velocity(ricardo_data_within_bounds, fig=fig, position=322)
                plt.grid()

                # plot either ricardo acceleration measurements or transformed velocity
                ricardo_signal_type = "acceleration"
                if ricardo_signal_type == "acceleration":
                    ricardo.plot_acceleration_signal(ricardo_data_within_bounds["time"], acc, fig=fig, position=324)
                    plt.grid()
                    ricardo.plot_fft_acceleration_signal(ricardo_data_within_bounds, acc,10, fig=fig,position=326)
                    plt.grid()
                elif ricardo_signal_type == "velocity":
                    ricardo.plot_velocity_signal(ricardo_data_within_bounds["time"], acc, fig=fig, position=324)
                    plt.grid()
                    ricardo.plot_fft_velocity_signal(ricardo_data_within_bounds,acc, 10,fig=fig, position=326)
                    plt.grid()

            fig.suptitle(name)
            fig.savefig(Path("tmp8", name))

            plt.close(fig)

if __name__ == '__main__':

    sos_fn = "../data_proc/SOS.json"
    with open(sos_fn, 'r') as f:
        sos_data = json.load(f)

    sensar_data = sensar.load_sensar_data("../data/Sensar/processed/processed_settlements.pickle")

    fugro_data = fugro.load_rila_data(r"../data/Fugro/rila_data.pickle")
    fugro_data = fugro.merge_data(fugro_data)

    ricardo_data = ricardo.load_inframon_data("./inframon.pickle")

    crossing_fn = r"D:\software_development\rose\data\data_discontinuities\kruising.json"
    wissels_fn = r"D:\software_development\rose\data\data_discontinuities\wissel.json"
    overweg_fn = r"D:\software_development\rose\data\data_discontinuities\overweg.json"

    data_discontinuities_fns = [crossing_fn, wissels_fn, overweg_fn]

    # sensar_data, fugro_data, ricardo_data = filter_data_sets(sensar_data, fugro_data, ricardo_data, data_discontinuities_fns)

    plot_data_on_sos_segment(sos_data, sensar_data, fugro_data, ricardo_data["Jan"],0)