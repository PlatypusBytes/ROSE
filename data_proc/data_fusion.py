
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

import sensar
import fugro


def update_figure(ax,old_fig,new_fig,position):
    ax.remove()
    ax.figure=new_fig

    new_fig.axes[0] = ax

    new_fig.axes.append(ax)
    new_fig.add_axes(ax)

    ax.set_position(new_fig.axes[0].get_position())

    plt.close(old_fig)


def plot_data_on_sos_segment(sos_dict, sensar_dict, fugro_dict):

    for name, segment in sos_dict.items():

        fig = plt.figure(figsize=(20,5))
        plt.tight_layout()


        coordinates = np.array(list(segment.values())[0]['coordinates'])
        xlim = [min(coordinates[:,0]), max(coordinates[:,0])]
        ylim = [min(coordinates[:,1]), max(coordinates[:,1])]

        sensar_items_within_bounds = sensar.get_all_items_within_bounds(sensar_dict, xlim, ylim)

        fig2, ax = fugro.plot_average_height_in_range_vs_date(xlim, ylim, fugro_dict, fig=fig, position=121)

        if sensar_items_within_bounds:
            fig3, ax2 = sensar.plot_settlements_from_item_list_over_time(sensar_items_within_bounds, fig=fig, position=122)

        fig.savefig(Path("tmp", name))

        plt.close(fig)

    # return fig

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

    sensar_data = sensar.load_sensar_data("../data/Sensar/processed/processed_settlements.pickle")

    fugro_file_dir = r"D:\software_development\ROSE\data\Fugro\Amsterdam_Eindhoven\Deltares_AmsterdamEindhovenKRDZ"
    fugro_data = fugro.get_data_at_location(fugro_file_dir, location="all")

    plot_data_on_sos_segment(sos_data, sensar_data, fugro_data)

    # fig.show()

    pass