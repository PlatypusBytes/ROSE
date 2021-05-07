from data_proc.sensar import *
from data_proc.fugro import *

import json

# sos_dir = "../rose/SOS"
# sos_fn = "SOS.json"

sos_fn = "../data_proc/SOS.json"

with open(sos_fn, 'r') as f:
    sos_data = json.load(f)

sensar_data = load_sensar_data("../data/Sensar/processed/processed_settlements.pickle")

file_dir = r"D:\software_development\ROSE\data\Fugro\Amsterdam_Eindhoven\Deltares_AmsterdamEindhovenKRDZ"
res = get_data_at_location(file_dir, location="Amsterdam_Utrecht")
# res = get_data_at_location(file_dir, location="DenBosch_Eindhoven")
# res = get_data_at_location(file_dir, location="Utrecht_DenBosch")
# DenBosch_Eindhoven
# DenBosch_Eindhoven
import itertools
marker = itertools.cycle((',', '+', '.', 'o', '*'))

for name, segment in sos_data.items():

    coordinates = np.array(list(segment.values())[0]['coordinates'])

    xlim = [min(coordinates[:,0]), max(coordinates[:,0])]
    ylim = [min(coordinates[:,1]), max(coordinates[:,1])]

    items_within_bounds = get_all_items_within_bounds(sensar_data, xlim, ylim)

    heights = []
    for res_at_t in res["data"]:
        coordinates_in_range, heights_in_range = filter_data_within_bounds(xlim, ylim, res_at_t)
        mean_height = np.mean(heights_in_range)
        heights.append(mean_height)


        if coordinates_in_range.size and coordinates_in_range.size:
            # distance = np.append(0,np.cumsum((np.diff(coordinates_in_range[:,0])**2 + np.diff(coordinates_in_range[:,1])**2) **1/2))
            distance = coordinates_in_range[:,1]
            plt.plot(distance, heights_in_range, marker=next(marker),markersize=2)


    plt.title(name)

    plt.xlabel("y-coord")
    plt.ylabel("height [m NAP]")

    plt.close()
    heights = (np.array(heights) - heights[0]) * 1000

    array_sum = np.sum(heights)
    array_has_nan = np.isnan(array_sum)





    if items_within_bounds and not array_has_nan :

        fig, ax = plot_settlements_from_item_list_over_time(items_within_bounds)

        fig.show()

    # if items_within_bounds:

        # sorted_dates, sorted_settlements = get_all_dates_and_settlement_as_sorted_array(items_within_bounds)

        # all_dates = []
        # all_settlements = []
        # for item in items_within_bounds:
        #     dates = np.array([d.timestamp() for d in item['dates']])
        #     settlements = np.array(item['settlements'])
        #     all_settlements.append(settlements)
        #     all_dates.append(dates)
        #
        # all_dates, all_settlements = map_settlement_at_starting_date(all_dates, all_settlements)
        #
        # all_dates = np.array([date for dates in all_dates for date in dates])
        # all_settlements = np.array([settlement for settlements in all_settlements for settlement in settlements])
        #
        # # all_dates = all_dates_tmp
        # # all_settlements = all_settlements_tmp
        #
        # sorted_indices = np.argsort(all_dates)
        # sorted_dates = all_dates[sorted_indices]
        # sorted_settlements = all_settlements[sorted_indices]
        # sorted_velocities = all_velocities[sorted_indices]

        # new_dates, all_means, all_stds = get_statistical_information(sorted_dates, sorted_settlements)
        # diff = np.diff(sorted_dates)
        #
        # step_idxs = np.insert(np.where(diff>0),0,0)
        #
        # all_means = np.array([])
        # new_dates = np.array([])
        # all_stds = np.array([])
        # for i in range(1,len(step_idxs)):
        #     new_dates = np.append(new_dates,sorted_dates[step_idxs[i-1]])
        #     all_means = np.append(all_means, np.mean(sorted_settlements[step_idxs[i-1]:step_idxs[i]]))
        #     all_stds = np.append(all_stds, np.std(sorted_settlements[step_idxs[i-1]:step_idxs[i]]))

        # trend = np.polyfit(sorted_dates,sorted_settlements,1)



        # trend_new = np.polyfit(new_dates,all_means,1)
        # trend_3 = np.polyfit(new_dates,all_means + 2*all_stds,1)
        # trend_4 = np.polyfit(new_dates,all_means - 2*all_stds,1)
        #
        # # new_dates = [datetime.fromtimestamp(int(date)) for date in new_dates]
        #
        # plot_settlement_over_time(sorted_dates, sorted_settlements)
        # plot_settlement_over_time(new_dates, all_means)

        # new_dates2 = [datetime.fromtimestamp(int(date)) for date in new_dates]
        # sorted_dates = [datetime.fromtimestamp(int(date)) for date in sorted_dates]
        # #
        # plt.plot(sorted_dates, sorted_settlements, 'o')
        #
        # plt.plot(new_dates2,all_means,'o')
        # # # plt.plot(new_dates2,np.polyval(trend_new,new_dates))
        # # # plt.plot(new_dates2,np.polyval(trend_3,new_dates))
        # # # plt.plot(new_dates2,np.polyval(trend_4,new_dates))
        # #
        # #
        # # plt.plot(res["dates"], heights)
        # #
        # plt.title(name)
        # plt.xlabel('Date [y]')
        # plt.ylabel('Settlement [mm]')
        # plt.show()