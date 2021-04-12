from data_proc.sensar import *

import json

# sos_dir = "../rose/SOS"
# sos_fn = "SOS.json"

sos_fn = "../rose/SOS/SOS.json"

with open(sos_fn, 'r') as f:
    sos_data = json.load(f)

sensar_data = load_sensar_data("../data/Sensar/processed/processed_settlements.pickle")


for name, segment in sos_data.items():

    coordinates = np.array(list(segment.values())[0]['coordinates'])

    xlim = [min(coordinates[:,0]), max(coordinates[:,0])]
    ylim = [min(coordinates[:,1]), max(coordinates[:,1])]

    items_within_bounds = get_all_items_within_bounds(sensar_data, xlim, ylim)

    if items_within_bounds:

        all_dates = []
        all_settlements = []
        for item in items_within_bounds:
            dates = np.array([d.timestamp() for d in item['dates']])
            settlements = np.array(item['settlements'])
            all_settlements.append(settlements)
            all_dates.append(dates)

        all_dates, all_settlements = map_settlement_at_starting_date(all_dates, all_settlements)

        all_dates2 = np.array([date for dates in all_dates for date in dates])
        all_settlements2 = np.array([settlement for settlements in all_settlements for settlement in settlements])

        all_dates = all_dates2
        all_settlements = all_settlements2

        sorted_indices = np.argsort(all_dates)

        sorted_dates = all_dates[sorted_indices]
        sorted_settlements = all_settlements[sorted_indices]
        # sorted_velocities = all_velocities[sorted_indices]

        diff = np.diff(sorted_dates)

        step_idxs = np.insert(np.where(diff>0),0,0)

        all_means = np.array([])
        new_dates = np.array([])
        all_stds = np.array([])
        for i in range(1,len(step_idxs)):
            new_dates = np.append(new_dates,sorted_dates[step_idxs[i-1]])
            all_means = np.append(all_means, np.mean(sorted_settlements[step_idxs[i-1]:step_idxs[i]]))
            all_stds = np.append(all_stds, np.std(sorted_settlements[step_idxs[i-1]:step_idxs[i]]))

        # trend = np.polyfit(sorted_dates,sorted_settlements,1)



        trend_new = np.polyfit(new_dates,all_means,1)
        trend_3 = np.polyfit(new_dates,all_means + 2*all_stds,1)
        trend_4 = np.polyfit(new_dates,all_means - 2*all_stds,1)

        # new_dates = [datetime.fromtimestamp(int(date)) for date in new_dates]

        new_dates2 = [datetime.fromtimestamp(int(date)) for date in new_dates]
        sorted_dates = [datetime.fromtimestamp(int(date)) for date in sorted_dates]

        plt.plot(sorted_dates, sorted_settlements, 'o')
        plt.plot(new_dates2,all_means,'o')
        plt.plot(new_dates2,np.polyval(trend_new,new_dates))
        plt.plot(new_dates2,np.polyval(trend_3,new_dates))
        plt.plot(new_dates2,np.polyval(trend_4,new_dates))

        plt.title(name)
        plt.show()