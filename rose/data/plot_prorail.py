import json
import numpy as np
import matplotlib.pylab as plt
import matplotlib.dates as mdates
import datetime
from smooth import smooth


def plot_time_series(data: dict) -> None:

    import time
    t_ini = time.time()

    tracks = ["Track GT", "Track GH"]
    values = ["temperature", "cant", "settlement"]

    for v in values:
        for t in tracks:

            fig, ax = plt.subplots(figsize=(10, 4))
            ax.set_position([0.1, 0.25, 0.75, 0.6])

            for name in data:
                if data[name]["track"] == t:

                    date = [datetime.datetime.strptime(i, "%Y-%m-%d %H:%M:%S") for i in data[name][v]["time"]]
                    idx = np.argsort(date)
                    dat = list(map(float, data[name][v]["value"]))
                    # ax.plot(np.array(date)[idx], np.array(dat)[idx], label=name)
                    ax.plot(np.array(date)[idx], smooth(np.array(dat)[idx], 100), label=name)

            ax.set_title(f"{v} track: {t}")
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax.xaxis.set_major_locator(mdates.DayLocator(interval=20))

            for label in ax.get_xticklabels():
                label.set_ha("right")
                label.set_rotation(45)

            ax.set_xlabel("Time")
            ax.set_ylabel(f"{v}")
            # ax.set_ylim((-20, 30))
            ax.grid()
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., fontsize=6)
            plt.savefig(f"./{v}_{t}.png")
            plt.close()

    print(time.time() - t_ini)

    return


if __name__ == "__main__":
    with open("../../data/ProRail/processed/processed_geometry.json") as f:
        data = json.load(f)
    plot_time_series(data)
