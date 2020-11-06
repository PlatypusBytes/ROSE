import os
import json
import matplotlib.pylab as plt


class ComplexEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, complex):
            return [obj.real, obj.imag]
        # Let the base class default method raise the TypeError
        return json.JSONEncoder.default(self, obj)


def create_plot(frequency, stiff, damp, label_x, path_results, name):

    # create figure
    fig, ax = plt.subplots(1, 2, figsize=(8.5, 3.5))
    ax[0].set_position([0.13, 0.13, 0.35, 0.80])
    ax[1].set_position([0.60, 0.13, 0.35, 0.80])
    plt.rcParams.update({'font.size': 10})

    # plot stiffness
    ax[0].grid()
    ax[0].plot(frequency, stiff)
    ax[0].set_xlabel(label_x)
    ax[0].set_ylabel(r'K$_{dyn}$ [N/m]')
    ax[0].set_xlim((frequency[0], frequency[-1]))
    # ax[0].set_ylim(bottom=0)

    # plot damping
    ax[1].grid()
    ax[1].plot(frequency, damp)
    ax[1].set_xlabel(label_x)
    ax[1].set_ylabel(r'Damping [Ns/m]')
    ax[1].set_xlim((frequency[0], frequency[-1]))
    # ax[1].set_ylim(bottom=0)

    # save fig
    plt.savefig(os.path.join(path_results, f"{name}.png"))
    plt.savefig(os.path.join(path_results, f"{name}.pdf"))
    plt.close()
    return
