from myplotlib.relative_error import relative_error
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def plot_relerr_to_true_value(interp, title=None,
                              methods=["model", "cubic"],
                              xlim=[-1.5, 1.5], ylim=[-1, 1.3],
                              fontsize=26, height=10, ratio=10,
                              bins=100, font_scale=2):

    err = {}
    met = {}
    for idx, met_key in enumerate(methods):
        err[met_key] = relative_error(interp["origin"], interp[met_key])
        if met_key == "origin":     met[idx] = "Origin"
        if met_key == "model":      met[idx] = "Network"
        if met_key == "linear":     met[idx] = "Linear"
        if met_key == "cubic":      met[idx] = "Spline"
        if met_key == "hermite":    met[idx] = "Hermite"

    err = np.log10(relative_error(err[methods[1]], err[methods[0]]))

    sns.set(font_scale=font_scale)
    graph = sns.jointplot(x=interp["origin"].reshape(-1), y=err, kind="hex",
                          height=height, ratio=ratio,
                          marginal_kws=dict(bins=bins),
                          xlim=xlim, ylim=ylim)

    graph.ax_joint.set_xlabel("True Value", fontsize=fontsize)
    graph.ax_joint.set_ylabel("Error(({met0}-{met1})/{met1})".format(met0=met[0], met1=met[1]), fontsize=fontsize)
    plt.subplots_adjust(left=0.2, right=0.8, top=0.8, bottom=0.2)
    cbar_ax = graph.fig.add_axes([.85, .25, .05, .4])
    plt.colorbar(cax=cbar_ax)
    if not title is None:   plt.title(title, fontsize=fontsize)
    plt.show()
