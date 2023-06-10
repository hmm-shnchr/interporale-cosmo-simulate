from myplotlib.relative_error import relative_error
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def plot_true_vs_error(interp, methods,
                       height=10, ratio=10, fontsize=26,
                       xlim=[-1.5, 1.5], ylim=[-3, 1],
                       font_scale=2, bins=100):

    for met_key in methods:
        if met_key == "origin":     continue
        if met_key == "model":      met = "Network"
        if met_key == "linear":     met = "Linear"
        if met_key == "cubic":      met = "Spline"
        if met_key == "hermite":    met = "Hermite"

        err     = relative_error(interp["origin"], interp[met_key])
        err     = np.log10(err)
        origin  = interp["origin"].reshape(-1)

        sns.set(font_scale=font_scale)
        graph = sns.jointplot(x=origin, y=err, kind="hex",
                              height=height, ratio=ratio,
                              marginal_kws=dict(bins = bins),
                              xlim=xlim, ylim=ylim)

        graph.ax_joint.set_xlabel("True Value", fontsize=fontsize)
        graph.ax_joint.set_ylabel("{0}".format(met), fontsize=fontsize)
        plt.subplots_adjust(left=0.2, right=0.8, top=0.8, bottom=0.2)
        cbar_ax = graph.fig.add_axes([.85, .25, .05, .4])
        plt.colorbar(cax=cbar_ax)
        plt.show()
