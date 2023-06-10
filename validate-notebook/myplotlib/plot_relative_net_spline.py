from myplotlib.relative_error import relative_error
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns
import numpy as np


def plot_err_network_spline(interp, title=None,
                            methods=["model", "cubic"],
                            xlim=[-3, 1], ylim=[-3, 1],
                            fontsize=26, height=10, ratio=10,
                            bins=100, font_scale=2):

    err = {}
    for met_key in methods:
        err[met_key] = np.log10(relative_error(interp["origin"], interp[met_key]))

    sns.set(font_scale=font_scale)
    graph = sns.jointplot(x = err[methods[0]], y=err[methods[1]], kind="hex",
                          height=height, ratio=ratio,
                          marginal_kws=dict(bins=bins),
                          xlim=xlim, ylim=ylim)

    ax = graph.ax_joint
    ax.plot(ax.get_xlim(), ax.get_ylim(), ls="--", c=".3")

    label = {}
    for idx, met_key in enumerate(methods):
        if met_key == "model":   label[idx] = "Network"
        if met_key == "linear":  label[idx] = "Linear"
        if met_key == "cubic":   label[idx] = "Spline"
        if met_key == "hermite": label[idx] = "Hermite"
        
    graph.ax_joint.set_xlabel(label[0], fontsize=fontsize)
    graph.ax_joint.set_ylabel(label[1], fontsize=fontsize)
    plt.subplots_adjust(left=0.2, right=0.8, top=0.8, bottom=0.2)
    cbar_ax = graph.fig.add_axes([.85, .25, .05, .4])
    plt.colorbar(cax=cbar_ax)
    if not title is None:   plt.title(title, fontsize=fontsize)
    plt.show()

    return np.corrcoef(err[methods[0]], err[methods[1]])
