from myplotlib.relative_error import relative_error
import matplotlib.pyplot as plt
import numpy as np


def plot_histogram_comp_methods(interp, methods, title,
                                xlim=[1e-3, 1e+1], ylim=[0, 1e-1],
                                x_range_log=[-4, 3], is_cum=False, loc="best",
                                fontsize=26, labelsize=15, linewidth=2.5,
                                figsize=(8, 5), length_major=20, length_minor=10,
                                direction="in"):

    bins = np.logspace(x_range_log[0], x_range_log[1], (x_range_log[1] - x_range_log[0]) * 20, base=10)

    fig = plt.figure(figsize=figsize)

    ax_hist = fig.add_subplot(111)
    ax_hist.set_xscale("log")
    ax_hist.set_xlabel("Relative Error", fontsize=fontsize)
    ax_hist.set_ylabel("Relative Frequency", fontsize=fontsize)
    ax_hist.set_xlim(xlim)
    ax_hist.set_ylim(ylim)
    ax_hist.tick_params(labelsize=labelsize,
                        length=length_major,
                        direction=direction, which="major")
    ax_hist.tick_params(labelsize=labelsize,
                        length=length_minor, direction=direction,
                        which="minor")
    if is_cum:
        ax_cum = ax_hist.twimx()
        ax_cum.set_ylabel("Cumulative", fontsize=fontsize)
        ax_cum.tick_params(labelsize=labelsize,
                           length=length_major, direction = direction,
                           which="major")

    for met_key in methods:
        if met_key == "origin": continue

        err     = relative_error(interp["origin"], interp[met_key])
        weights = np.ones_like(err) / err.size

        if met_key == "model":      label = "Network"
        if met_key == "linear":     label = "Linear"
        if met_key == "cubic":      label = "Spline"
        if met_key == "hermite":    label = "Hermite"
        hist_num, hist_bins, patches = ax_hist.hist(err, bins=bins, weights=weights,
                                                    histtype="step", linewidth=linewidth,
                                                    label="{label}".format(label=label))
        mode_index = hist_num.argmax()
        print("most numbers of the data : ({0}, {1})".format(hist_bins[mode_index], hist_bins[mode_index+1]))
        print("mode : {mode}".format(mode=(hist_bins[mode_index] + hist_bins[mode_index+1]) / 2))

    ax_hist.legend(loc=loc, fontsize=int(fontsize * 0.6))
    plt.title(title, fontsize=fontsize)
    plt.show()
