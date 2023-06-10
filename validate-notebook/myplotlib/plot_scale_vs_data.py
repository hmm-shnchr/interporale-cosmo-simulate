import matplotlib.pyplot as plt


def plot_scale_vs_data(origin, model, scalefactor, method, p_key, method_key,
                       figsize = (10, 6), loc = "upper left",
                       linewidth = 2.5, fontsize = 26, labelsize = 15, direction = "in",
                       origin_c = "orange", pred_c = "blue",
                       origin_m = "o", pred_m = "s", origin_size = 300, pred_size = 200, inputp_size = 300,
                       legend_scale = 0.5):

    if method_key not in ["linear", "cubic", "spline", "hermite"]:
        print("{method_key} is not supported.".format(method_key=method_key))
        return None

    fig, (ax_model, ax_method)  = plt.subplots(ncols=2, sharey=True, figsize=figsize)
    fig.subplots_adjust(wspace=0.0)

    ## Plot Model's data.
    mask_origin = origin == model
    ax_model.scatter(scalefactor[mask_origin], origin[mask_origin], label = "Input points", marker = origin_m, color = origin_c, s = inputp_size, zorder = 3)
    ax_model.plot(scalefactor, origin, label = "Origin", linewidth = linewidth, color = origin_c, marker = origin_m, markersize = origin_size, markerfacecolor = "white")
    ax_model.plot(scalefactor, model, label = "Network", linewidth = linewidth, color = pred_c, marker = pred_m, markersize = pred_size)
    ax_model.tick_params(labelsize = labelsize, direction = direction)
    ax_model.set_ylabel(p_key, fontsize = fontsize)
    ax_model.set_xlabel("Scale Factor", fontsize = fontsize)
    ax_model.legend(loc = loc, fontsize = int(fontsize * legend_scale))

    ## Plot Interpolate-method's data.
    ax_method.scatter(scalefactor[mask_origin], origin[mask_origin], label = "Input points", marker = origin_m, color = origin_c, s = inputp_size, zorder = 3)
    ax_method.plot(scalefactor, origin, label = "Origin", linewidth = linewidth, color = origin_c, marker = origin_m, markersize = origin_size, markerfacecolor = "white")
    ax_method.plot(scalefactor, method, label = method_key, linewidth = linewidth, color = pred_c, marker = pred_m, markersize = pred_size)
    ax_method.tick_params(labelsize = labelsize, direction = direction)
    ax_method.set_xlabel("Scale Factor", fontsize = fontsize)
    ax_method.legend(loc = loc, fontsize = int(fontsize * legend_scale))