import matplotlib.pyplot as plt
import numpy as np


def plot_figure(train_acc, test_acc, save_dir, save_fig_name,
                fontsize = 26, labelsize = 15,
                length_major = 20, length_minor = 13,
                ylim = [0.09, 1.0],
                linewidth = 2.5, figsize = (8, 5)):

    epochs = np.arange(train_acc.size) * 10
    fig = plt.figure(figsize = figsize)
    ax_acc = fig.add_subplot(111)
    ax_acc.plot(epochs[1:], train_acc[1:], label = "Training", linewidth = linewidth, color = "orange")
    ax_acc.plot(epochs[1:], test_acc[1:], label = "Testing", linewidth = linewidth, color = "blue")
    ax_acc.set_yscale("log")
    ax_acc.set_xlabel("Epochs", fontsize = fontsize)
    ax_acc.set_ylabel("Accuracy", fontsize = fontsize)
    ax_acc.legend(loc = "upper right", fontsize = int(fontsize * 0.6))
    ax_acc.tick_params(labelsize = labelsize, length = length_major, direction = "in", which = "major")
    ax_acc.tick_params(labelsize = labelsize, length = length_minor, direction = "in", which = "minor")
    ax_acc.set_ylim(ylim)
    plt.title("Training and Testing Accuracy", fontsize = fontsize)
    plt.tight_layout()
    plt.savefig(save_dir + save_fig_name)
    plt.show()


if __name__ == "__main__":
    dir_list = np.loadtxt("directory_list.txt", dtype = "str")
    for idx in range(1, len(dir_list)):
        print("{0}. {1}".format(idx, dir_list[idx]))
    dir_idx = int(input("select a directory>> "))
    train_acc = np.loadtxt(dir_list[dir_idx] + "data_acc_train_x1.csv", delimiter = ",")
    test_acc = np.loadtxt(dir_list[dir_idx] + "data_acc_test_x1.csv", delimiter = ",")

    plot_figure(train_acc, test_acc, dir_list[dir_idx], "fig_acc_x1.png")
