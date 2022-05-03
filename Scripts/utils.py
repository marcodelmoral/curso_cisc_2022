import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def plot_figures(figures, nrows=1, ncols=1, figsize=(7.16, 2.625), file=None):
    """Plot a dictionary of figures.

    Parameters
    ----------
    figures : <title, figure> dictionary
    ncols : number of columns of subplots wanted in the display
    nrows : number of rows of subplots wanted in the figure
    """

    fig, axeslist = plt.subplots(ncols=ncols, nrows=nrows, figsize=figsize)
    for ind, title in enumerate(figures):
        axeslist.ravel()[ind].imshow(title, cmap=plt.gray())
        axeslist.ravel()[ind].set_axis_off()
    plt.gca().set_axis_off()
    # plt.tight_layout()
    fig.subplots_adjust(hspace=-0.29, wspace=0.02)
    # plt.margins(0, 0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    if file:
        plt.savefig(f"{file}.png")
        plt.savefig(f"{file}.pdf")


def exp_smooth(points, factor=0.8):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points


def plot_metrics(history, metrics=["loss", "accuracy"], figsize=(8, 8)):
    for metric in metrics:
        plt.plot(history.history[metric])
        plt.plot(history.history["val_" + metric], linestyle="--")
        plt.title("model " + metric)
        plt.ylabel(metric)
        plt.xlabel("epoch")
        plt.legend([metrics, "val_" + metric], loc="upper left")
        plt.show()


def plot_kfold(cv, X, y, n_splits, file=None, lw=20, autoshow=True):
    _, ax = plt.subplots()
    cmap_data = plt.cm.tab10
    cmap_cv = plt.cm.Blues
    """Create a sample plot for indices of a cross-validation object."""

    # Generate the training/testing visualizations for each CV split
    for ii, (tr, tt) in enumerate(cv.split(X, y)):
        # Fill in indices with the training/test groups
        indices = np.array([np.nan] * len(X))
        indices[tt] = 1
        indices[tr] = 0

        # Visualize the results
        ax.scatter(
            range(len(indices)),
            [ii + 0.5] * len(indices),
            c=indices,
            marker="_",
            lw=lw,
            cmap=cmap_cv,
            vmin=-0.2,
            vmax=1.2,
        )
    # Plot the data classes and groups at the end
    scat = ax.scatter(
        range(len(X)),
        [ii + 1.5] * len(X),
        c=y.map(
            {
                "normal_superficiel": 0,
                "normal_intermediate": 1,
                "normal_columnar": 2,
                "light_dysplastic": 3,
                "moderate_dysplastic": 4,
                "severe_dysplastic": 5,
                "carcinoma_in_situ": 6,
            }
        ),
        marker="_",
        lw=lw,
        cmap=cmap_data,
    )

    # handles = scat.legend_elements()[0]
    # labels = y.unique()

    # legend1 = ax.legend(
    #     handles, labels, loc=(0.05, 0.135), title="Cell types", ncol=7,
    # )

    # ax.add_artist(legend1)

    scat2 = ax.scatter(
        range(len(X)),
        [ii + 2.5] * len(X),
        c=y.map(
            {
                "normal_superficiel": 0,
                "normal_intermediate": 0,
                "normal_columnar": 0,
                "light_dysplastic": 1,
                "moderate_dysplastic": 1,
                "severe_dysplastic": 1,
                "carcinoma_in_situ": 1,
            }
        ),
        marker="_",
        lw=lw,
        cmap=matplotlib.colors.ListedColormap(["cornflowerblue", "maroon"]),
    )

    # handles2 = scat2.legend_elements()[0]

    # labels2 = ["Normal", "Abnormal"]

    # legend2 = ax.legend(
    #     handles2, labels2, loc="lower center", title="Diagnostic", ncol=7,
    # )

    # ax.add_artist(legend2)

    yticklabels = list(range(n_splits)) + ["Multiclase", "Binario"]
    ax.set(
        yticks=np.arange(n_splits + 2) + 0.5,
        yticklabels=yticklabels,
        ylim=[n_splits + 2.2, -0.2],
        xlim=[0, len(X)],
    )
    ax.set_ylabel("Kfold split")
    ax.set_xlabel("Datapoint index")

    ax.tick_params(axis="both", direction="out")
    ax.get_xaxis().tick_bottom()  # remove unneeded ticks
    ax.get_yaxis().tick_left()
    ax.set_title(f"{type(cv).__name__}")
    # ax.legend(
    #     [Patch(color=cmap_cv(0.8)), Patch(color=cmap_cv(0.1))],
    #     ["Testing set", "Training set"],
    #     loc=(0.455, 0.270),
    #     prop={"size": 9},
    # )

    if autoshow:
        plt.show()


def plot_kfold_subplot(
    kf, skf, X, y, n_splits, lw=20, autoshow=True, figsize=(8, 8)
):
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=figsize, sharex=True, sharey=True
    )
    cmap_data = plt.cm.tab10
    cmap_cv = plt.cm.Blues
    """Create a sample plot for indices of a cross-validation object."""
    for ax, cv in zip((ax1, ax2), (kf, skf)):
        for ii, (tr, tt) in enumerate(cv.split(X, y)):
            # Fill in indices with the training/test groups
            indices = np.array([np.nan] * len(X))
            indices[tt] = 1
            indices[tr] = 0

            # Visualize the results
            scat_inner = ax.scatter(
                range(len(indices)),
                [ii + 0.5] * len(indices),
                c=indices,
                marker="_",
                lw=lw,
                cmap=cmap_cv,
                vmin=-0.2,
                vmax=1.2,
            )
        # Plot the data classes and groups at the end
        scat = ax.scatter(
            range(len(X)),
            [ii + 1.5] * len(X),
            c=y.map(
                {
                    "normal_superficiel": 0,
                    "normal_intermediate": 1,
                    "normal_columnar": 2,
                    "light_dysplastic": 3,
                    "moderate_dysplastic": 4,
                    "severe_dysplastic": 5,
                    "carcinoma_in_situ": 6,
                }
            ),
            marker="_",
            lw=lw,
            cmap=cmap_data,
        )

        handles = scat.legend_elements()[0]
        labels = y.unique()

        scat2 = ax.scatter(
            range(len(X)),
            [ii + 2.5] * len(X),
            c=y.map(
                {
                    "normal_superficiel": 0,
                    "normal_intermediate": 0,
                    "normal_columnar": 0,
                    "light_dysplastic": 1,
                    "moderate_dysplastic": 1,
                    "severe_dysplastic": 1,
                    "carcinoma_in_situ": 1,
                }
            ),
            marker="_",
            lw=lw,
            cmap=matplotlib.colors.ListedColormap(["cornflowerblue", "maroon"]),
        )

        handles2 = scat2.legend_elements()[0]

        labels2 = ["Normal", "Abnormal"]

        yticklabels = list(range(1, n_splits + 1)) + [
            "Tipo de célula",
            "Diagnóstico",
        ]
        ax.set(
            yticks=np.arange(n_splits + 2) + 0.5,
            yticklabels=yticklabels,
            ylim=[n_splits + 2.2, -0.2],
            xlim=[0, len(X)],
        )

        ax.tick_params(axis="both", direction="out")
        ax.get_xaxis().tick_bottom()  # remove unneeded ticks
        ax.get_yaxis().tick_left()
        ax.set_title(f"{type(cv).__name__}")

    fig.tight_layout()
    fig.add_subplot(111, frameon=False)

    plt.tick_params(
        labelcolor="none", top=False, bottom=False, left=False, right=False
    )
    # plt.grid(False)
    plt.xlabel("Índice")
    plt.ylabel("División")
    # plt.title("Cross-validation comparison")
    plt.minorticks_off()
    # hand = [Line2D(color=cmap_cv(0.8)), Line2D(color=cmap_cv(0.1))]
    hand = scat_inner.legend_elements()[0]
    hand.extend(handles)
    hand.extend(handles2)
    lab = ["Entrenamiento", "Validación"]
    lab.extend(labels)
    lab.extend(labels2)

    lab = [l.replace("_", " ").capitalize() for l in lab]
    fig.legend(hand, lab, loc="upper right", bbox_to_anchor=(1.17, 0.85))

    if autoshow:

        plt.show()


def plot_figures(figures, nrows=1, ncols=1, figsize=(7.16, 2.625)):
    """Plot a dictionary of figures.

    Parameters
    ----------
    figures : <title, figure> dictionary
    ncols : number of columns of subplots wanted in the display
    nrows : number of rows of subplots wanted in the figure
    """

    fig, axeslist = plt.subplots(ncols=ncols, nrows=nrows, figsize=figsize)
    titless = []
    for ind, title in enumerate(figures):
        axeslist.ravel()[ind].imshow(title[0], cmap=plt.gray())
        axeslist.ravel()[ind].set_axis_off()
        titless.append(title[1])
    for ax, col in zip(axeslist[0], titless):
        ax.set_title(col)
    plt.gca().set_axis_off()
    # plt.tight_layout()
    # fig.subplots_adjust(hspace=-0.29, wspace=0.02)
    # plt.margins(0, 0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())


def plot_image(i, predictions_array, true_label, img, class_names):
    true_label, img = true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img.astype(np.uint8), cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = "blue"
    else:
        color = "red"

    plt.xlabel(
        "{} {:2.0f}% ({})".format(
            class_names[predicted_label],
            100 * np.max(predictions_array),
            class_names[true_label],
        ),
        color=color,
    )


def plot_value_array(i, predictions_array, true_label, class_names):
    true_label = true_label[i]
    plt.grid(False)
    plt.xticks(range(len(class_names)), class_names, rotation=90)
    plt.yticks([])
    thisplot = plt.bar(
        range(len(class_names)), predictions_array, color="#777777"
    )
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color("red")
    thisplot[true_label].set_color("blue")


def exp_smooth(points, factor=0.8):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points


def plot_metrics(
    history,
    metrics=["loss", "accuracy"],
    figsize=(8, 8),
    smooth=False,
    **kwargs,
):
    alpha = 1
    for metric in metrics:
        plt.figure(figsize=figsize)
        if smooth:
            factor = 0.8
            if kwargs.get("factor", None):
                factor = kwargs["factor"]
            plt.plot(
                exp_smooth(history.history[metric], factor=factor),
                linestyle="--",
            )
            plt.plot(
                exp_smooth(history.history["val_" + metric], factor=factor),
                linestyle="--",
            )
            alpha = 0.3
            plt.plot(history.history[metric], linestyle="-", alpha=alpha)
        plt.plot(history.history["val_" + metric], linestyle="-", alpha=alpha)
        plt.title(metric)
        plt.ylabel(metric)
        plt.xlabel("epoch")
        plt.legend([metric, "val_" + metric], loc="upper left")
        plt.show()

