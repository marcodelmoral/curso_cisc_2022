from itertools import cycle

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy import interp
from sklearn.metrics import (
    PrecisionRecallDisplay,
    auc,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
)
from tensorflow.keras.utils import to_categorical

map_7_classes = {
    0: "normal_superficiel",
    1: "normal_intermediate",
    2: "normal_columnar",
    3: "light_dysplastic",
    4: "moderate_dysplastic",
    5: "severe_dysplastic",
    6: "carcinoma_in_situ",
}

map_2_classes = {0: "normal", 1: "anormal"}

map_normal_anormal = {
    "normal_superficiel": 0,
    "normal_intermediate": 0,
    "normal_columnar": 0,
    "light_dysplastic": 1,
    "moderate_dysplastic": 1,
    "severe_dysplastic": 1,
    "carcinoma_in_situ": 1,
}


def exp_smooth(points, factor=0.8):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points


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


def plot_roc(datos, figsize=(10, 8)):
    datos = datos.replace({v: k for k, v in map_2_classes.items()})
    fpr, tpr, _ = roc_curve(datos["Real"].values, datos["Pred"].values)
    auc_keras = auc(fpr, tpr)
    plt.figure(figsize=figsize)
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.plot(fpr, tpr, label=f"Área bajo la curva (AUC) = {auc_keras}")
    plt.xlabel("Tasa de falsos positivos")
    plt.ylabel("Tasa de falsos negativos")
    plt.title("Curva ROC")
    plt.legend(loc="best")
    plt.show()


def plot_precision_recall(datos, figsize=(10, 8)):
    _, ax = plt.subplots(figsize=figsize)
    datos = datos.replace({v: k for k, v in map_2_classes.items()})
    display = PrecisionRecallDisplay.from_predictions(
        datos["Real"].values, datos["Pred"].values, ax=ax
    )
    _ = display.ax_.set_title("Curva Precision-Recall para problema binario")


def plot_roc_multiclass(datos, class_list):
    n_classes = len(class_list)
    datos = datos.replace({v: k for k, v in map_7_classes.items()})
    y = to_categorical(datos["Real"].values)
    ypred = to_categorical(datos["Pred"].values)
    # Plot linewidth.
    lw = 2
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y[:, i], ypred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y[:, i], ypred[:, i])
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Compute macro-average ROC curve and ROC area

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Zoom in view of the upper left corner.
    plt.figure(figsize=(10, 8))
    plt.xlim(0, 0.2)
    plt.ylim(0.8, 1)
    plt.plot(
        fpr["micro"],
        tpr["micro"],
        label=f"Micro-promedio de curva ROC: {roc_auc['micro']:.4f}",
        color="deeppink",
        linestyle=":",
        linewidth=4,
    )

    plt.plot(
        fpr["macro"],
        tpr["macro"],
        label=f"Macro-promedio de curva ROC: {roc_auc['macro']:.4f}",
        color="navy",
        linestyle=":",
        linewidth=4,
    )

    colors = cycle(
        [
            "aqua",
            "darkorange",
            "cornflowerblue",
            "forestgreen",
            "orchid",
            "darkblue",
            "olive",
        ]
    )
    for i, color in zip(range(len(class_list)), colors):
        plt.plot(
            fpr[i],
            tpr[i],
            color=color,
            lw=lw,
            label=f"AUC {class_list[i]} = {roc_auc[i]:.4f}",
        )

    plt.plot([0, 1], [0, 1], "k--", lw=lw)
    plt.xlim([0, 1])
    plt.ylim([0, 1.05])
    plt.xlabel("Tasa de falsos positivos")
    plt.ylabel("Tasa de falsos negativos")
    plt.title("Curva ROC multiclase")
    plt.legend(loc="lower right", prop={"size": 10})
    plt.show()


def plot_precision_recall_curve_multiclass(datos, class_list, figsize=(10, 8)):
    n_classes = len(class_list)
    datos = datos.replace({v: k for k, v in map_7_classes.items()})
    Y_test = to_categorical(datos["Real"].values)
    y_score = to_categorical(datos["Pred"].values)

    # For each class
    precision = dict()
    recall = dict()
    average_precision = dict()
    for i in range(7):
        precision[i], recall[i], _ = precision_recall_curve(
            Y_test[:, i], y_score[:, i]
        )
        average_precision[i] = average_precision_score(
            Y_test[:, i], y_score[:, i]
        )

    # A "micro-average": quantifying score on all classes jointly
    precision["micro"], recall["micro"], _ = precision_recall_curve(
        Y_test.ravel(), y_score.ravel()
    )
    average_precision["micro"] = average_precision_score(
        Y_test, y_score, average="micro"
    )
    # setup plot details
    colors = cycle(
        ["navy", "turquoise", "darkorange", "cornflowerblue", "teal"]
    )

    _, ax = plt.subplots(figsize=figsize)

    f_scores = np.linspace(0.2, 0.8, num=4)
    lines, labels = [], []
    for f_score in f_scores:
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        (l,) = plt.plot(x[y >= 0], y[y >= 0], color="gray", alpha=0.2)
        plt.annotate("f1={0:0.1f}".format(f_score), xy=(0.9, y[45] + 0.02))

    display = PrecisionRecallDisplay(
        recall=recall["micro"],
        precision=precision["micro"],
        average_precision=average_precision["micro"],
    )
    display.plot(
        ax=ax,
        name="Micro-promedio PR",
        color="red",
        linestyle=":",
        linewidth=4,
    )

    for i, color in zip(range(n_classes), colors):
        display = PrecisionRecallDisplay(
            recall=recall[i],
            precision=precision[i],
            average_precision=average_precision[i],
        )
        display.plot(
            ax=ax, name=f"PR {class_list[i]}", color=color,
        )

    # add the legend for the iso-f1 curves
    handles, labels = display.ax_.get_legend_handles_labels()
    handles.extend([l])
    labels.extend(["iso-f1 curves"])
    # set the legend and the axes
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.legend(handles=handles, labels=labels, loc="lower left")
    ax.set_title("Curva PR multiclase")


def plot_augmented(generator, augmentor, figsize=(10, 10)):
    plt.figure(figsize=figsize)
    batch = generator.next()
    image_ = batch[0].astype("uint8")
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        augmented = augmentor(image_[0])
        plt.imshow(augmentor(image_[0]).numpy().astype("uint8"))
        plt.axis("off")
    # plt.title(class_names[np.argmax(label, axis=0)])
    plt.show()


def plot_softmax(
    i, predictions_array, label_array, img_array, class_names, figsize=(15, 7)
):

    plt.figure(figsize=figsize)
    plt.subplot(1, 2, 1)

    true_label, img = label_array[i], np.array(Image.open(img_array[i]))
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
        f'{class_names[predicted_label]} ({100 * np.max(predictions_array):2.0f}%) {"Correcto" if color=="blue" else class_names[true_label]}',
        color=color,
    )

    plt.subplot(1, 2, 2)
    red_patch = mpatches.Patch(color="red", label="Incorrecto")
    blue_patch = mpatches.Patch(color="blue", label="Correcto")

    plt.legend(handles=[red_patch, blue_patch])
    plt.grid(False)
    plt.xticks(range(len(class_names)), class_names, rotation=90)
    # plt.yticks([])
    thisplot = plt.bar(
        range(len(class_names)), predictions_array, color="#777777"
    )
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color("red")
    thisplot[true_label].set_color("blue")
    ax = plt.gca()

    rects = ax.patches

    for rect, label in zip(rects, predictions_array):
        height = rect.get_height()
        ax.text(
            rect.get_x() + rect.get_width() / 2,
            height,
            f"{label:.2f}",
            ha="center",
            va="bottom",
        )


def sample_val_data(val_generator):
    samples = []
    mapper = {v: k for k, v in val_generator.class_indices.items()}
    for filename, label in zip(val_generator.filenames, val_generator.labels):
        samples.append({"filename": filename, "label": mapper[label]})

    df_samples = pd.DataFrame(samples)

    return (
        df_samples.groupby("label")
        .apply(lambda x: x.sample(n=1))
        .reset_index(drop=True)
        .to_dict("records")
    )

