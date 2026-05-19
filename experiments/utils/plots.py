import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from scope.utils.plot import plot_prediction


def plot_auc_roc(
    report: dict,
    title: str = "ROC Curve",
    save_path: str | None = None,
    show: bool = False,
):
    fpr = report.get("fpr")
    tpr = report.get("tpr")
    auc_roc = report.get("auc_roc")

    if fpr is None or tpr is None:
        raise ValueError("ROC curve is only available for binary classification reports.")

    fpr = np.asarray(fpr)
    tpr = np.asarray(tpr)

    fig, ax = plt.subplots(figsize=(6, 5))

    ax.plot(
        fpr,
        tpr,
        label=f"AUC ROC = {auc_roc:.3f}",
        linewidth=2,
    )
    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        color="gray",
        label="Random",
    )


    ax.set_title(title)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300)

    if show:
        plt.show()

    plt.close(fig)

def plot_confusion_matrix(
    confusion_matrix,
    normalize: bool = False,
    labels=None,
    title: str = "Confusion Matrix",
    save_path: str | None = None,
    show: bool = False,
):
    cm = np.asarray(confusion_matrix)

    if labels is None:
        labels = [str(i) for i in range(cm.shape[0])]

    fig, ax = plt.subplots(figsize=(6, 5))

    image = ax.imshow(cm, cmap="Blues")
    fig.colorbar(image, ax=ax)

    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    threshold = cm.max() / 2 if cm.size else 0

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            text_value = f"{cm[i, j]:.2f}" if normalize else str(int(cm[i, j]))

            color = "white" if cm[i, j] > threshold else "black"

            ax.text(
                j,
                i,
                text_value,
                ha="center",
                va="center",
                color=color,
                fontsize=12,
            )

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300)

    if show:
        plt.show()

    plt.close(fig)


def plot_correct_predictions_by_class(
    predictions,
    y_true,
    n_per_class: int = 3,
    save_dir: str | None = None,
    show: bool = False,
):
    y_true = np.asarray(y_true)

    selected_by_class = {}

    for index, prediction in enumerate(predictions):
        true_class = int(y_true[index])
        predicted_class = int(prediction.predicted_class)

        if true_class != predicted_class:
            continue

        if true_class not in selected_by_class:
            selected_by_class[true_class] = []

        if len(selected_by_class[true_class]) < n_per_class:
            selected_by_class[true_class].append((index, prediction))

    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    for class_label, selected_predictions in selected_by_class.items():
        for local_index, (prediction_index, prediction) in enumerate(selected_predictions, start=1):
            save_path = None

            if save_dir:
                save_path = os.path.join(
                    save_dir,
                    f"correct_prediction_class_{class_label}_"
                    f"{local_index}_sample_{prediction_index}.png"
                )

            plot_prediction(
                prediction=prediction,
                plot=show,
                save_path=str(save_path) if save_path else None,
            )