import os
from pathlib import Path

import numpy as np

from scope.utils.plot import (
    plot_dist_voting,
    plot_dist_spider,
    plot_dist_bars,
    plot_dissimilarity_matrix,
    plot_auc_roc as plot_auc_roc_scope,
    plot_confusion_matrix as plot_confusion_matrix_scope,
)


def plot_auc_roc(
    report: dict,
    title: str = "ROC Curve",
    save_path: str | None = None,
    show: bool = False,
    figsize: tuple[int, int] = (10, 5),
):
    plot_auc_roc_scope(
        report=report,
        title=title,
        save_path=save_path,
        show=show,
        figsize=figsize,
    )

def plot_confusion_matrix(
    confusion_matrix,
    normalize: bool = False,
    labels=None,
    figsize: tuple[int, int] = (6, 5),
    cmap: str = 'flare',
    title: str = "Confusion Matrix",
    save_path: str | None = None,
    show: bool = False,
):
   plot_confusion_matrix_scope(
       confusion_matrix=confusion_matrix,
       normalize=normalize,
       labels=labels,
       figsize=figsize,
       cmap=cmap,
       title=title,
       save_path=save_path,
       show=show,
   )


def plot_correct_predictions_by_class(
    predictions,
    y_true,
    cmap: str = 'flare',
    figsize: tuple[int, int] = (10, 5),
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
            dist_voting_path = None
            dist_bars_path = None
            dist_spider_path = None
            dissimilarity_matrix_path = None

            if save_dir:

                dist_voting_path = os.path.join(
                    save_dir,
                    f"correct_prediction_class_{class_label}_"
                    f"{local_index}_sample_{prediction_index}_dist_voting.png"
                )
                dist_bars_path = os.path.join(
                    save_dir,
                    f"correct_prediction_class_{class_label}_"
                    f"{local_index}_sample_{prediction_index}_dist_bars.png"
                )
                dist_spider_path = os.path.join(
                    save_dir,
                    f"correct_prediction_class_{class_label}_"
                    f"{local_index}_sample_{prediction_index}_dist_spider.png"
                )
                dissimilarity_matrix_path = os.path.join(
                    save_dir,
                    f"correct_prediction_class_{class_label}_"
                    f"{local_index}_sample_{prediction_index}_dissimilarity_matrix.png"
                )

            plot_dist_voting(
                prediction=prediction,
                cmap=cmap,
                figsize=figsize,
                save_path=dist_voting_path,
                show=show,
            )
            plot_dist_bars(
                prediction=prediction,
                cmap=cmap,
                figsize=figsize,
                save_path=dist_bars_path,
                show=show,
            )
            plot_dist_spider(
                prediction=prediction,
                cmap=cmap,
                figsize=figsize,
                save_path=dist_spider_path,
                show=show,
            )
            plot_dissimilarity_matrix(
                dissimilarity_matrix=prediction.dissimilarity_matrix,
                cmap=cmap,
                save_path=dissimilarity_matrix_path,
                show=show,
            )