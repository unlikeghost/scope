import numpy as np
import seaborn as sns
from typing import Dict
from matplotlib.patches import Patch

from pathlib import Path

import matplotlib.pyplot as plt
from torch.distributed._pycute import layout

from ..prediction import PoligonPrediction, DistPrediction


def plot_polygon_prediction(
    prediction: PoligonPrediction,
    show: bool = False,
    save_path: str | None = None,
) -> None:

    n_cols = len(prediction.scores)
    keys = prediction.scores.keys()

    colors = ['red', 'blue', 'green', 'orange', 'purple']

    fig, ax = plt.subplots(
        nrows=1, ncols=n_cols, figsize=(20, 10), sharex=False,
    )

    for index, key in enumerate(keys):

        # Plot the convex hulls
        ax[index].plot(
            *prediction.convex_hull_clusters[key].exterior.xy, # noqa
            color=colors[index],
            linewidth=2,
            linestyle='--',
            label=f"Convex Hull Cluster"
        )

        geom = prediction.convex_hull_queries[key]
        if hasattr(geom, 'exterior'):
            x, y = geom.exterior.xy
        else:
            x, y = geom.xy

        ax[index].plot(
            x, y,
            color=colors[index],
            linewidth=2,
            linestyle='-',
            label="Convex Hull Query"
        )
        # Plot the centroids
        ax[index].scatter(
            prediction.convex_hull_clusters[key].centroid.x,
            prediction.convex_hull_clusters[key].centroid.y,
            marker='x',
            color='black',
            s=100,
            label='Cluster Centroid'
        )
        ax[index].scatter(
            prediction.convex_hull_queries[key].centroid.x,
            prediction.convex_hull_queries[key].centroid.y,
            marker='+',
            color='black',
            s=100,
            label='Query Centroid'
        )

        # Plot the query points
        ax[index].scatter(
            prediction.query_points[key][:, 0],
            prediction.query_points[key][:, 1],
            color='red',
            label='Query Points',
            s=30
        )

        ax[index].set_title(
            f"Cluster {key}\n"
            f"Prediction score = {prediction.scores[key]:.2f}"
        )
        ax[index].set_xlabel('NCD')
        ax[index].set_ylabel('CDM')
        ax[index].set_aspect('equal', adjustable='box')
        ax[index].legend()
        ax[index].grid(
            True,
            linestyle=':', alpha=0.6
        )

    plt.tight_layout()
    if show:
        plt.show()

    if save_path:
        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches='tight'
        )
    plt.close(fig)


def plot_dissimilarity_matrix(
    dissimilarity_matrix: Dict[str, np.ndarray],
    cmap: str = 'rocket',
    show: bool = False,
    save_path: str | None = None
):
    cluster_keys = [k for k in dissimilarity_matrix if 'Cluster' in k]
    n_clusters = len(cluster_keys)
    sample = dissimilarity_matrix[cluster_keys[0]]

    _, n_compressors, n_metrics, n_classes = sample.shape

    n_rows = n_clusters * n_metrics
    n_columns = n_compressors

    fig, ax = plt.subplots(
        nrows=n_rows,
        ncols=n_columns,
        figsize=(n_columns * 6, n_rows * 4.5),
        layout='tight'
    )
    fig.set_layout_engine(layout='tight', w_pad=3)

    if n_rows == 1 and n_columns == 1:
        ax = np.array([[ax]])
    elif n_rows == 1:
        ax = ax[np.newaxis, :]
    elif n_columns == 1:
        ax = ax[:, np.newaxis]

    row_vmins = {}
    row_vmaxs = {}

    row_idx = 0
    for key in cluster_keys:
        for metric_idx in range(n_metrics):
            row_data = [
                dissimilarity_matrix[key][:, comp_idx, metric_idx, :]
                for comp_idx in range(n_compressors)
            ]
            row_vmins[row_idx] = np.concatenate(row_data).min()
            row_vmaxs[row_idx] = np.concatenate(row_data).max()
            row_idx += 1

    row_idx = 0
    for class_idx, key in enumerate(cluster_keys):
        for metric_idx in range(n_metrics):
            vmin = row_vmins[row_idx]
            vmax = row_vmaxs[row_idx]

            for compressor_idx in range(n_compressors):
                metric = dissimilarity_matrix[key][:, compressor_idx, metric_idx, :]

                support_data = metric[:-1]
                query_data = metric[-1:]
                labels = [f'Support {n}' for n in range(support_data.shape[0])] + ['Query']
                combined_heatmap = np.vstack([support_data, query_data])

                current_ax = ax[row_idx, compressor_idx]

                sns.heatmap(
                    combined_heatmap,
                    vmin=vmin,
                    vmax=vmax,
                    cmap=cmap,
                    cbar=True,
                    cbar_kws={
                        'label': 'Metric Dissimilarity',
                        'format': '%0.2f',
                        'pad': 0.07,
                        'fraction': 0.1,
                        'shrink': 0.85
                    },
                    xticklabels=np.arange(n_classes),
                    yticklabels=labels,
                    ax=current_ax
                )

                cbar = current_ax.collections[0].colorbar
                cbar.set_label('Metric Dissimilarity', labelpad=15)

                current_ax.tick_params(axis='both', labelsize=9, labelcolor='#666666', length=0)
                current_ax.spines[['top', 'right', 'left', 'bottom']].set_visible(False)
                current_ax.set_title(
                    f"Class {class_idx}  ·  Metric {metric_idx}  |  Compressor {compressor_idx}",
                    fontsize=9, color='#666666'
                )

            row_idx += 1

    if show:
        plt.show()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.close(fig)


def plot_auc_roc(
    report: dict,
    title: str = "ROC Curve",
    figsize: tuple[int, int] = (10, 5),
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

    fig, ax = plt.subplots(figsize=figsize, layout='tight')

    ax.plot(fpr, tpr, linewidth=2, label=f"AUC = {auc_roc:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="#bbbbbb", linewidth=0.8)
    ax.fill_between(fpr, tpr, alpha=0.08)

    ax.spines[['top', 'right']].set_visible(False)
    ax.spines['left'].set_color('#cccccc')
    ax.spines['left'].set_linewidth(0.8)
    ax.spines['bottom'].set_color('#cccccc')
    ax.spines['bottom'].set_linewidth(0.8)

    ax.tick_params(axis='x', labelsize=8, labelcolor='#888888', direction='out')
    ax.tick_params(axis='y', labelsize=8, labelcolor='#888888', direction='out')
    ax.grid(alpha=0.2, linewidth=0.6, color='gray')
    ax.set_axisbelow(True)

    ax.legend(loc="lower right", fontsize=9, framealpha=0.3)

    fig.suptitle(title, fontsize=13)

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
    cmap: str = 'rocket',
    figsize: tuple[int, int] = (10, 5),
    title: str = "Confusion Matrix",
    save_path: str | None = None,
    show: bool = False,
):
    cm = np.asarray(confusion_matrix)
    if labels is None:
        labels = [str(i) for i in range(cm.shape[0])]

    fig, ax = plt.subplots(figsize=figsize, layout='tight')

    image = ax.imshow(cm, cmap=cmap)

    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=7, labelcolor='#888888')
    cbar.outline.set_visible(False)

    ax.spines[['top', 'right', 'left', 'bottom']].set_visible(False)
    ax.tick_params(axis='x', labelsize=9, labelcolor='#666666', length=0)
    ax.tick_params(axis='y', labelsize=9, labelcolor='#666666', length=0)

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels([f"Class {l}" for l in labels], fontsize=9)
    ax.set_yticklabels([f"Class {l}" for l in labels], fontsize=9)

    threshold = cm.max() / 2 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            text_value = f"{cm[i, j]:.2f}" if normalize else str(int(cm[i, j]))
            color = "white" if cm[i, j] > threshold else "black"
            ax.text(j, i, text_value, ha="center", va="center", color=color, fontsize=11)

    fig.suptitle(title, fontsize=13)

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300)
    if show:
        plt.show()
    plt.close(fig)


def plot_dist_voting(
    prediction: DistPrediction,
    cmap: str = 'rocket',
    figsize: tuple[int, int] = (7, 5),
    show: bool = False,
    save_path: str | None = None
):
    n_classes = len(prediction.distances)
    n_classifiers = len(prediction.distances[0])
    class_labels = list(prediction.distances.keys())

    cmap_object = sns.color_palette(cmap, as_cmap=True)
    colors = [
        cmap_object(0.85) if i == prediction.predicted_class else cmap_object(0.30)
        for i in range(n_classes)
    ]

    votes = [prediction.scores[i] for i in range(n_classes)]

    fig, ax = plt.subplots(figsize=figsize, layout='tight')

    bars = ax.bar(class_labels, votes, color=colors, edgecolor='white', linewidth=1.5, width=0.5)

    for bar, v in zip(bars, votes):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.15,
            str(int(v)),
            ha='center', va='bottom',
            fontsize=11, fontweight='bold',
            color=colors[votes.index(v)]
        )

    ax.set_ylim(0, n_classifiers + 1.5)
    ax.set_xticks(range(n_classes))
    ax.set_xticklabels([f"Class {l}" for l in class_labels], fontsize=10, color='#666666')
    ax.tick_params(axis='y', labelsize=7)

    ax.spines[['top', 'right']].set_visible(False)

    ax.grid(axis='y', color='gray', alpha=0.2, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.axhline(n_classifiers / 2, color='gray', linewidth=0.8, linestyle='--')  # línea de mayoría


    fig.suptitle(
        f"Votes per class",
        fontsize=13,
    )

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300)
    if show:
        plt.show()
    plt.close(fig)


def plot_dist_spider(
    prediction: DistPrediction,
    cmap: str = 'rocket',
    figsize: tuple[int, int] = (10, 5),
    show: bool = False,
    save_path: str | None = None
):
    # --- Data ---
    n_classes = len(prediction.distances)
    n_classifiers = len(prediction.distances[0])
    class_labels = list(prediction.distances.keys())
    classifier_labels = prediction.classifier_labels

    dist_matrix = np.array([prediction.distances[i] for i in range(n_classes)])
    dist_matrix = (dist_matrix - np.mean(dist_matrix)) / np.std(dist_matrix)

    cmap_object = sns.color_palette(cmap, as_cmap=True)
    won_color = cmap_object(0.85)
    lost_color = cmap_object(0.30)

    angles = np.linspace(0, 2 * np.pi, n_classifiers, endpoint=False).tolist()
    angles += angles[:1]

    r_max = np.abs(dist_matrix).max() * 1.2

    fig, axes = plt.subplots(
        1, n_classes,
        figsize=figsize,
        subplot_kw=dict(polar=True),
        layout='constrained'
    )
    axes = np.array(axes).flatten()

    for cls_idx in range(n_classes):
        ax = axes[cls_idx]
        values = dist_matrix[cls_idx].tolist()
        values += values[:1]
        is_winner = cls_idx == prediction.predicted_class
        color = won_color if is_winner else lost_color

        ax.plot(angles, values, color=color, linewidth=2 if is_winner else 1.2)
        ax.fill(angles, values, color=color, alpha=0.35 if is_winner else 0.1)
        ax.scatter(angles[:-1], values[:-1], color=color, s=30 if is_winner else 20, zorder=5, linewidths=0)

        ax.set_ylim(-r_max, r_max)
        ax.spines['polar'].set_visible(False)
        ax.grid(color='gray', alpha=0.3, linewidth=0.6)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(classifier_labels, fontsize=7)
        ax.tick_params(axis='x', pad=10)
        ax.tick_params(axis='y', labelsize=5, labelcolor='#888888')

        ax.set_title(
            f"Class {class_labels[cls_idx]}",
            fontsize=8, pad=12,
            fontweight='bold' if is_winner else 'normal',
        )

    fig.suptitle(
        f"Distance profile per class",
        fontsize=13,
    )

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300)
    if show:
        plt.show()
    plt.close(fig)


def plot_dist_bars(
    prediction: DistPrediction,
    cmap: str = 'rocket',
    figsize: tuple[int, int] = (10, 5),
    show: bool = False,
    save_path: str | None = None
):
    n_classes = len(prediction.distances)
    class_labels = list(prediction.distances.keys())
    classifier_labels = prediction.classifier_labels

    dist_matrix = np.array([prediction.distances[i] for i in range(n_classes)])

    cmap_object = sns.color_palette(cmap, as_cmap=True)
    class_colors = [
        cmap_object(0.85) if cls_idx == prediction.predicted_class else cmap_object(0.3)
        for cls_idx in range(n_classes)
    ]

    winner_per_classifier = np.argmin(dist_matrix, axis=0)
    bar_colors = [class_colors[winner] for winner in winner_per_classifier]

    # Diferencia direccional: clase 0 − clase 1
    diff = dist_matrix[0] - dist_matrix[1]

    fig, ax = plt.subplots(figsize=figsize, layout='tight')

    bars = ax.barh(classifier_labels, diff, color=bar_colors, edgecolor='white', linewidth=1.2)
    ax.axvline(0, color='gray', linewidth=0.8, linestyle='--')

    x_min, x_max = min(diff), max(diff)
    x_range = x_max - x_min
    offset = x_range * 0.02

    for bar, d, winner in zip(bars, diff, winner_per_classifier):
        ha_pos = 'right' if d < 0 else 'left'
        text_x = d - offset if d < 0 else d + offset
        ax.text(
            text_x,
            bar.get_y() + bar.get_height() / 2,
            f'{class_labels[winner]}  {d:.3f}',
            va='center', ha=ha_pos,
            fontsize=8,
            color=class_colors[winner]
        )

    ax.set_xlim(x_min - x_range * 0.2, x_max + x_range * 0.2)
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(axis='x', labelsize=7)
    ax.tick_params(axis='y', labelsize=8)
    ax.set_title(
        f'← {class_labels[0]} wins  |  {class_labels[1]} wins →',
        fontsize=9, pad=8, color='gray'
    )
    ax.set_xlabel('Δ distance (class 0 − class 1)', fontsize=9)

    legend_elements = [Patch(facecolor=class_colors[i], label=class_labels[i]) for i in range(n_classes)]
    ax.legend(handles=legend_elements, fontsize=8, framealpha=0.3)

    fig.suptitle(
        f"Distance margin per classifier",
        fontsize=13,
    )

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300)
    if show:
        plt.show()
    plt.close(fig)