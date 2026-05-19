import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

from ..prediction import Prediction


def plot_prediction(
    prediction: Prediction,
    plot: bool = False,
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
        ax[index].plot(
            *prediction.convex_hull_queries[key].exterior.xy, # noqa
            color=colors[index],
            linewidth=2,
            linestyle='-',
            label=f"Convex Hull Query"
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
    if plot:
        plt.show()

    if save_path:
        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches='tight'
        )
    plt.close(fig)