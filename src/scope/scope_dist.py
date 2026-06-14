import numpy as np
from typing import (
    List, Dict, Any, Optional
)
from .prediction import DistPrediction
from .compression.matrix import (
        _compressor_indexes,
        _dissimilarity_indexes
)

from .model import _SCoPE


class SCoPEDistances(_SCoPE):
    def __init__(
        self,
        compressors: List[str],
        join_string: str = ' ',
        keep_similar: bool = False,
        class_weights: Optional[Dict[int, float]] = None,
        dissimilarity_metrics: List[str] = ['ncd', 'cdm'],  # noqa
    ):
        super().__init__(
            compressors=compressors,
            join_string=join_string,
            keep_similar=keep_similar,
            dissimilarity_metric_names=dissimilarity_metrics,
            class_weights=class_weights
        )

    def compute_classification_score(
        self,
        *args, **kwargs
    ) -> Any:
        ...

    def _get_predicted_class(
        self,
        scores: Dict[int, float],
    ) -> int:
        ...

    @staticmethod
    def _calculate_distance(
        support: np.ndarray,
        query: np.ndarray,
    ) -> np.ndarray:
        return np.linalg.norm(support - query, axis=-1, keepdims=True)

    @staticmethod
    def _calculate_similarity(
        support: np.ndarray,
        query: np.ndarray,
    ) -> np.ndarray:

        dot_product = np.sum(support * query, axis=-1, keepdims=True)

        norm_support = np.linalg.norm(support, axis=-1, keepdims=True)
        norm_query = np.linalg.norm(query, axis=-1, keepdims=True)

        epsilon = 1e-8
        cosine_similarity = dot_product / (norm_support * norm_query + epsilon)

        return cosine_similarity

    def _predict(
        self,
        dissimilarity_matrix: Dict[str, np.ndarray],
    ) -> DistPrediction:

        classifier_labels = [
            f"{c} | {m}"
            for c in sorted(self._dissimilarity_matrix.compressor_names,
                            key=lambda x: _compressor_indexes[x])
            for m in sorted(self._dissimilarity_matrix.dissimilarity_metric_names,
                            key=lambda x: _dissimilarity_indexes[x])
        ]

        distances_per_class = {}
        euclidean_distances = {}
        cosine_distances = {}
        cluster_keys = [k for k in dissimilarity_matrix if 'Cluster' in k]

        for index, key in enumerate(cluster_keys):

            support_data = dissimilarity_matrix[key][:-1].mean(axis=0, keepdims=True)

            query_data = dissimilarity_matrix[key][-1:][0]

            _, n_compressors, n_metrics, _ = support_data.shape

            euc_dist = self._calculate_distance(
                support=support_data,
                query=query_data,
            )

            cos_dist = 1 - self._calculate_similarity(
                support=support_data,
                query=query_data
            )

            distances = euc_dist * cos_dist
            distances_per_class[index] = distances.flatten().tolist()

            euclidean_distances[index] = euc_dist.flatten().tolist()
            cosine_distances[index] = cos_dist.flatten().tolist()

        distances_values = np.array(list(distances_per_class.values())).T

        preds_per_classifier = np.argmin(distances_values, axis=1)

        votes = np.bincount(preds_per_classifier, minlength=len(cluster_keys))

        votes_dict = {
            idx: float(count) if count else 0 for idx, count in enumerate(votes)
        }

        predicted_class = np.argmax(votes)

        return DistPrediction(
            scores=votes_dict,
            predicted_class=predicted_class.item(),
            distances=distances_per_class,
            wining_votes=np.max(votes).item(),
            euclidean_distances=euclidean_distances,
            cosine_distances=cosine_distances,
            classifier_labels=classifier_labels,
            dissimilarity_matrix=dissimilarity_matrix,
        )