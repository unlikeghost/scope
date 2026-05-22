import pickle
import numpy as np

from warnings import warn
from shapely import convex_hull
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Any
from shapely.geometry import Point, MultiPoint, Polygon
from shapely.geometry.base import BaseGeometry

from .prediction import Prediction, GeometryMetrics, PoligonPrediction, DistPrediction
from .compression import DissimilarityMatrix
from .compression.matrix import _compressor_indexes, _dissimilarity_indexes


class _SCoPE(ABC):
    def __init__(
        self,
        compressors: List[str],
        join_string: str = ' ',
        keep_similar: bool = False,
        class_weights: Dict[int, float] = None,
        dissimilarity_metric_names: List[str] = ['ncd', 'cdm'], # noqa
    ):
        self._dissimilarity_matrix = DissimilarityMatrix(
            compressor_names=compressors,
            dissimilarity_metric_names=dissimilarity_metric_names,
            join_string=join_string,
            keep_similar=keep_similar,
        )
        self._class_weights = class_weights

    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str):
        with open(path, 'rb') as f:
            return pickle.load(f)

    def predict(
        self,
        kw_samples: List[Dict[int, str]],
        queries: List[str],
    ) -> List[Prediction]:
        dissimilarity_matrices = self._dissimilarity_matrix(
            queries=queries,
            supports=kw_samples,
        )

        predictions = [
            self._predict(dissimilarity_matrix=dissimilarity_matrix)
            for dissimilarity_matrix in dissimilarity_matrices
        ]

        return predictions

    @abstractmethod
    def _predict(
        self,
        dissimilarity_matrix: Dict[str, np.ndarray],
    ) -> Any:
        ...

    @abstractmethod
    def _get_predicted_class(
        self,
        scores: Dict[int, float],
    ) -> int:
        ...


class SCoPEPoligon(_SCoPE):

    def __init__(
        self,
        compressors: List[str],
        join_string: str = ' ',
        keep_similar: bool = False,
        dissimilarity_metrics: List[str] = ['ncd', 'cdm'],  # noqa
    ):
        if len(dissimilarity_metrics) != 2:
            raise ValueError(
                "SCoPEPoligon only supports two dissimilarity metrics."
            )
        super().__init__(
            compressors=compressors,
            join_string=join_string,
            keep_similar=keep_similar,
            dissimilarity_metric_names=dissimilarity_metrics
        )

    def compute_classification_score(
        self,
        convex_hull_cluster: Polygon,
        convex_hull_query: Polygon,
        query_points: np.ndarray
    ) -> Tuple[float, GeometryMetrics]:
        """
           Score más bajo = query más parecida al cluster.
           1.- normalized_distance — mide qué tan lejos está el centroide de la query del centroide del cluster,
            pero escalado por el "radio equivalente" del cluster (√(área/π)).
            Sin normalizar, un cluster grande siempre parecería "más cercano" aunque la query esté en su periferia.

           2.- fraction_outside — cuenta qué fracción de los puntos de la query
            (las combinaciones NCD×CDM con cada compresor) cae fuera del convex hull del cluster.
            Va de 0.0 (todos dentroIo) a 1.0 (todos fuera). Es una penalización directa:
            si los puntos no caben en el espacio del cluster, algo está mal.

           3.- IoU — mide el solapamiento geométrico entre ambos convex hulls.
            Va en el denominador, entonces cuanto más se solapan las formas, más baja el score.
            Captura algo que las otras dos métricas no ven:
            puedes tener centroides cercanos pero distribuciones muy distintas (un hull grande y uno pequeño sin solapamiento real).

            La combinación es importante porque cada métrica sola tiene puntos ciegos:
            centroides cercanos pero query fuera del hull,
            o muchos puntos dentro pero formas incompatibles.
            Las tres juntas se compensan.
        """

        cluster_area = convex_hull_cluster.area
        cluster_radius = np.sqrt(cluster_area / np.pi)

        centroid_distance = convex_hull_cluster.centroid.distance(
            convex_hull_query.centroid
        )
        normalized_distance = centroid_distance / (cluster_radius + 1e-9)

        # Convertir a objetos Point aquí
        points = [Point(p[0], p[1]) for p in query_points]

        n_outside = sum(
            1 for p in points
            if not convex_hull_cluster.covers(p)
        )
        fraction_outside = n_outside / len(points)

        intersection_area = convex_hull_cluster.intersection(convex_hull_query).area
        union_area = convex_hull_cluster.union(convex_hull_query).area
        iou = intersection_area / (union_area + 1e-9)

        score = (normalized_distance + fraction_outside) / (iou + 1e-9)

        return score, GeometryMetrics(
            normalized_distance=normalized_distance,
            fraction_outside=fraction_outside,
            iou=iou
        )

    def _get_predicted_class(
        self,
        scores: Dict[int, float],
    ) -> int:
        return max(scores, key=scores.get)

    @staticmethod
    def _get_data(
        dissimilarity_matrix: np.ndarray,
    ) -> Tuple[BaseGeometry, BaseGeometry, np.ndarray]:
        support_data = dissimilarity_matrix[:-1].transpose(0, 2, 1, 3)

        query_data = dissimilarity_matrix[-1:][0].transpose(1, 0, 2)

        query_array = query_data.reshape(
            query_data.shape[0], -1
        ).T

        support_array = support_data.reshape(
            support_data.shape[0], -1
        ).T

        if support_array.shape[-1] < 2:
            warn("This method just works for 2 or more support samples.")
            return None, None, None

        support_points = np.vstack([
            poly.T
            for poly in support_array
        ])

        cluster_points = MultiPoint(support_points)
        query_points = MultiPoint(query_array)
        convex_hull_cluster = convex_hull(cluster_points)
        convex_hull_query = convex_hull(query_points)

        return convex_hull_cluster, convex_hull_query, query_array

    def _predict(
        self,
        dissimilarity_matrix: Dict[str, np.ndarray],
    ) -> PoligonPrediction:

        scores = {}
        query_points = {}
        convex_hull_queries = {}
        convex_hull_clusters = {}
        geometric_metrics = {}

        cluster_keys = [k for k in dissimilarity_matrix if 'Cluster' in k]

        for index, key in enumerate(cluster_keys):

            convex_hull_cluster, convex_hull_query, query_points_array = self._get_data(
                dissimilarity_matrix=dissimilarity_matrix[key],
            )

            if convex_hull_cluster is None:
                geometric_metrics[index] = GeometryMetrics(0.0, 1.0, 0.0)
                scores[index] = -float('inf')
                query_points[index] = None
                convex_hull_queries[index] = None
                convex_hull_clusters[index] = None

                continue

            score, geo = self.compute_classification_score(
                convex_hull_cluster=convex_hull_cluster,
                convex_hull_query=convex_hull_query,
                query_points=query_points_array,
            )

            geometric_metrics[index] = geo
            scores[index] = -score
            query_points[index] = query_points_array
            convex_hull_queries[index] = convex_hull_query
            convex_hull_clusters[index] = convex_hull_cluster

        predicted_class = self._get_predicted_class(scores=scores)

        return PoligonPrediction(
            scores=scores,
            predicted_class=predicted_class,
            convex_hull_clusters=convex_hull_clusters,
            convex_hull_queries=convex_hull_queries,
            query_points=query_points,
            geometry_metrics=geometric_metrics,
        )


class SCoPEDistances(_SCoPE):
    def __init__(
        self,
        compressors: List[str],
        join_string: str = ' ',
        keep_similar: bool = False,
        class_weights: Dict[int, float] = None,
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
        )