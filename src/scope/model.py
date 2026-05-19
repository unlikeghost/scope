import pickle
import numpy as np

from shapely import convex_hull
from typing import List, Dict, Tuple
from shapely.geometry import Point, MultiPoint, Polygon
from shapely.geometry.base import BaseGeometry

from .prediction import Prediction, GeometryMetrics
from .compression import DissimilarityMatrix


class SCoPE:

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

    @staticmethod
    def compute_classification_score(
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

    @staticmethod
    def _get_data(
        dissimilarity_matrix: Dict[str, np.ndarray],
        key: str,
    ) -> Tuple[BaseGeometry, BaseGeometry, np.ndarray]:

        query_key = key.replace('Cluster', 'Sample')
        cluster_key = key

        cluster_data = dissimilarity_matrix[cluster_key]
        query_points_array = dissimilarity_matrix[query_key][0].T

        cluster_points = np.vstack([
            poly.T
            for poly in cluster_data
        ])

        cluster_points = MultiPoint(cluster_points)
        query_points = MultiPoint(query_points_array)

        convex_hull_cluster = convex_hull(cluster_points)
        convex_hull_query = convex_hull(query_points)

        return convex_hull_cluster, convex_hull_query, query_points_array

    @staticmethod
    def _get_predicted_class(
        scores: Dict[int, float],
    ) -> int:
        return max(scores, key=scores.get)

    def _predict(
        self,
        dissimilarity_matrix: Dict[str, np.ndarray],
    ) -> Prediction:

        scores = {}
        query_points = {}
        convex_hull_queries = {}
        convex_hull_clusters = {}
        geometric_metrics = {}

        cluster_keys = [k for k in dissimilarity_matrix if 'Cluster' in k]

        for index, key in enumerate(cluster_keys):
            convex_hull_cluster, convex_hull_query, query_points_array = self._get_data(
                dissimilarity_matrix=dissimilarity_matrix,
                key=key,
            )
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

        if self._class_weights:
            scores = {
                cls: score / self._class_weights[cls]
                for cls, score in scores.items()
            }

        predicted_class = self._get_predicted_class(scores=scores)

        return Prediction(
            convex_hull_clusters=convex_hull_clusters,
            convex_hull_queries=convex_hull_queries,
            query_points=query_points,
            scores=scores,
            predicted_class=predicted_class,
            geometry_metrics=geometric_metrics,
        )

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

    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str):
        with open(path, 'rb') as f:
            return pickle.load(f)