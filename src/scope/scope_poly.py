from traceback import print_tb

import numpy as np

from shapely import convex_hull
from typing import List, Dict, Tuple
from shapely.geometry import Point, MultiPoint, Polygon
from shapely.geometry.base import BaseGeometry

from .prediction import GeometryMetrics, PolygonPrediction
from .model import _SCoPE


class SCoPEPolygon(_SCoPE):

    def __init__(
        self,
        compressors: List[str],
        join_string: str = ' ',
        keep_similar: bool = False,
        dissimilarity_metrics: List[str] = ['ncd', 'cdm'],  # noqa
    ):
        if len(dissimilarity_metrics) != 2:
            raise ValueError(
                "SCoPEPolygon only supports two dissimilarity metrics."
            )
            
        super().__init__(
            compressors=compressors,
            join_string=join_string,
            keep_similar=keep_similar,
            dissimilarity_metric_names=dissimilarity_metrics
        )

    def compute_classification_score(
        self,
        convex_hull_cluster: BaseGeometry,
        convex_hull_query: BaseGeometry,
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

        ns, nc, nm, nf = dissimilarity_matrix.shape
        dissimilarity_matrix = dissimilarity_matrix.transpose(0, 2, 1, 3)

        dissimilarity_matrix_reshaped = dissimilarity_matrix.reshape(ns, nm, nc * nf)

        # support_data = dissimilarity_matrix_reshaped[:-1].mean(axis=0, keepdims=True)
        support_data = dissimilarity_matrix_reshaped[:-1]

        query_data = dissimilarity_matrix_reshaped[-1:]

        support_points = np.vstack([
            poly.T
            for poly in support_data
        ])
        
        query_points = query_data[0].T

        if support_data.shape[-1] < 2:
            raise ValueError("This method just works for 2 or more support samples.")

        support_points = MultiPoint(support_points)
        query_points_data = MultiPoint(query_points)

        convex_hull_cluster = convex_hull(support_points)
        convex_hull_query = convex_hull(query_points_data)

        return convex_hull_cluster, convex_hull_query, query_points

    def _predict(
        self,
        dissimilarity_matrix: Dict[str, np.ndarray],
    ) -> PolygonPrediction:

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

        return PolygonPrediction(
            scores=scores,
            predicted_class=predicted_class,
            convex_hull_clusters=convex_hull_clusters,
            convex_hull_queries=convex_hull_queries,
            query_points=query_points,
            geometry_metrics=geometric_metrics,
            dissimilarity_matrix=dissimilarity_matrix,
        )