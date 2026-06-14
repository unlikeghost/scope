import numpy as np
from dataclasses import dataclass
from typing import Dict, List
from shapely.geometry.base import BaseGeometry


@dataclass(frozen=True)
class GeometryMetrics:
    normalized_distance: float
    fraction_outside: float
    iou: float


@dataclass(frozen=True)
class Prediction:
    scores: Dict[int, float]
    predicted_class: int
    dissimilarity_matrix:  Dict[str, np.ndarray]


@dataclass(frozen=True)
class PolygonPrediction(Prediction):
    convex_hull_clusters: Dict[int, BaseGeometry]
    convex_hull_queries: Dict[int, BaseGeometry]
    query_points: Dict[int, np.ndarray]
    geometry_metrics: Dict[int, GeometryMetrics]


@dataclass(frozen=True)
class DistPrediction(Prediction):
    euclidean_distances: Dict[int, list[float]]
    cosine_distances: Dict[int, list[float]]
    distances: Dict[int, list[float]]
    wining_votes: int
    classifier_labels: List[str]