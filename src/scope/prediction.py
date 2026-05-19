import numpy as np
from dataclasses import dataclass
from typing import Dict
from shapely.geometry.base import BaseGeometry


@dataclass(frozen=True)
class GeometryMetrics:
    normalized_distance: float
    fraction_outside: float
    iou: float


@dataclass(frozen=True)
class Prediction:
    convex_hull_clusters: Dict[int, BaseGeometry]
    convex_hull_queries: Dict[int, BaseGeometry]
    query_points: Dict[int, np.ndarray]
    scores: Dict[int, float]
    geometry_metrics: Dict[int, GeometryMetrics]
    predicted_class: int
