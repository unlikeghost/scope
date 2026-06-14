import pickle
import numpy as np

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

from .prediction import Prediction
from .compression import DissimilarityMatrix


class _SCoPE(ABC):
    def __init__(
        self,
        compressors: List[str],
        join_string: str = ' ',
        keep_similar: bool = False,
        class_weights: Optional[Dict[int, float]] = None,
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
        kw_samples: List[Dict[int, List[str]]],
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
