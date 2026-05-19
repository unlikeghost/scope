# -*- coding: utf-8 -*-
"""
    SCoPE
    Dissimilarity Matrix
    Jesus Alan Hernandez Galvan
"""
import time
import numpy as np
from functools import lru_cache
from itertools import product as combine
# from itertools import combinations_with_replacement as combine
from abc import (
    abstractmethod,
    ABC
)
from typing import (
    Union,
    List,
    Dict
)
from .compressors import compute_compression
from .metrics import (
    _ncd, _cdm, _clm
)

_compressor_indexes = {
    'gzip': 0,
    'bz2': 1,
    'zlib': 2,
    'lzma': 3,
}

_dissimilarity_indexes = {
    'ncd': 0,
    'cdm': 1,
    'clm': 2,
}

_dissimilarity_fns = {
    'ncd': lambda c_x1, c_x2, c_x1x2: _ncd(c_x1, c_x2, c_x1x2),
    'cdm': lambda c_x1, c_x2, c_x1x2: _cdm(c_x1, c_x2, c_x1x2),
    'clm': lambda c_x1, c_x2, c_x1x2: _clm(c_x1, c_x2, c_x1x2),
}

@lru_cache(maxsize=None)
def _compute_compression_size(
    sequence: Union[str, bytes],
    compressor: str
) -> int:

    if len(sequence) == 0:
        raise ValueError(f"WARNING: Empty sequence for compression with {compressor}")

    compressed_sequence = compute_compression(
        sequence=sequence,
        compressor=compressor,
    )
    return len(compressed_sequence)



class DissimilarityMatrixBase(ABC):
    cluster_key: str= 'SCoPE_Cluster_'
    sample_key: str= 'SCoPE_Sample_'

    @staticmethod
    def _validate_compressor_names(compressor_names):
        invalid_compressors = [c for c in compressor_names if c not in _compressor_indexes]
        if invalid_compressors:
            raise ValueError(
                f"Invalid compressor(s): {', '.join(invalid_compressors)}. "
                f"Valid options are: {', '.join(_compressor_indexes.keys())}"
            )

    @staticmethod
    def _validate_metric_names(compression_metric_names):
        invalid_metrics = [m for m in compression_metric_names if m not in _dissimilarity_indexes]
        if invalid_metrics:
            raise ValueError(
                f"Invalid compression metric(s): {', '.join(invalid_metrics)}. "
                f"Valid options are: {', '.join(_dissimilarity_indexes)}"
            )

    @staticmethod
    def _validate_join_str_value(join_string):
        if not isinstance(join_string, str):
            raise ValueError(
                f"Invalid join string: {join_string}. join_string must be a string object."
            )

    def _validate_args(self, compressor_names, compression_metric_names, join_string):
        self._validate_compressor_names(compressor_names)
        self._validate_metric_names(compression_metric_names)
        self._validate_join_str_value(join_string)

    def __init__(self,
        compressor_names: Union[str, List[str]] = 'gzip',
        dissimilarity_metric_names: Union[str, List[str]] = 'ncd',
        join_string: str = ' ',
    ):
        if compressor_names == 'all':
            compressor_names = list(_compressor_indexes.keys())
        if dissimilarity_metric_names == 'all':
            dissimilarity_metric_names = list(_dissimilarity_indexes.keys())

        if isinstance(compressor_names, str):
            compressor_names = [compressor_names]

        if isinstance(dissimilarity_metric_names, str):
            dissimilarity_metric_names = [dissimilarity_metric_names]

        self._validate_args(compressor_names, dissimilarity_metric_names, join_string)

        self.dissimilarity_metric_names = set(sorted(dissimilarity_metric_names))
        self.compressor_names = set(sorted(compressor_names))
        self.join_string = join_string

        self._total_compressors = len(_compressor_indexes)
        self._total_metrics = len(_dissimilarity_indexes)

        self._n_compressors = len(self.compressor_names)
        self._n_metrics = len(self.dissimilarity_metric_names)

        self._index_compressors: list = [_compressor_indexes[c] for c in self.compressor_names]
        self._index_metrics: list = [_dissimilarity_indexes[m] for m in self.dissimilarity_metric_names]

    @abstractmethod
    def _compute_one(self,
        query: str,
        support: List[str]
    ) -> np.ndarray:
        raise NotImplementedError()

    def _compute_matrix(self,
        query: str,
        supports: Dict[Union[int, str], List[str]],
    ) -> Dict[str, np.ndarray]:
        start_ = time.perf_counter()

        _output = {}

        for key in supports:
            support: List[str] = supports[key]

            cluster_dissimilarity: np.ndarray = self._compute_one(
                query=query,
                support=support
            )

            cluster = cluster_dissimilarity[:-1, :]
            sample = cluster_dissimilarity[-1:, :]

            _output[f'{self.cluster_key}{key}'] = cluster
            _output[f'{self.sample_key}{key}'] = sample
        end_ = time.perf_counter()

        output_time = end_ - start_
        _output['DM_Time'] = output_time
        return  _output

    def __call__(self,
         queries: Union[
             List[str],
             str
         ],
         supports: Union[
             List[Dict[int, List[str]]],
             Dict[int, List[str]]
         ]
    ) -> List[Dict[str, np.ndarray]]:

        if not isinstance(queries, (list, str)):
            raise ValueError(
                f"'queries' must be a string or list of strings. "
                f"Got {type(queries)} instead."
            )

        if not isinstance(supports, (list, dict)):
            raise ValueError(
                "'supports' must be a dictionary or list of dictionaries. "
            )

        if isinstance(queries, str):
            queries = [queries]

        if isinstance(supports, dict):
            supports = [supports]

        if len(queries) != len(supports):
            raise ValueError(
                f"'samples' and 'kw_samples' must have the same length "
                f"(got {len(queries)} and {len(supports)})."
            )

        result = [
            self._compute_matrix(query, support)
            for query, support in zip(queries, supports)
        ]

        return result

    def to_dict(self):
        return {
            k: v
            for k, v in self.__dict__.items()
            if not callable(v) and not k.startswith("_")
        }


class DissimilarityMatrixV1(DissimilarityMatrixBase):
    def __init__(
        self,
        compressor_names: Union[str, List[str]] = 'gzip',
        dissimilarity_metric_names: Union[str, List[str]] = 'ncd',
        join_string: str = ' ',
        keep_similar: bool = False,
        epsilon: float = 1e-10,
    ):
        super().__init__(
            compressor_names=compressor_names,
            dissimilarity_metric_names=dissimilarity_metric_names,
            join_string=join_string
        )
        self.epsilon = epsilon
        self.keep_similar = keep_similar

    def _compute_one(self,
        query: str,
        support: List[str]
    ) -> np.ndarray:

        _output = np.full(
            shape=(
                len(support) + 1,
                self._total_compressors,
                self._total_metrics,
                len(support) + 1,
            ),
            fill_value=np.nan
        )
        support = sorted(support, key=len, reverse=True)

        support_and_query = support + [query]

        combination_index = combine(
            range(len(support_and_query)),
            repeat=2
        )

        for xi, xj in combination_index:
            x1: str = support_and_query[xi]
            x2: str = support_and_query[xj]

            if (x1 == x2) and not self.keep_similar:
                continue

            for _compressor in self.compressor_names:
                c_i: int = _compressor_indexes[_compressor]
                c_x1: int = _compute_compression_size(x1, _compressor)
                c_x2: int = _compute_compression_size(x2, _compressor)
                c_x1x2: int = _compute_compression_size(f'{x1}{self.join_string}{x2}', _compressor)
                c_x2x1: int = _compute_compression_size(f'{x2}{self.join_string}{x1}', _compressor)

                for _metric in self.dissimilarity_metric_names:
                    m_i = _dissimilarity_indexes[_metric]
                    score_x1x2: float = _dissimilarity_fns[_metric](
                        c_x1=c_x1,
                        c_x2=c_x2,
                        c_x1x2=c_x1x2,
                    )
                    score_x2x1: float = _dissimilarity_fns[_metric](
                        c_x1=c_x2,
                        c_x2=c_x1,
                        c_x1x2=c_x2x1,
                    )

                    _output[xi, c_i, m_i, xj] = score_x1x2
                    _output[xj, c_i, m_i, xi] = score_x2x1

        nan_mask_matrix = ~np.isnan(_output)

        n_items = len(support)

        if self.keep_similar:
            n_features = self._n_compressors * (n_items + 1)
        else:
            n_features = self._n_compressors * n_items

        result = _output[nan_mask_matrix]

        result_matrix = result.reshape(
            n_items + 1,
            self._n_metrics,
            n_features
        )

        return result_matrix