# -*- coding: utf-8 -*-
"""
    SCoPE
    Sampling
    Jesus Alan Hernandez Galvan
"""
import warnings
import numpy as np
from typing import (
    Generator,
    Tuple
)


class SampleGenerator:
    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        seed: int = 42,
    ):
        self.seed = seed
        rng = np.random.RandomState(seed)

        if x.shape[0] != y.shape[0]:
            raise ValueError("x and y must have the same number of samples.")

        if y.ndim > 1:
            y = np.argmax(y, axis=1)

        self.unique_classes = np.unique(y)
        self.class_counts = {
            cls: int(np.sum(y == cls))
            for cls in self.unique_classes
        }

        indices_order = np.arange(len(x))
        rng.shuffle(indices_order)

        min_class_size = min(self.class_counts.values())
        self._available_samples_per_class = min_class_size - 1
        self._indices_order = indices_order
        self._x = x
        self._y = y
        self._num_samples = None

    def sampling(
        self,
        num_samples: int
    ) -> Generator[tuple[int, str, int, dict[int, list[str]]], None]:
        if num_samples <= 0:
            raise ValueError("num_samples must be greater than 0.")

        if num_samples > self._available_samples_per_class:
            warnings.warn(
                f"Requested {num_samples} samples, but smallest class has only "
                f"{self._available_samples_per_class} available samples (excluding target). "
                f"num_samples will be set to {self._available_samples_per_class}.",
                UserWarning,
            )
            num_samples = self._available_samples_per_class

        self._num_samples = num_samples
        return iter(self)

    def __iter__(self) -> Generator[tuple[int, str, int, dict[int, list[str]]], None]:
        for index in range(self._x.shape[0]):
            sample_to_predict, expected_label, current_kw_samples = self.__getitem__(index)
            yield index, sample_to_predict, expected_label, current_kw_samples

    def __getitem__(self, idx: int) -> Tuple[str, int, dict[int, list[str]]]:
        if self._x is None or self._y is None:
            raise ValueError("Sampling must be performed before accessing samples.")
        if idx >= len(self._indices_order):
            raise IndexError("Index out of range.")

        pos: int = self._indices_order[idx]

        expected_label: int = int(self._y[pos])
        sample_to_predict: str = self._x[pos]

        local_rng = np.random.RandomState(self.seed + idx)

        current_kw_samples: dict = {
            int(cls): [] for cls in sorted(self.unique_classes)
        }
        for cls in sorted(self.unique_classes):
            mask = np.where(self._y == cls)[0]
            mask = mask[mask != pos]

            sampled_indices = local_rng.choice(
                mask, size=self._num_samples, replace=False
            )

            current_kw_samples[int(cls)] = sorted(self._x[sampled_indices].tolist())

        return sample_to_predict, expected_label, current_kw_samples

    def __len__(self) -> int:
        return len(self._indices_order)