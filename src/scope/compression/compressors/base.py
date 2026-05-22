# -*- coding: utf-8 -*-
"""
    SCoPE
    Compression Functions
    Jesus Alan Hernandez Galvan
"""
from typing import Union
from abc import ABC, abstractmethod


class BaseCompressor(ABC):
    def __init__(self, compressor_name: str, compression_level: int = 9):

        if not 1 <= compression_level <= 9:
            raise ValueError("Compression level must be between 1 and 9")

        self.encoding : str = "utf-8"

        self._compressor_name: str = compressor_name
        self._compression_level: int = compression_level

    @abstractmethod
    def compress(self, sequence: bytes) -> bytes:
        raise NotImplementedError("This method must be implemented by subclasses")

    def __repr__(self) -> str:
        return f'Compressor(name={self._compressor_name}, level={self._compression_level})'

    def __call__(self, sequence: Union[str, bytes]) -> bytes:
        if len(sequence) == 0:
            raise ValueError(
                f"Empty sequence provided to {self._compressor_name} compressor.",
                "Compression requires non-empty input data."
            )

        if not isinstance(sequence, (bytes, str)):  # noqa
            raise TypeError(
                "Input sequence must be of type 'str' or 'bytes'"
            )

        # Convert string to bytes if necessary
        sequence_encoded = sequence.encode(self.encoding) if isinstance(sequence, str) else sequence

        return self.compress(sequence_encoded)
