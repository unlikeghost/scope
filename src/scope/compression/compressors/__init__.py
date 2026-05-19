from enum import Enum
from typing import Union

from .base import BaseCompressor
from .integrated import (
    Bz2Compressor, GZipCompressor,
    ZlibCompressor, LzmaCompressor,
    ZstdCompressor
)

__all__ = [
    'CompressorType',
    'compute_compression',
]

class CompressorType(Enum):
    GZIP = "gzip"
    BZ2 = "bz2"
    ZLIB = "zlib"
    LZMA = "lzma"
    ZSTD = "zstd"


_STRATEGIES = {
    CompressorType.GZIP: GZipCompressor,
    CompressorType.BZ2: Bz2Compressor,
    CompressorType.ZLIB: ZlibCompressor,
    CompressorType.LZMA: LzmaCompressor,
    CompressorType.ZSTD: ZstdCompressor,
}


def get_compressor(name: Union[str, CompressorType]) -> BaseCompressor:
    if isinstance(name, str):
        try:
            compressor_enum = CompressorType(name.lower())
        except ValueError:
            allowed = sorted(c.value for c in CompressorType)
            raise ValueError(
                f"'{name}' is not a valid compressor name.",
                f"Expected one of: {', '.join(allowed)}"
            )
    elif isinstance(name, CompressorType):
        compressor_enum = name
    else:
        raise TypeError("Expected 'name' to be str or CompressorType")

    compressor_class = _STRATEGIES[compressor_enum]
    return compressor_class()


def compute_compression(
    sequence: Union[str, bytes],
    compressor: str,
) -> bytes:
    compressor_instance = get_compressor(
        name=compressor.lower(),
    )

    return compressor_instance(sequence)
