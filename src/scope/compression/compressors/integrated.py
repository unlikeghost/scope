# -*- coding: utf-8 -*-
"""
    SCoPE
    Compression Functions
    Jesus Alan Hernandez Galvan
"""
import compression.zstd as zstd
import gzip
import bz2
import zlib
import lzma

from .base import BaseCompressor


class ZstdCompressor(BaseCompressor):
    def __init__(self):
        super().__init__(
            compressor_name='zstd'
        )
    def compress(self, sequence: bytes) -> bytes:
        return zstd.compress(
            sequence,
            level=9
        )

class LzmaCompressor(BaseCompressor):
    def __init__(self):
        super().__init__(
            compressor_name='lzma',
        )

    def compress(self, sequence: bytes) -> bytes:
        return lzma.compress(
            sequence,
        )


class Bz2Compressor(BaseCompressor):
    def __init__(self, compression_level: int = 9):
        super().__init__(
            compressor_name="bz2",
            compression_level=compression_level,
        )

    def compress(self, sequence: bytes) -> bytes:
        return bz2.compress(
            sequence,
            compresslevel=self._compression_level
        )


class ZlibCompressor(BaseCompressor):
    def __init__(self, compression_level: int = 9):
        super().__init__(
            compressor_name="zlib",
            compression_level=compression_level,
        )

    def compress(self, sequence: bytes) -> bytes:
        return zlib.compress(
            sequence,
            level=self._compression_level
        )


class GZipCompressor(BaseCompressor):
    def __init__(self, compression_level: int = 9):
        super().__init__(
            compressor_name="zlib",
            compression_level=compression_level,
        )

    def compress(self, sequence: bytes) -> bytes:
        return gzip.compress(
            sequence,
            compresslevel=self._compression_level
        )