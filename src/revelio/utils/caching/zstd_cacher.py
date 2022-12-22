import io
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import zstandard as zstd

from .cacher import Cacher

_DECOMPRESS_WRITE_SIZE = 131072
_COMPRESS_WRITE_SIZE = 131591


class ZstdCacher(Cacher):
    def __init__(self, level: int = 10) -> None:
        self._level = level

    def save(self, filename: Path, **data: npt.ArrayLike) -> None:
        cctx = zstd.ZstdCompressor(level=self._level)
        with (
            io.BytesIO() as tmp,
            open(filename, "wb") as f,
            cctx.stream_writer(f, write_size=_COMPRESS_WRITE_SIZE) as compressor,
        ):
            np.savez(tmp, **data)
            tmp.seek(0)
            while True:
                chunk = tmp.read(_COMPRESS_WRITE_SIZE)
                if not chunk:
                    break
                compressor.write(chunk)

    def load(self, filename: Path) -> dict[str, Any]:
        cctx = zstd.ZstdDecompressor()
        with (
            io.BytesIO() as tmp,
            open(filename, "rb") as f,
            cctx.stream_writer(tmp, write_size=_DECOMPRESS_WRITE_SIZE) as decompressor,
        ):
            while True:
                chunk = f.read(_DECOMPRESS_WRITE_SIZE)
                if not chunk:
                    break
                decompressor.write(chunk)
            tmp.seek(0)
            with np.load(tmp) as data:
                d = dict(data)
        return d
