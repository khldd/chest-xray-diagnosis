"""Pre-resized image cache: decode each 1024 px PNG once, store it at the training size.

Training is limited by PNG decoding, not by the GPU. Every transform we use starts with
Resize((S, S)), so resizing once up front with the same operation gives identical tensors:
Pillow returns an unchanged copy when an image is resized to its own size.

The cache is one uint8 memmap [n_images, S, S] (grayscale: X-rays have equal R, G, B), plus
an index {filename: row}. 112,120 images at 224 px take about 5.6 GB.
"""
from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from PIL import Image
from torchvision import transforms

_WORKER: dict = {}


def _init_worker(path: str, n: int, size: int) -> None:
    _WORKER["arr"] = np.memmap(path, dtype=np.uint8, mode="r+", shape=(n, size, size))
    _WORKER["resize"] = transforms.Resize((size, size))  # exactly what the transforms do


def _fill(job: tuple[int, str]) -> int:
    row, src = job
    image = Image.open(src).convert("RGB")  # same conversion as ChestXrayDataset
    _WORKER["arr"][row] = np.asarray(_WORKER["resize"](image))[:, :, 0]
    return row


def build_cache(
    filenames: list[str], lookup: dict[str, str], size: int, cache_dir: str, workers: int = 4
) -> tuple[str, dict[str, int]]:
    """Create (or reuse) the cache for `filenames`. Returns (memmap path, filename -> row)."""
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"images_{size}.u8")
    index_path = os.path.join(cache_dir, f"index_{size}.json")
    if os.path.exists(index_path):
        with open(index_path, encoding="utf-8") as f:
            index = json.load(f)
        if set(filenames) <= set(index):
            return path, index

    names = sorted(set(filenames))
    index = {name: i for i, name in enumerate(names)}
    np.memmap(path, dtype=np.uint8, mode="w+", shape=(len(names), size, size)).flush()
    jobs = [(i, lookup[name]) for i, name in enumerate(names)]
    init = (path, len(names), size)
    with ProcessPoolExecutor(workers, initializer=_init_worker, initargs=init) as ex:
        for done, _ in enumerate(ex.map(_fill, jobs, chunksize=256), start=1):
            if done % 10000 == 0:
                print(f"cache: {done}/{len(jobs)} images", flush=True)
    with open(index_path, "w", encoding="utf-8") as f:  # written last: marks the cache complete
        json.dump(index, f)
    return path, index


class CachedImages:
    """Dict-like filename -> PIL RGB image, read from the memmap (opened lazily per worker)."""

    def __init__(self, path: str, index: dict[str, int], size: int):
        self.path, self.index, self.size = path, index, size
        self._arr = None

    def open(self, filename: str) -> Image.Image:
        if self._arr is None:
            shape = (len(self.index), self.size, self.size)
            self._arr = np.memmap(self.path, dtype=np.uint8, mode="r", shape=shape)
        return Image.fromarray(np.array(self._arr[self.index[filename]])).convert("RGB")

    def __getstate__(self):  # DataLoader workers re-open the memmap themselves
        state = self.__dict__.copy()
        state["_arr"] = None
        return state
