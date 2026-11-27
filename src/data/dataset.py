"""PyTorch dataset for NIH ChestX-ray14."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class ChestXrayDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        labels: np.ndarray,
        image_path_lookup,  # dict filename -> path, or src.data.cache.CachedImages
        transform,
    ):
        self.filenames = df["Image Index"].tolist()
        self.labels = labels
        self.lookup = image_path_lookup
        self.transform = transform

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx: int):
        name = self.filenames[idx]
        if hasattr(self.lookup, "open"):  # src.data.cache.CachedImages: pre-resized memmap
            image = self.lookup.open(name)
        else:
            image = Image.open(self.lookup[name]).convert("RGB")  # grayscale X-ray -> 3 channels
        return self.transform(image), torch.tensor(self.labels[idx], dtype=torch.float32)
