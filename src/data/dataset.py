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
        image_path_lookup: dict[str, str],
        transform,
    ):
        self.filenames = df["Image Index"].tolist()
        self.labels = labels
        self.lookup = image_path_lookup
        self.transform = transform

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx: int):
        path = self.lookup[self.filenames[idx]]
        image = Image.open(path).convert("RGB")  # grayscale X-ray -> 3 channels for the CNN
        return self.transform(image), torch.tensor(self.labels[idx], dtype=torch.float32)
