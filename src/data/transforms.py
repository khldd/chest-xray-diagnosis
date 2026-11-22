"""Image transforms. Evaluation transforms must match what the API uses at inference."""
from __future__ import annotations

from torchvision import transforms

from src.config import IMAGENET_MEAN, IMAGENET_STD


def eval_transforms(img_size: int = 224) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def train_transforms(img_size: int = 224, augmentation: str = "flip") -> transforms.Compose:
    """augmentation: 'none' | 'flip' (prototype default) | 'strong'."""
    if augmentation == "none":
        return eval_transforms(img_size)
    steps: list = [transforms.Resize((img_size, img_size))]
    if augmentation == "flip":
        steps.append(transforms.RandomHorizontalFlip(p=0.5))
    elif augmentation == "strong":
        steps += [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.RandomResizedCrop(img_size, scale=(0.85, 1.0)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
        ]
    else:
        raise ValueError(f"Unknown augmentation: {augmentation}")
    steps += [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return transforms.Compose(steps)
