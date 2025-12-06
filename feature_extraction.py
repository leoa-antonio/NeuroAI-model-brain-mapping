import os
from typing import List, Union
from pathlib import Path

import numpy as np
from PIL import Image

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights

# Optional progress bar: if tqdm not installed, we just fall back to a no-op wrapper
try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(x, *args, **kwargs):
        return x


# ---------------------------
# Utility: choose best device
# ---------------------------

def _choose_device(explicit: Union[str, None] = None) -> torch.device:
    """
    Choose the best available device.

    Priority:
        1. explicit argument, if given
        2. CUDA
        3. MPS (Apple Silicon)
        4. CPU
    """
    if explicit is not None:
        return torch.device(explicit)

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------
# Dataset for image paths
# ---------------------------

class ImagePathDataset(Dataset):
    """
    Dataset that loads images from file paths and applies torchvision transforms.
    """

    def __init__(self, image_paths: List[Union[str, Path]], transform=None):
        self.image_paths = [Path(p) for p in image_paths]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        path = self.image_paths[idx]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img


# ---------------------------
# Core: batched feature extraction
# ---------------------------

def extract_resnet_avgpool_features_batched(
    image_paths: List[Union[str, Path]],
    batch_size: int = 32,
    device: Union[str, None] = None,
    num_workers: int = 2,
    use_tqdm: bool = True,
) -> np.ndarray:
    """
    Fast, batched ResNet50 'avgpool' feature extraction.

    Parameters
    ----------
    image_paths : list of str or Path
        Paths to image files.
    batch_size : int, default=32
        Number of images per forward pass.
    device : {'cuda', 'mps', 'cpu'} or None, default=None
        Which device to use. If None, picks CUDA, then MPS, then CPU.
    num_workers : int, default=2
        Number of DataLoader workers for parallel image loading.
        On macOS, small values like 0–2 are usually safest.
    use_tqdm : bool, default=True
        If True and tqdm is available, show a progress bar.

    Returns
    -------
    features : np.ndarray
        Feature matrix of shape (N_images, 2048), where each row is the
        ResNet50 avgpool activation for one image.
    """
    if len(image_paths) == 0:
        raise ValueError("image_paths is empty; nothing to extract.")

    device = _choose_device(device)
    print(f"[feature_extraction] Using device: {device}")

    # Load pretrained ResNet50 and its transforms
    weights = ResNet50_Weights.DEFAULT
    resnet = models.resnet50(weights=weights)

    # Drop the final fully-connected layer; keep up to avgpool
    feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
    feature_extractor.to(device)
    feature_extractor.eval()

    preprocess = weights.transforms()

    # Dataset + DataLoader
    dataset = ImagePathDataset(image_paths, transform=preprocess)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    all_features = []
    iterator = loader
    if use_tqdm:
        iterator = tqdm(loader, desc="Extracting ResNet50 features", leave=False)

    with torch.no_grad():
        for batch in iterator:
            batch = batch.to(device, non_blocking=True)
            feats = feature_extractor(batch)       # (B, 2048, 1, 1)
            feats = feats.view(feats.size(0), -1)  # (B, 2048)
            all_features.append(feats.cpu().numpy())

    features = np.concatenate(all_features, axis=0)

    if features.shape[0] != len(image_paths):
        raise RuntimeError(
            f"Expected {len(image_paths)} feature rows, "
            f"but got {features.shape[0]}"
        )

    print(f"[feature_extraction] Done. Feature matrix shape: {features.shape}")
    return features


# ---------------------------
# Simple wrapper (compatibility)
# ---------------------------

def extract_resnet_avgpool_features(image_paths: List[Union[str, Path]]) -> np.ndarray:
    """
    Backwards-compatible wrapper around the batched extractor.

    Uses a default batch_size=32 and automatic device selection.
    """
    return extract_resnet_avgpool_features_batched(
        image_paths,
        batch_size=32,
        device=None,        # auto: cuda > mps > cpu
        num_workers=2,
        use_tqdm=True,
    )


# ---------------------------
# Quick manual test
# ---------------------------

if __name__ == "__main__":
    # Simple test: extract features from all images in test_images/
    test_dir = Path("test_images")
    image_paths = sorted(test_dir.glob("*"))

    print("Found", len(image_paths), "images in", test_dir)

    if len(image_paths) == 0:
        print("Put some .jpg or .png files into encoding_models/test_images/ and run again.")
        raise SystemExit

    feats = extract_resnet_avgpool_features_batched(
        image_paths,
        batch_size=16,
        device=None,       # auto-select
        num_workers=0,
        use_tqdm=True,
    )
    print("Feature matrix shape:", feats.shape)  # (n_images, 2048)
