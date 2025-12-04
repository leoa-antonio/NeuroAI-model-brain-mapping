import os
from glob import glob

import numpy as np
from PIL import Image

import torch
from torchvision import models
from torchvision.models import ResNet50_Weights


def get_resnet50_model_and_transform():
    """
    Load a pretrained ResNet50 and its corresponding preprocessing transforms.
    """
    weights = ResNet50_Weights.DEFAULT
    model = models.resnet50(weights=weights)
    model.eval()
    preprocess = weights.transforms()
    return model, preprocess


def extract_resnet_avgpool_features(image_paths):
    """
    Extract 2048-dim features from the 'avgpool' layer of ResNet50
    for each image in image_paths.

    Args:
        image_paths: list of paths to image files

    Returns:
        features: numpy array of shape (n_images, 2048)
    """
    model, preprocess = get_resnet50_model_and_transform()

    # This will hold one 2048-dim vector per image
    features_list = []

    # Hook function to capture the avgpool output
    def hook_fn(module, input, output):
        # output has shape (1, 2048, 1, 1) -> flatten to (2048,)
        feat = output.detach().cpu().numpy().reshape(-1)
        features_list.append(feat)

    # Register the hook ONCE, before looping over images
    handle = model.avgpool.register_forward_hook(hook_fn)

    # Loop over images and run a forward pass for each
    for path in image_paths:
        img = Image.open(path).convert("RGB")
        x = preprocess(img).unsqueeze(0)  # shape (1, 3, 224, 224)

        with torch.no_grad():
            _ = model(x)

    # Remove the hook after we're done
    handle.remove()

    # Stack all feature vectors into (n_images, 2048)
    features = np.stack(features_list, axis=0)
    return features


if __name__ == "__main__":
    # Simple test: extract features from all images in test_images/
    image_paths = sorted(glob(os.path.join("test_images", "*")))
    print("Found", len(image_paths), "images")

    if len(image_paths) == 0:
        print("Put some .jpg or .png files into encoding_models/test_images/ and run again.")
        raise SystemExit

    features = extract_resnet_avgpool_features(image_paths)
    print("Feature matrix shape:", features.shape)  # (n_images, 2048)
