import os
from glob import glob

import numpy as np
from PIL import Image

import torch
from torchvision import models, transforms

def get_resnet50_model():
    """
    Load a pretrained ResNet50 model.
    We will use it only as a feature extractor (no training).
    """
    model = models.resnet50(pretrained=True)
    model.eval()
    return model

def get_preprocess_transform():
    """
    Standard ImageNet preprocessing for ResNet.
    """
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

def extract_resnet_avgpool_features(image_paths):
    """
    Extract 2048-dim features from the 'avgpool' layer of ResNet50
    for each image in image_paths.

    Returns:
        features: numpy array of shape (n_images, 2048)
    """
    model = get_resnet50_model()
    preprocess = get_preprocess_transform()

    # We will hook into the 'avgpool' layer
    layer_name = "avgpool"
    features_list = []

    def hook_fn(module, input, output):
        # output has shape (1, 2048, 1, 1) -> flatten to (2048,)
        feat = output.detach().cpu().numpy().reshape(-1)
        features_list.append(feat)

    # Register hook
    handle = dict(model.named_modules())[layer_name].register_forward_hook(hook_fn)

    for path in image_paths:
        img = Image.open(path).convert("RGB")
        x = preprocess(img).unsqueeze(0)  # shape (1, 3, 224, 224)
        features_list.clear()
        with torch.no_grad():
            _ = model(x)
        # features_list[0] now contains the features for this image
        features_list[0].setflags(write=False)  # just to be safe
        # Collect a copy
        feat_copy = np.array(features_list[0], copy=True)
        features_list.append(feat_copy)

    # Remove the hook
    handle.remove()

    # Stack all feature vectors
    # NOTE: we stored each feature vector twice above, but we only want the last copy
    # So just take every second element: indices 1, 3, 5, ...
    feats = np.array(features_list[1::2])
    return feats

if __name__ == "__main__":
    # Simple test: extract features from all images in test_images/
    image_paths = sorted(glob(os.path.join("test_images", "*")))
    print("Found", len(image_paths), "images")

    if len(image_paths) == 0:
        print("Put some .jpg or .png files into encoding_models/test_images/ and run again.")
        exit()

    features = extract_resnet_avgpool_features(image_paths)
    print("Feature matrix shape:", features.shape)  # (n_images, 2048)
