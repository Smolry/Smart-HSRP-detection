import torch.nn as nn
from torchvision import models


def build_model(pretrained=True):
    """
    Build the EfficientNet-B0 binary classifier used in the notebook.

    The original training initialized EfficientNet-B0 with ImageNet
    weights, replaced the classifier with one output, and initially
    froze the feature extractor.
    """
    weights = "IMAGENET1K_V1" if pretrained else None
    model = models.efficientnet_b0(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, 1)
    return model


def freeze_backbone(model):
    for param in model.features.parameters():
        param.requires_grad = False


def unfreeze_last_feature_blocks(model, n_blocks=2):
    for param in model.features[-n_blocks:].parameters():
        param.requires_grad = True
