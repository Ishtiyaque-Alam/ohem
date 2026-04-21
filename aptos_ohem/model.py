"""
Model definition – EfficientNet-B4 for 5-class APTOS classification.
"""
import torch
import torch.nn as nn

try:
    import timm
    _USE_TIMM = True
except ImportError:
    _USE_TIMM = False
    import torchvision.models as tv_models


NUM_CLASSES = 5


def build_model(pretrained: bool = True) -> nn.Module:
    """Return EfficientNet-B4 with a 5-class head."""
    if _USE_TIMM:
        model = timm.create_model(
            "efficientnet_b4",
            pretrained=pretrained,
            num_classes=NUM_CLASSES,
        )
    else:
        # Fallback: torchvision EfficientNet-B4 (torch >= 1.11)
        weights = tv_models.EfficientNet_B4_Weights.IMAGENET1K_V1 if pretrained else None
        model   = tv_models.efficientnet_b4(weights=weights)
        in_feat = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_feat, NUM_CLASSES)
    return model


if __name__ == "__main__":
    m = build_model(pretrained=False)
    x = torch.randn(2, 3, 380, 380)
    print("Output shape:", m(x).shape)   # (2, 5)
