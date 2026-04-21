"""
Model definition – MobileNetV3-Large for 5-class APTOS classification.

Chosen for its lightweight footprint (~5 M params) and fast inference,
making it well-suited for Kaggle T4 GPU sessions.
"""
import torch
import torch.nn as nn
import torchvision.models as tv_models


NUM_CLASSES = 5


def build_model(pretrained: bool = True) -> nn.Module:
    """Return MobileNetV3-Large with a 5-class classification head."""
    weights = (
        tv_models.MobileNet_V3_Large_Weights.IMAGENET1K_V2
        if pretrained else None
    )
    model = tv_models.mobilenet_v3_large(weights=weights)

    # Replace the final linear layer (1280 → NUM_CLASSES)
    in_feat = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_feat, NUM_CLASSES)
    return model


if __name__ == "__main__":
    m = build_model(pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    print("Output shape:", m(x).shape)   # (2, 5)
    total = sum(p.numel() for p in m.parameters()) / 1e6
    print(f"Params: {total:.1f} M")
