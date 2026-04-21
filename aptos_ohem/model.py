"""
Model definition: MobileNetV3-Large for image classification.
"""
import torch
import torch.nn as nn
import torchvision.models as tv_models


def build_model(num_classes: int, pretrained: bool = True) -> nn.Module:
    """Return MobileNetV3-Large with a configurable classification head."""
    weights = (
        tv_models.MobileNet_V3_Large_Weights.IMAGENET1K_V2
        if pretrained
        else None
    )
    model = tv_models.mobilenet_v3_large(weights=weights)

    # Replace the final linear layer (1280 -> num_classes)
    in_feat = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_feat, num_classes)
    return model


if __name__ == "__main__":
    m = build_model(num_classes=7, pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    print("Output shape:", m(x).shape)
    total = sum(p.numel() for p in m.parameters()) / 1e6
    print(f"Params: {total:.1f} M")
