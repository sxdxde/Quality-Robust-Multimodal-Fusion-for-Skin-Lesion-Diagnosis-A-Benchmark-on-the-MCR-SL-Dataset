"""EfficientNet-B0 image encoder.

Returns both the pooled feature vector (for the late-fusion baseline) and the
last conv feature map (for the channel-gated fusion variant), so a single
forward pass serves both fusion variants without re-running the backbone.
"""
import timm
import torch
import torch.nn as nn


class EfficientNetB0Encoder(nn.Module):
    """ImageNet-pretrained EfficientNet-B0 feature extractor.

    feature_map: (B, 1280, H, W) — last conv output, pre-pooling.
    pooled: (B, 1280) — global-average-pooled feature vector.
    """

    FEATURE_DIM = 1280

    def __init__(self, pretrained: bool = True):
        super().__init__()
        self.backbone = timm.create_model(
            "efficientnet_b0", pretrained=pretrained, num_classes=0, global_pool=""
        )
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor):
        feature_map = self.backbone.forward_features(x)  # (B, 1280, H, W)
        pooled = self.pool(feature_map).flatten(1)  # (B, 1280)
        return pooled, feature_map
