"""Assembles the full model for a given ablation variant.

Variants (see CLAUDE.md ablation matrix):
  1. image_only        — EfficientNet-B0 pooled vector -> binary head only.
  2. late_fusion        — + metadata, concat -> MLP -> heads.
  3. channel_gated       — + metadata, SE-style channel gating -> heads. (main method)
  4. text_channel_gated  — stretch ablation; same as (3) but metadata_encoder
                           is swapped for a frozen sentence-encoder wrapper.
                           Not built until the stretch ablation is started.

`quality_aware=True` attaches the QualityHead for robustness analysis #2.
"""
import torch
import torch.nn as nn

from models.fusion import FUSION_REGISTRY, LateFusion
from models.heads import AuxDiagnosisHead, BinaryHead, QualityHead
from models.image_encoder import EfficientNetB0Encoder
from models.metadata_encoder import MetadataEncoder, build_metadata_encoder

VARIANTS = ("image_only", "late_fusion", "channel_gated")


class MCRSLModel(nn.Module):
    def __init__(
        self,
        variant: str,
        metadata_encoder: MetadataEncoder = None,
        num_aux_classes: int = 9,
        quality_aware: bool = False,
        pretrained_backbone: bool = True,
        fusion_hidden_dim: int = 256,
        fusion_dropout: float = 0.3,
    ):
        super().__init__()
        assert variant in VARIANTS, f"unknown variant {variant!r}, expected one of {VARIANTS}"
        self.variant = variant
        self.quality_aware = quality_aware

        self.image_encoder = EfficientNetB0Encoder(pretrained=pretrained_backbone)
        image_dim = self.image_encoder.FEATURE_DIM

        if variant == "image_only":
            self.metadata_encoder = None
            self.fusion = None
            head_in_dim = image_dim
        else:
            if metadata_encoder is None:
                metadata_encoder = build_metadata_encoder()
            self.metadata_encoder = metadata_encoder
            fusion_cls = LateFusion if variant == "late_fusion" else FUSION_REGISTRY["channel_gated"]
            self.fusion = fusion_cls(
                image_dim, metadata_encoder.mlp[-2].out_features, fusion_hidden_dim, fusion_dropout
            )
            head_in_dim = self.fusion.out_dim

        self.binary_head = BinaryHead(head_in_dim)
        self.aux_head = AuxDiagnosisHead(head_in_dim, num_aux_classes)
        self.quality_head = QualityHead(head_in_dim) if quality_aware else None

    def forward(self, image: torch.Tensor, categorical: dict = None, numerical: dict = None, numerical_missing: dict = None):
        pooled, feature_map = self.image_encoder(image)

        if self.variant == "image_only":
            fused = pooled
        else:
            metadata_vec = self.metadata_encoder(categorical, numerical, numerical_missing)
            fused = self.fusion(pooled, metadata_vec, feature_map)

        out = {
            "binary_logits": self.binary_head(fused),
            "aux_logits": self.aux_head(fused),
        }
        if self.quality_head is not None:
            out["quality_pred"] = self.quality_head(fused)
        return out
