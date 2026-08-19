"""Two fusion variants, per the ablation matrix.

1. LateFusion       — concat(pooled image vector, metadata vector) -> MLP. The
   standard MetaBlock-style baseline; a fair comparison point, not the
   contribution.
2. ChannelGatedFusion — metadata vector -> linear -> sigmoid gate (1280-d),
   elementwise-multiplies the image feature map's channels (broadcast over
   spatial dims) before global pooling. Same idea as an SE-block, but
   conditioned on metadata instead of the block's own pooled features. Main
   method for this paper.
"""
import torch
import torch.nn as nn


class LateFusion(nn.Module):
    def __init__(self, image_dim: int = 1280, metadata_dim: int = 128, hidden_dim: int = 256, dropout: float = 0.3):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(image_dim + metadata_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.out_dim = hidden_dim

    def forward(self, pooled_image: torch.Tensor, metadata_vec: torch.Tensor, feature_map: torch.Tensor = None):
        return self.mlp(torch.cat([pooled_image, metadata_vec], dim=-1))


class ChannelGatedFusion(nn.Module):
    """Metadata-conditioned channel gating (SE-style), applied to the image
    encoder's last conv feature map before pooling.
    """

    def __init__(self, image_channels: int = 1280, metadata_dim: int = 128, hidden_dim: int = 256, dropout: float = 0.3):
        super().__init__()
        self.gate = nn.Linear(metadata_dim, image_channels)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(image_channels, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.out_dim = hidden_dim

    def forward(self, pooled_image: torch.Tensor, metadata_vec: torch.Tensor, feature_map: torch.Tensor):
        gate = torch.sigmoid(self.gate(metadata_vec))  # (B, C)
        gated_map = feature_map * gate.unsqueeze(-1).unsqueeze(-1)  # broadcast over H, W
        gated_pooled = self.pool(gated_map).flatten(1)  # (B, C)
        return self.mlp(gated_pooled)


FUSION_REGISTRY = {
    "late": LateFusion,
    "channel_gated": ChannelGatedFusion,
}
