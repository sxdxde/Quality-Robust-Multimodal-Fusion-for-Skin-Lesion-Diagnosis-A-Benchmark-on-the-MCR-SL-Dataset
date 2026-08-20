"""Classification/regression heads and losses.

- BinaryHead: main malignant vs. non-malignant head. Weighted BCE by default;
  focal loss available via `focal_gamma`.
- AuxDiagnosisHead: 9-class unified-diagnosis head (exploratory table only),
  class-weighted CE, down-weighted in the combined loss (see config.py).
- QualityHead: small regression head predicting mean image_rating from the
  pooled fusion features, for robustness analysis #2 (quality-aware
  training). Low loss weight; only attached when quality-aware training is
  enabled.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class BinaryHead(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x).squeeze(-1)  # (B,) logits

    @staticmethod
    def loss(logits: torch.Tensor, targets: torch.Tensor, pos_weight: torch.Tensor = None, focal_gamma: float = 0.0) -> torch.Tensor:
        if focal_gamma > 0:
            bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
            p_t = torch.exp(-bce)
            focal_weight = (1 - p_t) ** focal_gamma
            if pos_weight is not None:
                alpha = torch.where(targets == 1, pos_weight, torch.ones_like(targets))
                focal_weight = focal_weight * alpha
            return (focal_weight * bce).mean()
        return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight)


class AuxDiagnosisHead(nn.Module):
    def __init__(self, in_dim: int, num_classes: int = 9):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)  # (B, num_classes) logits

    @staticmethod
    def loss(logits: torch.Tensor, targets: torch.Tensor, class_weights: torch.Tensor = None) -> torch.Tensor:
        return F.cross_entropy(logits, targets, weight=class_weights)


class QualityHead(nn.Module):
    """Predicts mean per-lesion image_rating (1-10 scale, passed in normalized)."""

    def __init__(self, in_dim: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x).squeeze(-1)

    @staticmethod
    def loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(pred, target)


def supervised_contrastive_loss(embeddings: torch.Tensor, labels: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    """SupCon (Khosla et al. 2020) on the fused pre-head embedding, using the
    binary malignant/non-malignant label. Pulls same-class lesions together,
    pushes different-class apart — an auxiliary signal on top of the
    classification loss, intended to help given how few malignant examples
    each batch has.

    Returns 0 if the batch doesn't contain at least one positive pair (i.e.
    at least 2 examples of some class) — SupCon is undefined otherwise, and
    with malignant:non-malignant ~1:4.6 some small batches will have 0-1
    malignant examples.
    """
    device = embeddings.device
    labels = labels.view(-1, 1)
    batch_size = labels.shape[0]

    same_class = torch.eq(labels, labels.T).float().to(device)
    self_mask = torch.eye(batch_size, device=device)
    positive_mask = same_class - self_mask  # exclude self-comparisons
    if positive_mask.sum() == 0:
        return torch.tensor(0.0, device=device)

    z = F.normalize(embeddings, dim=-1)
    sim = torch.matmul(z, z.T) / temperature
    sim = sim - sim.max(dim=1, keepdim=True).values.detach()  # numerical stability

    exp_sim = torch.exp(sim) * (1 - self_mask)  # exclude self from the denominator
    log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-12)

    mean_log_prob_pos = (positive_mask * log_prob).sum(dim=1) / positive_mask.sum(dim=1).clamp(min=1)
    valid = positive_mask.sum(dim=1) > 0
    if valid.sum() == 0:
        return torch.tensor(0.0, device=device)
    return -mean_log_prob_pos[valid].mean()
