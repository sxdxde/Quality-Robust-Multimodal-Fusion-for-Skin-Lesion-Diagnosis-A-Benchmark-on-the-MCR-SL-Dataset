"""Tabular metadata encoder.

Config-driven so the exact field list can be finalized once the real MCR-SL
CSVs are verified (see data/schema.py) without touching this module:
- Categorical fields -> nn.Embedding per field, cardinality+1 slots (last
  index reserved for an explicit "unknown"/missing category). Missing values
  are NOT imputed — they're routed to the unknown embedding, matching the
  dataset authors' policy of leaving missingness to end users.
- Numerical fields -> expected already z-score normalized by the caller using
  train-fold statistics (see data/dataset.py); missing numerics are passed in
  as 0.0 alongside a parallel missing-mask bit so the model can tell "0" from
  "missing".

Output: 128-d metadata vector via a 2-layer MLP over
concat(embeddings, numerics, numeric_missing_mask).
"""
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class CategoricalFieldSpec:
    name: str
    cardinality: int  # number of known categories (unknown gets index `cardinality`)
    embed_dim: int = 12


@dataclass
class NumericalFieldSpec:
    name: str


class MetadataEncoder(nn.Module):
    def __init__(
        self,
        categorical_fields: list[CategoricalFieldSpec],
        numerical_fields: list[NumericalFieldSpec],
        out_dim: int = 128,
        hidden_dim: int = 128,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.categorical_fields = categorical_fields
        self.numerical_fields = numerical_fields

        self.embeddings = nn.ModuleDict(
            {
                f.name: nn.Embedding(f.cardinality + 1, f.embed_dim)
                for f in categorical_fields
            }
        )

        embed_total = sum(f.embed_dim for f in categorical_fields)
        numeric_total = len(numerical_fields) * 2  # value + missing-mask bit

        in_dim = embed_total + numeric_total
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, categorical: dict[str, torch.Tensor], numerical: dict[str, torch.Tensor], numerical_missing: dict[str, torch.Tensor]):
        """
        categorical[name]: (B,) long tensor, values in [0, cardinality] (cardinality == unknown)
        numerical[name]: (B,) float tensor, pre-normalized, 0.0 where missing
        numerical_missing[name]: (B,) float tensor, 1.0 if missing else 0.0
        """
        parts = []
        for f in self.categorical_fields:
            parts.append(self.embeddings[f.name](categorical[f.name]))
        for f in self.numerical_fields:
            parts.append(numerical[f.name].unsqueeze(-1))
            parts.append(numerical_missing[f.name].unsqueeze(-1))
        x = torch.cat(parts, dim=-1)
        return self.mlp(x)
