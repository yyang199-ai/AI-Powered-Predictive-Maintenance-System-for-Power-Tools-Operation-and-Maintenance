"""Small quality-gated Transformer/LSTM multi-task models (CPU)."""
import math
import torch
from torch import nn


class MaintenanceModel(nn.Module):
    def __init__(self, kind="hybrid", dimension=32, quality_gate=True):
        super().__init__()
        self.kind, self.quality_gate = kind, quality_gate
        self.encoders = nn.ModuleList([nn.Sequential(nn.Linear(2, dimension), nn.GELU()) for _ in range(3)])
        self.gate = nn.Linear(6, 3)
        self.context = nn.Linear(6, dimension)
        if kind in ("hybrid", "transformer"):
            layer = nn.TransformerEncoderLayer(dimension, 2, dimension * 2, dropout=0.0, batch_first=True)
            self.transformer = nn.TransformerEncoder(layer, 1, enable_nested_tensor=False)
        if kind in ("hybrid", "lstm"):
            self.lstm = nn.LSTM(dimension, dimension, batch_first=True)
        self.classifier = nn.Linear(dimension, 4)
        self.health = nn.Linear(dimension, 1)
        self.rul = nn.Linear(dimension, 3)

    def forward(self, features, quality, context):
        embeddings = torch.stack([encoder(features[..., i*2:i*2+2]) for i, encoder in enumerate(self.encoders)], dim=-2)
        logits = self.gate(torch.cat([quality, context], dim=-1))
        available = (quality > 0).float()
        weights = torch.exp(logits - logits.max(dim=-1, keepdim=True).values)
        weights = weights * quality if self.quality_gate else available
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        x = (embeddings * weights.unsqueeze(-1)).sum(dim=-2)
        x = x + self.context(torch.cat([context, quality], dim=-1))
        steps = torch.arange(x.shape[1], device=x.device).float().unsqueeze(1)
        frequencies = torch.exp(torch.arange(0, x.shape[-1], 2, device=x.device) * (-math.log(10000) / x.shape[-1]))
        position = torch.zeros(x.shape[1], x.shape[-1], device=x.device)
        position[:, 0::2], position[:, 1::2] = torch.sin(steps * frequencies), torch.cos(steps * frequencies)
        x = x + position
        if self.kind in ("hybrid", "transformer"):
            # Every sequence ends at the forecast time. Never predicts earlier steps using later ones.
            invalid = quality.sum(dim=-1) == 0
            # All-invalid sequences are rejected by the sequence builder/inference gate.
            x = self.transformer(x, src_key_padding_mask=invalid)
        if self.kind in ("hybrid", "lstm"):
            x, _ = self.lstm(x)  # state resets for each complete history sequence
        state = x[:, -1]
        return self.classifier(state), torch.sigmoid(self.health(state)).squeeze(-1), torch.sort(torch.nn.functional.softplus(self.rul(state)), dim=-1).values


def multitask_loss(outputs, stage, hi, rul):
    logits, health, quantiles = outputs
    error = rul[:, None] - quantiles
    levels = torch.tensor([0.1, 0.5, 0.9], device=error.device)
    pinball = torch.maximum(levels * error, (levels - 1) * error).mean()
    return nn.functional.cross_entropy(logits, stage) + nn.functional.mse_loss(health, hi) + pinball
