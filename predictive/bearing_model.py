"""Small classifiers for real, single-channel bearing vibration windows.

These models classify CWRU fault locations. They have no health-index or RUL
head: seeded bearing defects do not provide a run-to-failure lifetime label.
"""
from pathlib import Path
import hashlib
import json
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

CLASS_LABELS = ["normal", "inner_race", "ball", "outer_race"]
CLASS_NAMES_ZH = ["正常", "内圈故障", "滚动体故障", "外圈故障"]
MODEL_KINDS = ("cnn", "softmax", "linear")
WINDOW_LENGTH = 1024


class KernelLinearAttention(nn.Module):
    """Positive-kernel attention, associating K^T V before multiplying Q.

    phi(x) = ELU(x) + 1. This is a different attention kernel from softmax,
    rather than an exact or universally more accurate softmax implementation.
    The attention core uses O(T d_head^2), without a T-by-T attention matrix.
    """

    def __init__(self, dimension=24, heads=4):
        super().__init__()
        if dimension <= 0 or heads <= 0 or dimension % heads:
            raise ValueError("dimension must be positive and divisible by heads")
        self.dimension, self.heads = dimension, heads
        self.qkv = nn.Linear(dimension, dimension * 3)
        self.project = nn.Linear(dimension, dimension)

    def forward(self, tokens):
        batch, steps, dimension = tokens.shape
        q, k, v = self.qkv(tokens).reshape(batch, steps, 3, self.heads,
                                          dimension // self.heads).permute(2, 0, 3, 1, 4)
        q, k = F.elu(q) + 1, F.elu(k) + 1
        summary = k.transpose(-2, -1) @ v
        denominator = (q * k.sum(dim=-2, keepdim=True)).sum(dim=-1, keepdim=True)
        value = (q @ summary) / denominator.clamp_min(1e-6)
        return self.project(value.transpose(1, 2).reshape(batch, steps, dimension))


class SoftmaxAttention(nn.Module):
    """Same qkv/project widths as KernelLinearAttention; standard softmax."""

    def __init__(self, dimension=24, heads=4):
        super().__init__()
        if dimension <= 0 or heads <= 0 or dimension % heads:
            raise ValueError("dimension must be positive and divisible by heads")
        self.dimension, self.heads = dimension, heads
        self.qkv = nn.Linear(dimension, dimension * 3)
        self.project = nn.Linear(dimension, dimension)

    def forward(self, tokens):
        batch, steps, dimension = tokens.shape
        q, k, v = self.qkv(tokens).reshape(batch, steps, 3, self.heads,
                                          dimension // self.heads).permute(2, 0, 3, 1, 4)
        weights = torch.softmax((q @ k.transpose(-2, -1)) /
                                math.sqrt(dimension // self.heads), dim=-1)
        value = weights @ v
        return self.project(value.transpose(1, 2).reshape(batch, steps, dimension))


class BearingClassifier(nn.Module):
    """Shared depthwise CNN followed by no attention, softmax, or linear attention."""

    def __init__(self, kind="linear", dimension=24, heads=4, window_length=1024):
        super().__init__()
        if kind not in MODEL_KINDS:
            raise ValueError(f"unknown bearing classifier: {kind}")
        if dimension <= 0 or heads <= 0 or dimension % 2 or dimension % heads or window_length != WINDOW_LENGTH:
            raise ValueError("use a positive even head-divisible dimension and 1024-point windows")
        self.kind, self.dimension, self.heads = kind, dimension, heads
        self.window_length = window_length
        self.stem = nn.Sequential(
            nn.Conv1d(1, 8, 7, stride=4, padding=3), nn.BatchNorm1d(8), nn.GELU(),
            nn.Conv1d(8, 8, 5, stride=2, padding=2, groups=8),
            nn.BatchNorm1d(8), nn.GELU(),
            nn.Conv1d(8, 16, 1), nn.BatchNorm1d(16), nn.GELU(),
            nn.Conv1d(16, 16, 5, stride=2, padding=2, groups=16),
            nn.BatchNorm1d(16), nn.GELU(),
            nn.Conv1d(16, dimension, 1), nn.BatchNorm1d(dimension), nn.GELU(),
        )
        if kind != "cnn":
            self.norm_attention = nn.LayerNorm(dimension)
            attention = KernelLinearAttention if kind == "linear" else SoftmaxAttention
            self.attention = attention(dimension, heads)
            self.norm_feedforward = nn.LayerNorm(dimension)
            self.feedforward = nn.Sequential(nn.Linear(dimension, dimension * 2),
                                             nn.GELU(), nn.Linear(dimension * 2, dimension))
        self.classifier = nn.Linear(dimension, len(CLASS_LABELS))

    def forward(self, waveform):
        if waveform.ndim != 3 or waveform.shape[1:] != (1, self.window_length):
            raise ValueError("bearing input must have shape [batch, 1, 1024]")
        tokens = self.stem(waveform).transpose(1, 2)
        if self.kind != "cnn":
            # A fixed position signal; no additional learned position parameters.
            steps = torch.arange(tokens.shape[1], device=tokens.device,
                                 dtype=tokens.dtype).unsqueeze(1)
            frequencies = torch.exp(torch.arange(0, self.dimension, 2,
                                                device=tokens.device, dtype=tokens.dtype)
                                    * (-math.log(10000) / self.dimension))
            position = torch.zeros(tokens.shape[1], self.dimension,
                                   device=tokens.device, dtype=tokens.dtype)
            position[:, 0::2] = torch.sin(steps * frequencies)
            position[:, 1::2] = torch.cos(steps * frequencies)
            tokens = tokens + position
            tokens = tokens + self.attention(self.norm_attention(tokens))
            tokens = tokens + self.feedforward(self.norm_feedforward(tokens))
        return self.classifier(tokens.mean(dim=1))


def computation_profile(model):
    """Analytical batch=1 multiply-accumulate estimates, not measured FLOPs.

    Excludes activation, normalization, softmax, positive-feature map, reduction,
    position construction, memory movement and preprocessing. A MAC is one
    multiplication plus accumulation; multiplying by 2 is only an approximate
    FLOP convention. All three modes use the same convolutional front end.
    """
    length = model.window_length
    convolution_macs = 0
    for layer in model.stem:
        if isinstance(layer, nn.Conv1d):
            length = (length + 2 * layer.padding[0] - layer.dilation[0] *
                      (layer.kernel_size[0] - 1) - 1) // layer.stride[0] + 1
            convolution_macs += (length * layer.out_channels *
                                 (layer.in_channels // layer.groups) * layer.kernel_size[0])
    dimension, heads = model.dimension, model.heads
    projections = feedforward = attention_core = 0
    if model.kind != "cnn":
        projections = 4 * length * dimension * dimension
        feedforward = 4 * length * dimension * dimension
        attention_core = (2 * length * dimension * dimension // heads if
                          model.kind == "linear" else 2 * length * length * dimension)
    classifier = dimension * len(CLASS_LABELS)
    return {
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "window_length": model.window_length,
        "token_count": length,
        "dimension": dimension,
        "heads": heads,
        "estimated_macs": convolution_macs + projections + feedforward + attention_core + classifier,
        "mac_breakdown": {"convolutions": convolution_macs, "attention_projections": projections,
                          "attention_core": attention_core, "feedforward": feedforward,
                          "classifier": classifier},
        "attention_pair_workspace_elements": (heads * length * length if model.kind == "softmax"
                                               else heads * (dimension // heads) ** 2
                                               if model.kind == "linear" else 0),
        "scope": "analytical batch=1 MAC estimate; excludes elementwise/reduction/normalization/IO",
        "attention_complexity": {"cnn": "no attention",
                                 "softmax": "O(T^2 * D)",
                                 "linear": "O(T * D^2 / heads)"}[model.kind],
    }


def normalize_bearing(waveform, normalization):
    """Apply training-only scalar statistics; one output for each raw 1024-point window."""
    values = np.asarray(waveform, dtype=np.float32)
    if values.ndim == 1:
        values = values[None, None, :]
    elif values.ndim == 2:
        values = values[:, None, :]
    if values.ndim != 3 or values.shape[1:] != (1, WINDOW_LENGTH) or not np.isfinite(values).all():
        raise ValueError("需要有限的单通道1024点振动窗口。")
    mean, std = float(normalization["mean"]), float(normalization["std"])
    if not math.isfinite(mean) or not math.isfinite(std) or std <= 0:
        raise ValueError("训练标准化统计量无效。")
    return torch.from_numpy(np.ascontiguousarray((values - mean) / std, dtype=np.float32))


def load_bearing_bundle(folder="artifacts/bearing"):
    folder = Path(folder)
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    path = folder / "model.pt"
    if hashlib.sha256(path.read_bytes()).hexdigest() != manifest.get("weights_sha256"):
        raise ValueError("轴承分类模型校验失败。")
    if manifest.get("class_labels") != CLASS_LABELS or manifest.get("task") != "bearing_fault_classification":
        raise ValueError("模型类别或任务与轴承分类接口不兼容。")
    model = BearingClassifier(**manifest["model_config"])
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    model.eval()
    return model, manifest


def predict_bearing(waveform, model, manifest, prepared=False):
    """Predict 12 kHz windows, applying the same raw preprocessing as training.

    Set prepared=True only for windows already processed by public_data's
    preprocess_waveform (e.g. the generated dataset.npz). A whole recording at
    another rate must first be resampled and then cut into 1024-point windows.
    """
    values = np.asarray(waveform, dtype=np.float32)
    # Validate shape and finiteness before passing data to the signal processor.
    normalize_bearing(values, {"mean": 0, "std": 1})
    if not prepared:
        from .public_data import preprocess_waveform
        rows = values.reshape(-1, WINDOW_LENGTH)
        values = np.stack([preprocess_waveform(row, sample_rate=12000, target_rate=12000)
                           for row in rows])[:, None, :].astype(np.float32)
    inputs = normalize_bearing(values, manifest["normalization"])
    with torch.inference_mode():
        scores = torch.softmax(model(inputs), dim=-1).cpu().numpy()
    return [{"class_index": int(row.argmax()), "class_label": CLASS_LABELS[int(row.argmax())],
             "class_name_zh": CLASS_NAMES_ZH[int(row.argmax())],
             "scores": {label: float(row[index]) for index, label in enumerate(CLASS_LABELS)},
             "score_scope": "uncalibrated classifier softmax scores", "source": "CWRU pretrained classifier"}
            for row in scores]
