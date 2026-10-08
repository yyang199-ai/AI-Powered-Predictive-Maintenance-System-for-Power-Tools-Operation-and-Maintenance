"""Training-only healthy baselines and quality-constrained sensor encoding."""
import numpy as np
from .data import FEATURES, QUALITY, CONTEXT


class ConditionBaseline:
    version = "healthy-linear-v1"

    @staticmethod
    def design(data):
        return np.column_stack([np.ones(len(data)), data.load, data.rpm / 10000, data.ambient_c / 30])

    def fit(self, training):
        # Known healthy labels are used ONLY for training baseline admission.
        healthy = training[(training.simulated_wear < 0.15) & (training[QUALITY].min(axis=1) > 0.9)]
        if len(healthy) < 20:
            raise ValueError("健康高质量训练样本不足。")
        x = self.design(healthy)
        y = healthy[FEATURES].to_numpy(float)
        self.coefficients = np.linalg.lstsq(x, y, rcond=None)[0]
        self.scale = np.maximum(np.std(y - x @ self.coefficients, axis=0), [0.1, 0.15, 0.15, 0.04, 0.5, 0.006])
        self.context_min = training[CONTEXT].min().to_numpy(float)
        self.context_max = training[CONTEXT].max().to_numpy(float)
        return self

    def transform(self, data, corrected=True):
        y = data[FEATURES].to_numpy(float)
        q = data[QUALITY].to_numpy(float).copy()
        if not np.isfinite(q).all() or np.any((q < 0) | (q > 1)):
            raise ValueError("质量分数必须在 0 到 1 之间。")
        for sensor in range(3):
            q[~np.isfinite(y[:, sensor*2:sensor*2+2]).all(axis=1), sensor] = 0
        context = data[CONTEXT].to_numpy(float)
        if not np.isfinite(context).all():
            raise ValueError("转速、独立负载和环境温度必须是有限数字。")
        expected = self.design(data) @ self.coefficients
        residual = (y - expected) / self.scale if corrected else y / self.scale
        residual = np.clip(np.nan_to_num(residual), -30, 30)
        # Invalid channels carry a mask; zero is never interpreted as measured zero.
        residual *= np.repeat(q > 0, 2, axis=1)
        context_scaled = (context - self.context_min) / np.maximum(self.context_max - self.context_min, 1e-6)
        unknown = ((context_scaled < -0.1) | (context_scaled > 1.1)).any(axis=1)
        unknown |= ~data['mode'].eq('drilling').to_numpy()
        return residual.astype('float32'), q.astype('float32'), context_scaled.astype('float32'), unknown

    def to_dict(self):
        return {"version": self.version, "features": FEATURES, **{k: getattr(self, k).tolist() for k in
                ["coefficients", "scale", "context_min", "context_max"]}}

    @classmethod
    def from_dict(cls, state):
        if state['features'] != FEATURES or state['version'] != cls.version:
            raise ValueError("基线版本或特征顺序不兼容。")
        obj = cls()
        for k in ["coefficients", "scale", "context_min", "context_max"]:
            setattr(obj, k, np.array(state[k]))
        return obj
