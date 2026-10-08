"""Model bundle loading and strictly past-only sequence construction."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
from .model import MaintenanceModel
from .preprocessing import ConditionBaseline

SEQUENCE_LENGTH = 16


def build_sequences(data, baseline, corrected=True, length=SEQUENCE_LENGTH, stride=1):
    features, quality, context, unknown = baseline.transform(data, corrected)
    times = pd.to_datetime(data.timestamp, utc=True, errors='raise').astype('int64').to_numpy() / 1e9
    arrays, targets, indices = [], [], []
    frame = data.reset_index(drop=True)
    for _, group in frame.groupby(["device_id", "run_id"], sort=False):
        positions = group.index.to_numpy()
        for end in range(length-1, len(positions), stride):
            ids = positions[end-length+1:end+1]
            # Never stitch gaps, restarts, or different run segments.
            if np.any(np.diff(frame.loc[ids, "sequence"].to_numpy()) != 1) or np.any(np.abs(np.diff(times[ids])-60)>1):
                continue
            if quality[ids].sum(axis=1).min() == 0 or unknown[ids].any():
                continue
            arrays.append((features[ids], quality[ids], context[ids]))
            row = frame.iloc[ids[-1]]
            targets.append((row.stage, row.hi_label, row.rul_hours))
            indices.append(ids[-1])
    if not arrays:
        raise ValueError("没有连续有效且工况已知的模型序列。")
    f, q, c = (np.stack([a[i] for a in arrays]) for i in range(3))
    targets = np.array(targets)
    return (torch.tensor(f), torch.tensor(q), torch.tensor(c)), (torch.tensor(targets[:,0], dtype=torch.long), torch.tensor(targets[:,1], dtype=torch.float32), torch.tensor(targets[:,2], dtype=torch.float32)), np.array(indices)


def load_bundle(folder="artifacts/model"):
    folder = Path(folder)
    manifest = json.loads((folder / 'manifest.json').read_text())
    import hashlib
    if hashlib.sha256((folder / 'weights.pt').read_bytes()).hexdigest() != manifest.get('weights_sha256'):
        raise ValueError('模型权重校验失败。')
    baseline = ConditionBaseline.from_dict(manifest['baseline'])
    model = MaintenanceModel(kind=manifest['kind'], quality_gate=manifest['quality_gate'])
    model.load_state_dict(torch.load(folder / 'weights.pt', map_location='cpu', weights_only=True))
    model.eval()
    return model, baseline, manifest


def predict_trajectory(data, folder="artifacts/model"):
    model, baseline, manifest = load_bundle(folder)
    result = data.copy().reset_index(drop=True)
    required = {'device_id','run_id','sequence','timestamp','mode'}
    if not required.issubset(result.columns):
        raise ValueError('缺少身份与时间字段。')
    if result.duplicated(['device_id','run_id','sequence']).any():
        raise ValueError('窗口身份重复。')
    sequence = pd.to_numeric(result.sequence, errors='raise')
    if not np.isfinite(sequence).all() or (sequence<0).any() or (sequence % 1 != 0).any():
        raise ValueError('窗口序号必须为非负整数。')
    result['sequence'] = sequence.astype(int)
    parsed = pd.to_datetime(result.timestamp, utc=True, errors='raise')
    if parsed.isna().any():raise ValueError('采样时间不能为空。')
    result['timestamp'] = parsed.map(lambda t: t.isoformat())
    result['health_index'] = np.nan
    result['risk_score'] = np.nan
    result['predicted_stage'] = np.nan
    for name in ['rul_lower_hours', 'rul_median_hours', 'rul_upper_hours']:
        result[name] = np.nan
    _, quality, _, unknown = baseline.transform(result)
    result['validity'] = np.where(unknown, 'unknown_condition', np.where((quality == 0).all(axis=1), 'insufficient_data', 'warming_up'))
    try:
        inputs, _, indices = build_sequences(result.assign(stage=0, hi_label=0, rul_hours=0), baseline, manifest['corrected'])
    except ValueError:
        inputs, indices = None, []
    if inputs is not None:
        with torch.inference_mode():
            outputs = [model(*(a[start:start+256] for a in inputs)) for start in range(0,len(indices),256)]
        logits, hi, rul = [torch.cat([o[i] for o in outputs]).numpy() for i in range(3)]
        result.loc[indices, 'health_index'] = hi
        result.loc[indices, 'risk_score'] = 1 - hi
        result.loc[indices, 'predicted_stage'] = logits.argmax(axis=1)
        result.loc[indices, ['rul_lower_hours', 'rul_median_hours', 'rul_upper_hours']] = rul
        degraded = (quality[indices] < 0.5).any(axis=1)
        result.loc[indices, 'validity'] = np.where(degraded, 'degraded', 'valid')
        # No supported-submodel claim when a critical channel is absent.
        result.loc[indices[degraded], ['rul_lower_hours','rul_median_hours','rul_upper_hours']] = np.nan
    result['model_version'] = manifest['version']
    result['baseline_version'] = baseline.version
    result['rul_unit'] = 'effective_operating_hours'
    result['source'] = result.get('source', 'user_upload')
    return result
