"""Verify the ready-to-use public dataset and CPU model without retraining."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from predictive.bearing_model import load_bearing_bundle, predict_bearing


def checksum(path):
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def verify():
    folder = ROOT / "artifacts" / "bearing"
    path = folder / "data" / "dataset.npz"
    data_manifest = json.loads((path.parent / "dataset_manifest.json").read_text(encoding="utf-8"))
    digest = checksum(path)
    if digest != data_manifest["dataset_sha256"]:
        raise ValueError("公开预处理数据 SHA-256 与交付清单不符。")
    with np.load(path, allow_pickle=False) as data:
        X, y, split, records = [data[name] for name in ("X", "y", "split", "record_id")]
        if X.shape != (len(y), 1, 1024) or not np.isfinite(X).all():
            raise ValueError("振动窗口形状或数值无效。")
        if not np.isin(y, range(4)).all() or not np.isin(split, ("train", "validation", "test")).all():
            raise ValueError("类别或数据分区无效。")
        counts = {}
        for part in ("train", "validation", "test"):
            mask = split == part
            counts[part] = {"windows": int(mask.sum()), "records": len(np.unique(records[mask]))}
            if set(y[mask]) != set(range(4)):
                raise ValueError(f"{part} 未覆盖四种故障部位类别。")
        for record in np.unique(records):
            if len(np.unique(split[records == record])) != 1 or len(np.unique(y[records == record])) != 1:
                raise ValueError("同一记录跨越分区或对应多个标签。")
        if len(y) != data_manifest["total_windows"]:
            raise ValueError("样本数量与清单不符。")
        for part, count in counts.items():
            if count["records"] != data_manifest["record_split"][part] or count["windows"] != data_manifest["window_split"][part]:
                raise ValueError("分区数量与清单不符。")
        torch.set_num_threads(2)
        model, manifest = load_bearing_bundle(folder)
        if manifest["data_protocol"]["dataset_sha256"] != digest:
            raise ValueError("模型的训练数据与当前公开数据不一致。")
        example = X[np.flatnonzero(split == "test")[0]]
        result = predict_bearing(example, model, manifest, prepared=True)[0]
        if not np.isfinite(list(result["scores"].values())).all():
            raise ValueError("模型产生了无效分类分数。")
    print("公开数据和模型校验通过；不需要首次下载或重新训练。")
    print(json.dumps({"source": "CWRU真实振动", "partitions": counts,
                      "selected_model": manifest["selected_model"],
                      "sample_prediction": result["class_name_zh"]}, ensure_ascii=False))
    nasa = ROOT / "data" / "public" / "NASA_IMS" / "prepared"
    if (nasa / "dataset_manifest.json").exists():
        nasa_manifest = json.loads((nasa / "dataset_manifest.json").read_text(encoding="utf-8"))
        if checksum(nasa / "features.csv") != nasa_manifest["features_sha256"]:
            raise ValueError("NASA 预处理特征校验失败。")
        print("NASA IMS 退化特征已附带；其代理标签不代表已验证寿命预测。")
    return counts


if __name__ == "__main__":
    try:
        verify()
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        print("交付校验失败：" + str(error), file=sys.stderr)
        sys.exit(1)
