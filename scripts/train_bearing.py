"""Train/evaluate three small CWRU classifiers with record-grouped partitions.

Run from the repository root:
  .venv/bin/python scripts/train_bearing.py --epochs 15

The public dataset contains seeded fault labels, not remaining-life labels.
"""
import argparse
import copy
import hashlib
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from predictive.bearing_model import (BearingClassifier, CLASS_LABELS, CLASS_NAMES_ZH,
                                     MODEL_KINDS, WINDOW_LENGTH, computation_profile,
                                     normalize_bearing, predict_bearing)


def _checksum(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_dataset(path, protocol="record_split", seed=42):
    """Validate provided partitions without moving overlapping windows across records."""
    path = Path(path)
    if path.is_dir():
        arrays = {name: np.load(path / f"{name}.npy", allow_pickle=False)
                  for name in ("X", "y", "split", "record_id")}
        for name in ("load_hp", "raw_example"):
            if (path / f"{name}.npy").exists():
                arrays[name] = np.load(path / f"{name}.npy", allow_pickle=False)
        parent = path
        checksum = {name: _checksum(path / f"{name}.npy") for name in ("X", "y", "split", "record_id")}
    else:
        with np.load(path, allow_pickle=False) as archive:
            arrays = {name: archive[name] for name in archive.files}
        parent, checksum = path.parent, _checksum(path)
    manifest_path = next((parent / name for name in ("dataset_manifest.json", "artifact_manifest.json")
                          if (parent / name).exists()), None)
    if manifest_path is None:
        raise ValueError("dataset_manifest.json is required to record source and preprocessing")
    source_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not {"X", "y", "split", "record_id"}.issubset(arrays):
        raise ValueError("dataset must contain X, y, split and record_id")
    X = np.asarray(arrays["X"], dtype=np.float32)
    y = np.asarray(arrays["y"])
    partitions = np.asarray(arrays["split"]).astype(str)
    records = np.asarray(arrays["record_id"]).astype(str)
    if X.ndim != 3 or X.shape[1:] != (1, WINDOW_LENGTH) or not np.isfinite(X).all():
        raise ValueError("X must be finite [N,1,1024] prepared vibration windows")
    if any(a.shape != (len(X),) for a in (y, partitions, records)):
        raise ValueError("labels, partitions and record IDs must align with X")
    if y.dtype.kind not in "iu" or not np.isin(y, range(4)).all():
        raise ValueError("integer labels must be normal=0, inner=1, ball=2, outer=3")
    y = y.astype(np.int64)
    if protocol == "load3_holdout":
        if "load_hp" not in arrays:
            raise ValueError("load3_holdout requires load_hp")
        loads = np.asarray(arrays["load_hp"])
        if loads.shape != (len(X),):
            raise ValueError("load_hp must align with X")
        partitions = np.full(len(X), "train", dtype="<U10")
        partitions[loads == 3] = "test"
        generator = np.random.default_rng(seed)
        for label in range(4):
            candidates = np.unique(records[(y == label) & (loads != 3)])
            generator.shuffle(candidates)
            if len(candidates) < 2:
                raise ValueError("need at least two non-held-out records per class")
            validation = candidates[:max(1, round(len(candidates) * .2))]
            partitions[np.isin(records, validation)] = "validation"
    elif protocol != "record_split":
        raise ValueError("unknown data protocol")
    if not np.isin(partitions, ("train", "validation", "test")).all() or (records == "").any():
        raise ValueError("invalid partition name or empty record identity")
    record_sets = {name: set(records[partitions == name]) for name in ("train", "validation", "test")}
    for first, second in (("train", "validation"), ("train", "test"), ("validation", "test")):
        if record_sets[first] & record_sets[second]:
            raise ValueError("record leakage: a recording appears in multiple partitions")
    counts = {}
    for name in ("train", "validation", "test"):
        mask = partitions == name
        class_counts = np.bincount(y[mask], minlength=4)
        if (class_counts == 0).any():
            raise ValueError(f"{name} must contain all four classes")
        counts[name] = {"windows": int(mask.sum()), "records": len(record_sets[name]),
                        "record_ids": sorted(record_sets[name]), "class_counts": class_counts.tolist(),
                        "class_record_counts": [len(set(records[mask & (y == label)])) for label in range(4)]}
    return X, y, partitions, {
        "name": protocol, "source": "CWRU public bearing vibration recordings",
        "dataset_path": str(path), "dataset_sha256": checksum,
        "source_manifest": source_manifest, "source_manifest_path": str(manifest_path), "counts": counts,
        "record_groups_disjoint": True,
        "sample_rate_hz": 12000, "window_length": 1024,
        "preprocessing": "record-level anti-alias resampling to 12kHz; window demean + level-3 Haar soft threshold",
        "independence_limit": "record grouping prevents overlap leakage, but physical bearings may recur across load recordings",
        "load_holdout_limit": "holding out load=3 tests a new load recording, not necessarily a new physical bearing",
    }, arrays.get("raw_example")


def fit_normalization(X, partitions):
    train = X[partitions == "train"]
    mean = float(train.mean(dtype=np.float64))
    std = float(train.std(dtype=np.float64))
    if not np.isfinite(mean) or not np.isfinite(std) or std < 1e-8:
        raise ValueError("training vibration variance is insufficient")
    return {"mean": mean, "std": std, "fit_partition": "train only",
            "scope": "one global scalar mean/std, not per-window unit amplitude normalization"}


def evaluate(model, inputs, labels, batch_size=128, record_ids=None):
    model.eval()
    scores, losses = [], []
    with torch.inference_mode():
        for start in range(0, len(inputs), batch_size):
            logits = model(inputs[start:start + batch_size])
            losses.append(float(F.cross_entropy(logits, labels[start:start + batch_size], reduction="sum")))
            scores.append(torch.softmax(logits, dim=-1))
    scores = torch.cat(scores).numpy()
    truth = labels.numpy()
    predicted = scores.argmax(axis=1)
    precision, recall, f1, support = precision_recall_fscore_support(truth, predicted, labels=list(range(4)),
                                                                  zero_division=0)
    result = {
        "samples": len(truth), "accuracy": float(accuracy_score(truth, predicted)),
        "macro_f1": float(f1.mean()), "loss": sum(losses) / len(truth),
        "confusion_matrix": confusion_matrix(truth, predicted, labels=list(range(4))).tolist(),
        "per_class": [{"class_index": index, "class_label": CLASS_LABELS[index],
                       "precision": float(precision[index]), "recall": float(recall[index]),
                       "f1": float(f1[index]), "support": int(support[index])} for index in range(4)],
        "healthy_window_false_positive_rate": float((predicted[truth == 0] != 0).mean()),
        "fault_window_recall": float((predicted[truth != 0] != 0).mean()),
        "scores_calibrated": False,
        "metric_scope": "record-held-out window metrics; overlapping windows are correlated, not independent tests",
    }
    if record_ids is not None:
        record_ids = np.asarray(record_ids).astype(str)
        if record_ids.shape != truth.shape:
            raise ValueError("record IDs must align with evaluated labels")
        record_results = []
        for record in sorted(np.unique(record_ids)):
            mask = record_ids == record
            targets = np.unique(truth[mask])
            if len(targets) != 1:
                raise ValueError("each original recording must have one fault-location label")
            target = int(targets[0])
            majority = int(np.bincount(predicted[mask], minlength=4).argmax())
            record_results.append({"record_id": record, "class_index": target, "windows": int(mask.sum()),
                                   "window_accuracy": float((predicted[mask] == target).mean()),
                                   "majority_prediction": majority, "majority_correct": majority == target})
        result.update({"record_count": len(record_results), "per_record": record_results,
                       "mean_record_window_accuracy": float(np.mean([row["window_accuracy"] for row in record_results])),
                       "record_majority_vote_accuracy": float(np.mean([row["majority_correct"] for row in record_results]))})
    return result, predicted


def train_one(kind, inputs, labels, partitions, args):
    torch.manual_seed(args.seed)
    model = BearingClassifier(kind, args.dimension, args.heads)
    masks = {name: partitions == name for name in ("train", "validation", "test")}
    train_x, train_y = inputs[masks["train"]], labels[masks["train"]]
    validation_x, validation_y = inputs[masks["validation"]], labels[masks["validation"]]
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(TensorDataset(train_x, train_y), batch_size=args.batch_size,
                        shuffle=True, generator=generator, num_workers=0)
    counts = torch.bincount(train_y, minlength=4).float()
    weights = counts.sum() / (4 * counts)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    best_state, best_loss, best_epoch, history = None, float("inf"), 0, []
    start = time.perf_counter()
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for waveforms, targets in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(waveforms), targets, weight=weights)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.detach()) * len(targets)
        validation, _ = evaluate(model, validation_x, validation_y)
        history.append({"epoch": epoch + 1, "train_weighted_loss": total_loss / len(train_y),
                        "validation_loss": validation["loss"],
                        "validation_accuracy": validation["accuracy"], "validation_macro_f1": validation["macro_f1"]})
        if validation["loss"] < best_loss:
            best_loss, best_epoch = validation["loss"], epoch + 1
            best_state = copy.deepcopy(model.state_dict())
        print(f"{kind} epoch={epoch + 1}/{args.epochs} val_acc={validation['accuracy']:.4f} val_f1={validation['macro_f1']:.4f}", flush=True)
    model.load_state_dict(best_state)
    model.eval()
    return model, {"history": history, "best_epoch": best_epoch,
                   "checkpoint_selection": "lowest unweighted validation cross-entropy; test not used",
                   "training_seconds": time.perf_counter() - start,
                   "training_class_weights": weights.tolist()}


def benchmark(model, prepared_sample, raw_example, normalization, repeats=100):
    """Measure processing latency; excludes acquisition, reading files and networking."""
    prepared_input = normalize_bearing(prepared_sample, normalization)
    mini_manifest = {"normalization": normalization}
    forward, pipeline = [], []
    with torch.inference_mode():
        for _ in range(15):
            model(prepared_input)
        for _ in range(repeats):
            start = time.perf_counter_ns()
            model(prepared_input)
            forward.append((time.perf_counter_ns() - start) / 1e6)
    result = {"repetitions": repeats, "batch_size": 1, "torch_threads": torch.get_num_threads(),
              "forward_median_ms": float(np.median(forward)), "forward_p95_ms": float(np.percentile(forward, 95)),
              "forward_scope": "normalized 1024-point window -> logits; excludes normalization/IO/acquisition"}
    if raw_example is not None:
        sample = np.asarray(raw_example, dtype=np.float32).reshape(-1)
        if sample.shape != (1024,) or not np.isfinite(sample).all():
            raise ValueError("raw_example must be one finite, unprocessed 12kHz 1024-point window")
        for _ in range(15):
            predict_bearing(sample, model, mini_manifest, prepared=False)
        for _ in range(repeats):
            start = time.perf_counter_ns()
            predict_bearing(sample, model, mini_manifest, prepared=False)
            pipeline.append((time.perf_counter_ns() - start) / 1e6)
        result.update({"pipeline_median_ms": float(np.median(pipeline)),
                       "pipeline_p95_ms": float(np.percentile(pipeline, 95)),
                       "pipeline_scope": "raw 12kHz window -> demean/Haar -> training normalization -> logits/softmax/label; excludes acquisition/IO/network"})
    else:
        # Still support inference benchmarking, but do not pretend X contains raw signals.
        for _ in range(15):
            predict_bearing(prepared_sample, model, mini_manifest, prepared=True)
        for _ in range(repeats):
            start = time.perf_counter_ns()
            predict_bearing(prepared_sample, model, mini_manifest, prepared=True)
            pipeline.append((time.perf_counter_ns() - start) / 1e6)
        result.update({"pipeline_median_ms": float(np.median(pipeline)),
                       "pipeline_p95_ms": float(np.percentile(pipeline, 95)),
                       "pipeline_scope": "already denoised window -> training normalization -> logits/softmax/label; raw preprocessing not measured"})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="artifacts/bearing/data/dataset.npz")
    parser.add_argument("--output", default="artifacts/bearing")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--dimension", type=int, default=24)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--benchmark-repeats", type=int, default=100)
    parser.add_argument("--selection-tolerance", type=float, default=.01,
                        help="max allowed absolute drop from best validation accuracy and macro F1")
    parser.add_argument("--protocol", choices=("record_split", "load3_holdout"), default="record_split")
    args = parser.parse_args()
    if args.epochs < 1 or args.threads < 1 or args.batch_size < 1 or args.benchmark_repeats < 10:
        parser.error("epochs/threads/batch-size must be positive; benchmark-repeats must be >=10")
    if args.learning_rate <= 0 or not 0 <= args.selection_tolerance <= .1:
        parser.error("invalid learning rate or selection tolerance")
    torch.set_num_threads(args.threads)
    np.random.seed(args.seed)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    X, y, partitions, data_protocol, raw_example = load_dataset(args.data, args.protocol, args.seed)
    # IDs are read again only for reporting per-record support, never for model inputs.
    if Path(args.data).is_dir():
        record_ids = np.load(Path(args.data) / "record_id.npy", allow_pickle=False).astype(str)
    else:
        with np.load(args.data, allow_pickle=False) as archive:
            record_ids = archive["record_id"].astype(str)
    normalization = fit_normalization(X, partitions)
    inputs, labels = normalize_bearing(X, normalization), torch.from_numpy(y)
    experiments, models = {}, {}
    for kind in MODEL_KINDS:
        model, evidence = train_one(kind, inputs, labels, partitions, args)
        validation, _ = evaluate(model, inputs[partitions == "validation"], labels[partitions == "validation"],
                                 record_ids=record_ids[partitions == "validation"])
        evidence.update({"validation": validation, "profile": computation_profile(model),
                         "benchmark": benchmark(model, X[np.flatnonzero(partitions == "validation")[0]:
                                                          np.flatnonzero(partitions == "validation")[0] + 1],
                                                raw_example, normalization, args.benchmark_repeats)})
        experiments[kind], models[kind] = evidence, model
        torch.save(model.state_dict(), output / f"model_{kind}.pt")
        evidence["serialized_weights_bytes"] = (output / f"model_{kind}.pt").stat().st_size
    best_accuracy = max(evidence["validation"]["accuracy"] for evidence in experiments.values())
    best_f1 = max(evidence["validation"]["macro_f1"] for evidence in experiments.values())
    eligible = [kind for kind in MODEL_KINDS
                if experiments[kind]["validation"]["accuracy"] >= best_accuracy - args.selection_tolerance
                and experiments[kind]["validation"]["macro_f1"] >= best_f1 - args.selection_tolerance]
    if not eligible:
        # If two maxima belong to different models, keep accuracy as the primary criterion.
        eligible = [max(MODEL_KINDS, key=lambda kind: experiments[kind]["validation"]["accuracy"])]
    selected = min(eligible, key=lambda kind: experiments[kind]["benchmark"]["pipeline_p95_ms"])
    # Test labels enter only after the model choice has been fixed.
    predictions = {}
    for kind, model in models.items():
        test, predicted = evaluate(model, inputs[partitions == "test"], labels[partitions == "test"],
                                   record_ids=record_ids[partitions == "test"])
        experiments[kind]["test"] = test
        predictions[kind] = predicted
    torch.save(models[selected].state_dict(), output / "model.pt")
    np.savez_compressed(output / "test_predictions.npz", y=y[partitions == "test"],
                        **predictions)
    softmax_profile, linear_profile = experiments["softmax"]["profile"], experiments["linear"]["profile"]
    comparison = {
        "reference": "same CNN, dimension/heads, FFN, optimization and partitions; attention kernel changes",
        "attention_parameter_counts_equal": softmax_profile["parameter_count"] == linear_profile["parameter_count"],
        "estimated_total_mac_reduction_linear_vs_softmax":
            1 - linear_profile["estimated_macs"] / softmax_profile["estimated_macs"],
        "estimated_attention_core_mac_reduction_linear_vs_softmax":
            1 - linear_profile["mac_breakdown"]["attention_core"] / softmax_profile["mac_breakdown"]["attention_core"],
        "measured_pipeline_p95_speedup_linear_vs_softmax":
            experiments["softmax"]["benchmark"]["pipeline_p95_ms"] / experiments["linear"]["benchmark"]["pipeline_p95_ms"],
        "interpretation": "MACs omit several operations; linear complexity need not reduce short-sequence wall-clock latency",
    }
    # Provenance checks may finish while CPU training runs. Refresh metadata only;
    # the actual data bytes used for training must remain identical.
    if Path(args.data).is_file() and _checksum(args.data) != data_protocol["dataset_sha256"]:
        raise ValueError("dataset bytes changed during training; discard this run and rerun")
    data_protocol["source_manifest"] = json.loads(
        Path(data_protocol["source_manifest_path"]).read_text(encoding="utf-8"))
    manifest = {
        "version": f"cwru-classification-{args.protocol}-seed{args.seed}-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "task": "bearing_fault_classification", "source": "real CWRU vibration, seeded bearing defects",
        "class_labels": CLASS_LABELS, "class_names_zh": CLASS_NAMES_ZH,
        "selected_model": selected,
        "model_config": {"kind": selected, "dimension": args.dimension, "heads": args.heads,
                         "window_length": WINDOW_LENGTH},
        "normalization": normalization,
        "input_contract": {"sample_rate_hz": 12000, "window_points": 1024, "channels": 1,
                           "raw_pipeline": "demean + level-3 Haar soft threshold + training-only scalar mean/std",
                           "prepared_argument": "prepared=True skips demean/Haar only, for processed NPZ windows"},
        "weights_sha256": _checksum(output / "model.pt"),
        "data_protocol": data_protocol, "experiments": experiments, "comparison": comparison,
        "selection": {"uses_test_labels": False, "tolerance_absolute": args.selection_tolerance,
                      "eligible_models": eligible,
                      "rule": "within tolerance of best validation accuracy and macro F1, then smallest CPU pipeline P95",
                      "selected": selected},
        "training": {"seed": args.seed, "epochs": args.epochs, "batch_size": args.batch_size,
                     "learning_rate": args.learning_rate, "optimizer": "AdamW", "weight_decay": 1e-4,
                     "class_weight_fit": "inverse frequency from training only", "dropout": 0},
        "runtime": {"python": platform.python_version(), "torch": torch.__version__,
                    "platform": platform.platform(), "threads": args.threads},
        "legacy_model_context": {"parameters": 17789,
                                 "task": "synthetic six-feature, four-wear-stage/HI/RUL multitask Transformer-LSTM",
                                 "comparison_limit": "different inputs/labels/tasks; not a fair accuracy or compute baseline for CWRU classifiers"},
        "limitations": ["single random seed; overlapping windows correlated",
                        "record-disjoint does not guarantee physical-bearing-disjoint specimens",
                        "seeded CWRU defects do not establish real electric-drill field performance",
                        "no life labels: no HI/RUL/3-7 day early-warning claims",
                        "softmax scores uncalibrated; no unknown-fault or sensor-quality guarantee",
                        "CPU processing benchmark excludes acquisition/disk/network and is not a measured MCU result",
                        "analytical MACs are not measured FLOPs, RAM or energy; no verified 70% compute target"],
    }
    text = json.dumps(manifest, ensure_ascii=False, indent=2)
    (output / "manifest.json").write_text(text, encoding="utf-8")
    (output / "benchmark_report.json").write_text(text, encoding="utf-8")
    print(json.dumps({"selected_model": selected, "test": experiments[selected]["test"],
                      "comparison": comparison}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
