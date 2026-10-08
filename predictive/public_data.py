"""Traceable CWRU vibration preparation, kept separate from simulated tool labels.

The four labels describe seeded fault locations, not natural wear or remaining life.
Splits isolate complete recordings; the same physical seeded bearing may still be
shared across load recordings. No temperature/current channels are invented.
"""
from __future__ import annotations

import hashlib
import json
import math
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.signal import hilbert, resample_poly

MIRROR_COMMIT = "0d07cb1caf0a5a68a955c2f43cfc21b9a208406a"
MIRROR_REPOSITORY = "https://github.com/s-whynot/CWRU-dataset"
ORIGIN_URL = "https://engineering.case.edu/bearingdatacenter"
NORMAL_RATE_BASIS = {
    "rate_hz": 48000,
    "source_type": "secondary_academic_open_source_mapping",
    "repository": "https://github.com/Aalto-Arotor/openConMo",
    "commit": "d2924b6683f1ba49661fae63df963d71efa10c15",
    "url": "https://github.com/Aalto-Arotor/openConMo/blob/d2924b6683f1ba49661fae63df963d71efa10c15/scripts/CWRU_download.py#L318",
    "interpretation": "openConMo maps normal files to 48 kHz; CWRU normal table and MAT do not supply per-record sampling rates",
    "official_mat_sample_rate_field_verified": False,
    "limitation": "This secondary mapping is the preprocessing convention; byte checksums do not verify the sampling-rate metadata.",
}
SAMPLE_RATE = 12000
WINDOW_SIZE = 1024
STRIDE = 512
CLASS_NAMES = ["正常", "内圈缺陷", "滚动体缺陷", "外圈缺陷"]
FAULT_TYPES = ["normal", "inner_race", "ball", "outer_race"]
FEATURE_COLUMNS = [
    "vibration_rms", "kurtosis", "crest_factor", "peak_to_peak",
    "spectral_centroid_hz", "peak_frequency_hz", "spectral_entropy",
    "band_energy_low", "band_energy_mid", "band_energy_high",
    "envelope_peak_frequency_hz",
]


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def cwru_records():
    """40 selected official record identities and verified mirror paths."""
    records = []
    for load, number in enumerate(range(97, 101)):
        records.append(_record(number, load, 0, 0, f"Normal/{number}_Normal_{load}.mat", 48000))
    starts = {"IR": [105, 169, 209], "B": [118, 185, 222], "OR": [130, 197, 234]}
    label = {"IR": 1, "B": 2, "OR": 3}
    for kind, numbers in starts.items():
        for diameter, first in zip([7, 14, 21], numbers):
            for load in range(4):
                number = first + load
                base = f"12k_Drive_End_Bearing_Fault_Data/{kind}/{diameter:03d}"
                if kind == "OR":
                    if diameter in (7, 21):
                        base += "/@6"
                    name = f"{number}@6_{load}.mat" if diameter in (7, 14) else f"{number}_{load}.mat"
                else:
                    name = f"{number}_{load}.mat"
                records.append(_record(number, load, label[kind], diameter, f"{base}/{name}", 12000))
    return records


def _record(number, load, fault, diameter_mil, mirror_path, rate):
    return {
        "record_id": f"CWRU-X{number:03d}", "original_mat_id": number,
        "fault_label": fault, "fault_type": FAULT_TYPES[fault],
        "class_name": CLASS_NAMES[fault], "load_hp": load,
        "nominal_rpm": [1797, 1772, 1750, 1730][load],
        "defect_diameter_inch": diameter_mil / 1000,
        "defect_diameter_mm": diameter_mil * 0.0254,
        "damage_label": "正常" if not fault else {7: "小尺寸人工缺陷", 14: "中尺寸人工缺陷", 21: "大尺寸人工缺陷"}[diameter_mil],
        "outer_race_position": "6 o'clock" if fault == 3 else None,
        "specimen_group": f"{FAULT_TYPES[fault]}-{diameter_mil:02d}-DE",
        "channel": "DE", "original_sample_rate": rate, "sample_rate": SAMPLE_RATE,
        "sample_rate_source": (NORMAL_RATE_BASIS["url"] if fault == 0 else
                               ORIGIN_URL + "/12k-drive-end-bearing-fault-data"),
        "sample_rate_basis": ("secondary normal=48k mapping; not a MAT field" if fault == 0 else
                              "official 12k drive-end experimental group"),
        "vibration_unit": "数据集加速度原单位（未独立校准）",
        "mirror_path": mirror_path,
        "official_file_url": f"https://engineering.case.edu/sites/default/files/{number}.mat",
        "download_url": f"https://raw.githubusercontent.com/s-whynot/CWRU-dataset/{MIRROR_COMMIT}/{quote(mirror_path, safe='/')}",
        "original_dataset_url": ORIGIN_URL,
    }


def assign_record_splits(records, seed=42):
    """28/8/4 recording split, stratified by the four fault-location classes."""
    rng = np.random.default_rng(seed)
    allocations = {0: (2, 1, 1), 1: (9, 2, 1), 2: (8, 3, 1), 3: (9, 2, 1)}
    result = {}
    for fault, counts in allocations.items():
        subset = sorted([r for r in records if r["fault_label"] == fault], key=lambda r: r["record_id"])
        order = rng.permutation(len(subset))
        offset = 0
        for split, count in zip(("train", "validation", "test"), counts):
            for index in order[offset: offset + count]:
                result[subset[int(index)]["record_id"]] = split
            offset += count
    return result


def download_cwru(raw_dir, workers=4, manifest_path=None):
    """Fetch original-format MATs from a pinned third-party mirror with curl.

    Official site access may be unavailable in the cloud network. This is stated
    in the manifest; a mirror SHA proves our exact bytes, not official equivalence.
    """
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    previous = {}
    if manifest_path and Path(manifest_path).exists():
        previous = {r["record_id"]: r for r in json.loads(Path(manifest_path).read_text()).get("records", [])}

    def fetch(record):
        path = raw_dir / f"{record['original_mat_id']}.mat"
        old = previous.get(record["record_id"], {})
        if path.exists() and old.get("sha256") and sha256_file(path) != old["sha256"]:
            raise ValueError(f"cached raw file checksum mismatch: {path}")
        if not path.exists():
            partial = path.with_suffix(".download")
            try:
                subprocess.run(["curl", "-L", "--fail", "--retry", "3", "--max-time", "180", "-sS",
                                record["download_url"], "-o", str(partial)], check=True)
                # Reject HTML/error bodies before allowing the artifact into a cache.
                load_record_waveform(partial, record)
                partial.replace(path)
            finally:
                partial.unlink(missing_ok=True)
        values, rpm = load_record_waveform(path, record)
        raw_hash = sha256_file(path)
        verified = bool(old.get("official_sha256") == raw_hash and old.get("official_match"))
        official_hash = old.get("official_sha256") if verified else None
        official_status = "cached_previous_byte_match" if verified else "not_verified"
        if not verified:
            official_temp = raw_dir / f"{record['original_mat_id']}.official-download"
            try:
                check = subprocess.run(["curl", "-L", "--fail", "--retry", "1", "--max-time", "90", "-sS",
                                        record["official_file_url"], "-o", str(official_temp)], capture_output=True)
                if check.returncode == 0:
                    official_hash = sha256_file(official_temp)
                    if official_hash != raw_hash:
                        raise ValueError(f"mirror bytes differ from official CWRU file: {record['record_id']}")
                    verified = True
                    official_status = "official_download_byte_match"
                else:
                    official_status = f"official_unavailable_curl_exit_{check.returncode}"
            finally:
                official_temp.unlink(missing_ok=True)
        return {**record, "sha256": raw_hash, "official_sha256": official_hash,
                "official_match": verified, "official_verification": official_status,
                "bytes": path.stat().st_size,
                "original_samples": len(values), "rpm": rpm, "raw_file": str(path)}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        records = list(pool.map(fetch, cwru_records()))
    return records


def load_record_waveform(path, record):
    data = loadmat(path)
    prefix = f"X{record['original_mat_id']:03d}"
    channel_key = f"{prefix}_DE_time"
    if channel_key not in data:
        # A few CWRU MAT metadata differ in prefix; never arbitrarily select a
        # duplicate DE channel. A unique channel is safe and tracked in metadata.
        candidates = [key for key in data if key.endswith("_DE_time")]
        if len(candidates) != 1:
            raise ValueError(f"ambiguous DE channel in {path}: {candidates}")
        channel_key = candidates[0]
    record["mat_channel_key"] = channel_key
    values = np.asarray(data[channel_key], dtype=np.float64).reshape(-1)
    if len(values) < WINDOW_SIZE or not np.isfinite(values).all():
        raise ValueError(f"invalid waveform in {path}")
    rpm_keys = [key for key in data if key == f"{prefix}RPM"]
    if not rpm_keys:
        rpm_keys = [key for key in data if key.endswith("RPM")]
    if len(rpm_keys) > 1:
        raise ValueError(f"ambiguous RPM metadata in {path}")
    if rpm_keys:
        rpm = float(np.asarray(data[rpm_keys[0]]).reshape(-1)[0])
        record["rpm_source"] = "MAT测量元数据"
    else:
        rpm = float(record["nominal_rpm"])
        record["rpm_source"] = "官网工况表标称值（此MAT未提供RPM）"
    return values, rpm


def resample_waveform(x, sample_rate, target_rate=SAMPLE_RATE):
    values = np.asarray(x, dtype=np.float64).reshape(-1)
    if not np.isfinite(values).all() or len(values) < 16:
        raise ValueError("waveform must contain at least 16 finite samples")
    if sample_rate <= 0 or target_rate <= 0:
        raise ValueError("sample rates must be positive")
    if sample_rate != target_rate:
        divisor = math.gcd(int(sample_rate), int(target_rate))
        # FIR polyphase resampling applies an anti-aliasing low-pass filter.
        values = resample_poly(values, int(target_rate) // divisor, int(sample_rate) // divisor)
    return values


def denoise_wavelet(x, level=3):
    """Haar DWT soft threshold, sigma=MAD(detail1)/0.67449, universal threshold.

    This is offline processing of one completed window. It uses no future
    windows and estimates no pooled threshold from validation/test recordings.
    """
    values = np.asarray(x, dtype=np.float64).reshape(-1)
    if len(values) < 16 or not np.isfinite(values).all():
        raise ValueError("wavelet input must contain at least 16 finite samples")
    if level < 1 or level > 6:
        raise ValueError("Haar levels must be between 1 and 6")
    original_length = len(values)
    block = 2 ** level
    padding = (-original_length) % block
    approx = np.pad(values, (0, padding), mode="reflect") if padding else values.copy()
    details = []
    for _ in range(level):
        detail = (approx[0::2] - approx[1::2]) / np.sqrt(2)
        approx = (approx[0::2] + approx[1::2]) / np.sqrt(2)
        details.append(detail)
    sigma = np.median(np.abs(details[0] - np.median(details[0]))) / 0.6744897501960817
    threshold = sigma * np.sqrt(2 * np.log(original_length))
    for detail in reversed(details):
        detail = np.sign(detail) * np.maximum(np.abs(detail) - threshold, 0)
        restored = np.empty(len(approx) * 2, dtype=np.float64)
        restored[0::2] = (approx + detail) / np.sqrt(2)
        restored[1::2] = (approx - detail) / np.sqrt(2)
        approx = restored
    return approx[:original_length]


def preprocess_waveform(x, sample_rate=SAMPLE_RATE, target_rate=SAMPLE_RATE):
    values = resample_waveform(x, sample_rate, target_rate)
    values = values - values.mean()
    return denoise_wavelet(values, level=3).astype(np.float32)


def extract_vibration_features(x, sample_rate=SAMPLE_RATE, prepared=False):
    values = np.asarray(x, dtype=np.float64).reshape(-1) if prepared else preprocess_waveform(x, sample_rate)
    if not prepared:
        sample_rate = SAMPLE_RATE
    if not np.isfinite(values).all() or len(values) < 16:
        raise ValueError("invalid feature waveform")
    values = values - values.mean()
    rms = float(np.sqrt(np.mean(values ** 2)))
    spectrum = np.abs(np.fft.rfft(values * np.hanning(len(values)))) ** 2
    frequencies = np.fft.rfftfreq(len(values), 1 / sample_rate)
    spectrum[0] = 0
    total = float(spectrum.sum())
    probs = spectrum / max(total, 1e-30)
    positive = probs[probs > 0]
    envelope = np.abs(hilbert(values))
    envelope = envelope - envelope.mean()
    envelope_spectrum = np.abs(np.fft.rfft(envelope * np.hanning(len(values)))) ** 2
    envelope_spectrum[0] = 0
    return {
        "vibration_rms": rms,
        "kurtosis": float(np.mean(values ** 4) / max(rms ** 4, 1e-30)),
        "crest_factor": float(np.max(np.abs(values)) / max(rms, 1e-15)),
        "peak_to_peak": float(np.ptp(values)),
        "spectral_centroid_hz": float(np.dot(frequencies, probs)),
        "peak_frequency_hz": float(frequencies[np.argmax(spectrum)]),
        "spectral_entropy": float(-np.dot(positive, np.log(positive)) / np.log(len(probs))),
        "band_energy_low": float(probs[(frequencies >= 0) & (frequencies < 1000)].sum()),
        "band_energy_mid": float(probs[(frequencies >= 1000) & (frequencies < 3000)].sum()),
        "band_energy_high": float(probs[frequencies >= 3000].sum()),
        "envelope_peak_frequency_hz": float(frequencies[np.argmax(envelope_spectrum)]),
    }


def prepare_cwru(raw_dir="data/public/CWRU/raw", output_dir="artifacts/bearing/data", max_windows_per_record=256, seed=42, download=True):
    raw_dir, output_dir = Path(raw_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_manifest_path = raw_dir.parent / "manifest.json"
    if download:
        records = download_cwru(raw_dir, manifest_path=raw_manifest_path)
    else:
        records = []
        for item in cwru_records():
            path = raw_dir / f"{item['original_mat_id']}.mat"
            values, rpm = load_record_waveform(path, item)
            previous = json.loads(raw_manifest_path.read_text()).get("records", []) if raw_manifest_path.exists() else []
            old = next((r for r in previous if r["record_id"] == item["record_id"]), {})
            records.append({**item, "sha256": sha256_file(path), "official_sha256": old.get("official_sha256"),
                            "official_match": bool(old.get("official_match") and old.get("sha256") == sha256_file(path)),
                            "official_verification": old.get("official_verification", "offline_not_verified"), "bytes": path.stat().st_size,
                            "original_samples": len(values), "rpm": rpm, "raw_file": str(path)})
    split_map = assign_record_splits(records, seed)
    arrays, rows = [], []
    raw_example = None
    for record in records:
        values, _ = load_record_waveform(record["raw_file"], record)
        values = resample_waveform(values, record["original_sample_rate"])
        starts = np.arange(0, len(values) - WINDOW_SIZE + 1, STRIDE)
        if max_windows_per_record and len(starts) > max_windows_per_record:
            starts = starts[np.linspace(0, len(starts) - 1, max_windows_per_record).round().astype(int)]
        record["split"] = split_map[record["record_id"]]
        record["prepared_samples"] = len(values)
        record["window_count"] = len(starts)
        record["raw_file"] = f"raw/{record['original_mat_id']}.mat"
        for sequence, start in enumerate(starts):
            raw_window = values[start: start + WINDOW_SIZE]
            waveform = preprocess_waveform(raw_window)
            if raw_example is None and record["original_sample_rate"] == SAMPLE_RATE:
                raw_example = raw_window.astype(np.float32)
                example_metadata = {"record_id": record["record_id"], "start_seconds": start / SAMPLE_RATE, "sample_rate": SAMPLE_RATE}
            arrays.append(waveform)
            row = {"record_id": record["record_id"], "specimen_group": record["specimen_group"],
                   "sequence": sequence, "start_seconds": start / SAMPLE_RATE, "duration_seconds": WINDOW_SIZE / SAMPLE_RATE,
                   "split": record["split"], "load_hp": record["load_hp"], "rpm": record["rpm"],
                   "sample_rate": SAMPLE_RATE, "channel": "DE", "fault_label": record["fault_label"],
                   "fault_type": record["fault_type"], "class_name": record["class_name"],
                   "defect_diameter_mm": record["defect_diameter_mm"], "damage_label": record["damage_label"],
                   **extract_vibration_features(waveform, prepared=True)}
            rows.append(row)
    frame = pd.DataFrame(rows)
    dataset = np.asarray(arrays, dtype=np.float32)[:, None, :]
    output_npz = output_dir / "dataset.npz"
    np.savez_compressed(output_npz, X=dataset, y=frame.fault_label.to_numpy(dtype=np.int64),
                        split=frame.split.to_numpy(dtype=str), record_id=frame.record_id.to_numpy(dtype=str),
                        load_hp=frame.load_hp.to_numpy(dtype=np.int64), rpm=frame.rpm.to_numpy(dtype=np.float32),
                        defect_diameter_mm=frame.defect_diameter_mm.to_numpy(dtype=np.float32),
                        start_seconds=frame.start_seconds.to_numpy(dtype=np.float64),
                        raw_example=raw_example)
    frame.to_csv(output_dir / "features.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(records).to_csv(output_dir / "records.csv", index=False, encoding="utf-8-sig")
    official_matches = sum(r.get("official_match", False) for r in records)
    raw_manifest = {
        "dataset": "Case Western Reserve University Bearing Data Center", "origin_url": ORIGIN_URL,
        "download_source": "third_party_original_format_MAT_mirror", "mirror_repository": MIRROR_REPOSITORY,
        "mirror_commit": MIRROR_COMMIT,
        "provenance_status": f"固定commit镜像原格式MAT，{official_matches}/40文件与官网独立下载SHA256一致；其余见逐记录校验状态。",
        "official_access_note": "初次云环境官网HTTP403；后续连通恢复，逐文件独立下载官网数据并比较SHA256。缓存记录保留此前校验结果。",
        "official_verified_files": official_matches,
        "normal_sample_rate_basis": NORMAL_RATE_BASIS,
        "records": records,
    }
    raw_manifest_path.write_text(json.dumps(raw_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    specimen_sets = {split: set(frame.loc[frame.split == split, "specimen_group"]) for split in ("train", "validation", "test")}
    manifest = {
        "schema_version": 1, "task": "bearing_fault_classification", "source": "CWRU",
        "class_names": CLASS_NAMES, "class_labels": FAULT_TYPES, "sample_rate": SAMPLE_RATE,
        "channel": "DE", "vibration_unit": "数据集加速度原单位（未独立校准）", "window_size": WINDOW_SIZE,
        "stride": STRIDE, "seed": seed, "total_windows": len(frame), "total_records": len(records),
        "record_split": {key: sum(r["split"] == key for r in records) for key in ("train", "validation", "test")},
        "window_split": frame.split.value_counts().to_dict(),
        "class_counts": frame.class_name.value_counts().to_dict(),
        "features_columns": FEATURE_COLUMNS,
        "normal_sample_rate_basis": NORMAL_RATE_BASIS,
        "preprocessing": {"resampling": "48k正常记录以scipy.signal.resample_poly(up=1,down=4)抗混叠降至12k；故障DE12不重采样",
                          "dc": "每个完成窗口独立去均值", "wavelet": "3级Haar DWT软阈值，MAD/0.67448975估计噪声，阈值sigma*sqrt(2*ln(N))",
                          "normalization": "本数据未做逐窗幅值归一化，训练器只使用训练集拟合全局标准化参数",
                          "feature_spectrum": "Hann窗FFT功率谱、包络谱和时域统计，频率分辨率11.71875Hz"},
        "split_protocol": "40个完整MAT记录按四类分层，28训练/8验证/4测试；同一记录所有重叠窗不跨分割。7:2:1按记录数，窗口比例由记录长度决定。",
        "leakage_limitations": "同一人工缺陷配置跨负载可能来自同一个物理轴承，因此record隔离不等于独立轴承；窗口也不是独立寿命试验。",
        "shared_specimen_groups_train_test": sorted(specimen_sets["train"] & specimen_sets["test"]),
        "additional_evaluation": {"protocol": "load3_holdout", "train_candidate_loads": [0, 1, 2], "validation_candidate_loads": [0, 1, 2], "test_loads": [3],
                                  "record_split": {"train": 23, "validation": 7, "test": 10},
                                  "validation_selection": "seed=42，按fault_label=0/1/2/3将候选record_id升序后随机排列；每类前max(1,round(n*0.2))为验证，其余训练。",
                                  "note": "完整负载3记录留出；训练/验证均可含负载0/1/2的不同文件，物理轴承仍可能共享，属于跨负载预验证。"},
        "missing_channels": ["current", "temperature"], "unavailable_targets": ["natural_wear_stage", "remaining_useful_life"],
        "raw_example": example_metadata,
        "raw_manifest": "data/public/CWRU/manifest.json", "dataset_sha256": sha256_file(output_npz),
        "features_sha256": sha256_file(output_dir / "features.csv"),
        "records_sha256": sha256_file(output_dir / "records.csv"),
        "official_verified_files": official_matches,
        "dataset_bytes": output_npz.stat().st_size, "records": records,
    }
    (output_dir / "dataset_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest
