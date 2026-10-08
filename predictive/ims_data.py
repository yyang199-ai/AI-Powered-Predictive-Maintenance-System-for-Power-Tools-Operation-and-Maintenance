"""Validated NASA/IMS run 2 import and descriptive degradation features.

The endpoint countdown is a retrospective dataset proxy, not an RUL predictor.
Only bearing 1 has the end-of-run failure described by a mirrored source PDF;
bearings 2-4 are censored. No synthetic current, temperature or wear labels are
added to these vibration recordings.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
import hashlib
import json
import re

import numpy as np
import pandas as pd
from scipy.stats import kurtosis

from .lab import haar_soft_denoise


IMS_RUN2_START = datetime(2004, 2, 12, 10, 32, 39)
IMS_RUN2_END = datetime(2004, 2, 19, 6, 22, 39)
IMS_RUN2_COUNT = 984
IMS_SAMPLING_RATE_HZ = 20_000
IMS_SNAPSHOT_POINTS = 20_480
IMS_INTERVAL_SECONDS = 600
IMS_MIRROR_COMMIT = "24d07ee057edac93866aef41e369539dce303da0"
IMS_MIRROR = "https://github.com/RicardoPSLopes/IMS-DATASET"
IMS_ARCHIVE_URL = f"https://codeload.github.com/RicardoPSLopes/IMS-DATASET/zip/{IMS_MIRROR_COMMIT}"
IMS_DESCRIPTION_URL = "https://raw.githubusercontent.com/Miltos-90/Failure_Classification_of_Bearings/main/Data%20description.pdf"
IMS_DESCRIPTION_SHA256 = "cf46d37c21f7f292c11bbbdd4695d876c417ed1d6425e3d87c962ae2182ae6ed"
_FILENAME = re.compile(r"^\d{4}\.\d{2}\.\d{2}\.\d{2}\.\d{2}\.\d{2}$")


def snapshot_timestamp(filename: str) -> datetime:
    if not _FILENAME.fullmatch(filename):
        raise ValueError("IMS snapshot filenames must have YYYY.MM.DD.HH.MM.SS format")
    try:
        return datetime.strptime(filename, "%Y.%m.%d.%H.%M.%S")
    except ValueError as exc:
        raise ValueError(f"invalid snapshot timestamp: {filename}") from exc


def ims_run2_files(raw_dir) -> list[Path]:
    folder = Path(raw_dir)
    if not folder.is_dir():
        raise ValueError(f"IMS raw directory does not exist: {folder}")
    files = sorted(path for path in folder.iterdir()
                   if path.is_file() and _FILENAME.fullmatch(path.name))
    if not files:
        raise ValueError("no timestamp-named IMS snapshots found")
    # Reject malformed timestamps even when the outer filename format matches.
    for path in files:
        snapshot_timestamp(path.name)
    return files


def check_complete_run2(files: list[Path]) -> bool:
    """Require the complete known 984-snapshot timeline, not just two endpoints."""
    actual = [snapshot_timestamp(path.name) for path in files]
    expected = [IMS_RUN2_START + timedelta(seconds=IMS_INTERVAL_SECONDS * i)
                for i in range(IMS_RUN2_COUNT)]
    return actual == expected


def read_ims_snapshot(path) -> np.ndarray:
    """Read the four *vibration* channels of an IMS run 2 ASCII snapshot."""
    path = Path(path)
    snapshot_timestamp(path.name)
    try:
        values = np.loadtxt(path, dtype=np.float64)
    except (OSError, ValueError) as exc:
        raise ValueError(f"cannot parse IMS snapshot: {path.name}") from exc
    if values.shape != (IMS_SNAPSHOT_POINTS, 4) or not np.isfinite(values).all():
        raise ValueError(f"{path.name}: expected {IMS_SNAPSHOT_POINTS} finite rows and four channels")
    return values


def ims_vibration_features(signal) -> dict[str, float]:
    """Offline Haar soft threshold, DC removal and one-sided Hann-window FFT.

    The three bands (0-1, 1-5, 5-10 kHz) are exploratory descriptors, not
    geometry-derived fault-frequency bands. Retain raw RMS to inspect whether
    denoising suppresses impulsive degradation information.
    """
    values = np.asarray(signal, dtype=float)
    if values.ndim != 1 or len(values) < 64 or not np.isfinite(values).all():
        raise ValueError("a finite one-dimensional vibration snapshot is required")
    raw = values - values.mean()
    denoised = haar_soft_denoise(raw, levels=6)
    denoised -= denoised.mean()
    rms = float(np.sqrt(np.mean(denoised ** 2)))
    spectrum = np.abs(np.fft.rfft(denoised * np.hanning(len(denoised)))) ** 2
    frequencies = np.fft.rfftfreq(len(denoised), 1 / IMS_SAMPLING_RATE_HZ)
    power = float(spectrum.sum())
    denominator = max(power, np.finfo(float).tiny)
    bands = [(0, 1000), (1000, 5000), (5000, 10000.1)]
    fractions = [float(spectrum[(frequencies >= low) & (frequencies < high)].sum()
                       / denominator) for low, high in bands]
    return {
        "raw_vibration_rms": float(np.sqrt(np.mean(raw ** 2))),
        "vibration_rms": rms,
        "vibration_kurtosis": float(kurtosis(denoised, fisher=False, bias=False))
        if rms > 1e-15 else 0.0,
        "crest_factor": float(np.max(np.abs(denoised)) / rms) if rms > 1e-15 else 0.0,
        "peak_to_peak": float(np.ptp(denoised)),
        "spectral_peak_hz": float(frequencies[np.argmax(spectrum)]),
        "spectral_centroid_hz": float((spectrum * frequencies).sum() / denominator),
        "band_low_fraction": fractions[0], "band_mid_fraction": fractions[1],
        "band_high_fraction": fractions[2],
    }


def prepare_ims_run2(raw_dir, output_dir, *, require_complete=True) -> dict:
    """Build one feature row per snapshot/bearing and a reproducible manifest.

    Partial imports are allowed only explicitly; they never receive endpoint
    labels. Censored channels never receive finite proxy-RUL labels.
    """
    files = ims_run2_files(raw_dir)
    complete = check_complete_run2(files)
    if require_complete and not complete:
        raise ValueError("run 2 is incomplete: require all 984 expected 10-minute snapshots; "
                         "use allow-partial only for exploratory features without endpoint labels")
    rows, source_files = [], []
    start = snapshot_timestamp(files[0].name)
    for file_number, path in enumerate(files):
        values = read_ims_snapshot(path)
        timestamp = snapshot_timestamp(path.name)
        source_files.append({"filename": path.name, "bytes": path.stat().st_size,
                             "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        for channel in range(4):
            observed = bool(complete and channel == 0)
            rows.append({
                "dataset": "IMS", "run_id": "run2", "snapshot_id": file_number,
                "timestamp_iso": timestamp.isoformat(),
                "elapsed_hours": (timestamp - start).total_seconds() / 3600,
                "bearing_id": f"bearing_{channel + 1}", "channel": channel + 1,
                "sampling_rate_hz": IMS_SAMPLING_RATE_HZ,
                "snapshot_duration_seconds": IMS_SNAPSHOT_POINTS / IMS_SAMPLING_RATE_HZ,
                **ims_vibration_features(values[:, channel]),
                "endpoint_proxy_rul_hours": (IMS_RUN2_END - timestamp).total_seconds() / 3600
                if observed else np.nan,
                "event_observed": observed, "is_censored": channel != 0 or not complete,
                "label_basis": "source_description_mirror_described_end_of_run_outer_race_failure"
                if observed else "no_verified_failure_endpoint_for_this_channel",
            })
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    features_path = out / "features.csv"
    frame = pd.DataFrame(rows)
    # End snapshots in this run have very little activity in all channels.
    # This flag is an exploratory amplitude comparison, not a sensor diagnosis,
    # nor proof that the shaft stopped. Do not treat the drop as repaired health.
    pivot = frame.pivot(index="snapshot_id", columns="bearing_id", values="raw_vibration_rms")
    early_reference = pivot.iloc[:min(10, len(pivot))].median().clip(lower=1e-12)
    low_activity = (pivot.div(early_reference) < .05).all(axis=1)
    flagged_ids = set(low_activity[low_activity].index.tolist())
    frame["activity_status"] = frame["snapshot_id"].map(
        lambda i: "low_activity_all_channels" if i in flagged_ids else "not_low_by_early_reference")
    frame.to_csv(features_path, index=False)
    provenance_file = Path(raw_dir) / "download_provenance.json"
    provenance = json.loads(provenance_file.read_text()) if provenance_file.is_file() else {
        "repository": IMS_MIRROR, "commit": IMS_MIRROR_COMMIT,
        "download_method": "local_import_no_remote_hash_verification",
        "official_source_verified_this_session": False,
    }
    manifest = {
        "schema_version": 1, "dataset": "NASA/IMS bearing run 2",
        "source_kind": "public_real_bearing_test_rig_vibration",
        "source_provenance": provenance, "download_status": "complete_run2" if complete else "partial_import",
        "source_description_mirror": {"url": IMS_DESCRIPTION_URL, "sha256": IMS_DESCRIPTION_SHA256,
                                      "reviewed_pages": [1, 2], "official_host_verified": False},
        "complete_timeline_verified": complete, "snapshot_count": len(files),
        "feature_rows": len(rows), "bearings": 4, "unique_failure_events": 1 if complete else 0,
        "first_snapshot": start.isoformat(),
        "last_snapshot": snapshot_timestamp(files[-1].name).isoformat(),
        "sampling_rate_hz": IMS_SAMPLING_RATE_HZ, "snapshot_points": IMS_SNAPSHOT_POINTS,
        "snapshot_duration_seconds": IMS_SNAPSHOT_POINTS / IMS_SAMPLING_RATE_HZ,
        "nominal_snapshot_interval_seconds": IMS_INTERVAL_SECONDS,
        "timestamp_timezone": "not supplied; retain source local-clock timestamps without invented UTC conversion",
        "units": "source ASCII amplitude; original calibration units were not independently verified",
        "available_channels": ["bearing_1_vibration", "bearing_2_vibration", "bearing_3_vibration", "bearing_4_vibration"],
        "unavailable_channels": ["current", "temperature", "electric_tool_load"],
        "preprocessing": {"dc_removal": True, "denoising": "Haar six-level MAD universal soft threshold; offline",
                          "fft": "one-sided Hann-window squared-amplitude spectrum",
                          "frequency_bands_hz": [[0, 1000], [1000, 5000], [5000, 10000]],
                          "retained_raw_rms": True},
        "activity_flags": {"rule": "all four raw RMS channels below 5% of their own first-10-snapshot median",
                           "low_activity_snapshots": sorted(flagged_ids),
                           "interpretation": "low amplitude observation only; not proof of stopped shaft or recovery",
                           "computed_from_full_run_for_descriptive_review": True},
        "labels": {
            "bearing_1": "outer-race failure at end of run according to mirrored source PDF page 2; endpoint countdown proxy only",
            "bearing_2_3_4": "right-censored at recording end; no finite proxy RUL labels",
            "four_wear_grades": "not provided and not inferred from RMS or time percentiles",
            "endpoint_proxy": "wall-clock hours to final recorded snapshot, including low-activity tail; exact physical failure time and effective runtime were not verified",
            "trained_rul_predictor": False,
        },
        "partition": "descriptive full-run trend only; do not randomly split snapshots for a generalization claim",
        "limitations": [
            "A single run and one described failure cannot validate general RUL error, early warning or tool transfer.",
            "Adjacent snapshots and bearings share the same rig experiment; they are not independent devices/runs.",
            "The recorded snapshots cover 1.024 seconds every 10 minutes, not continuous vibration monitoring.",
            "NASA official archive was not independently accessed this session; provenance is a pinned third-party mirror.",
            "The mirror declares no explicit data license; source attribution is retained and original raw data is not republished in Git.",
        ],
        "features_sha256": hashlib.sha256(features_path.read_bytes()).hexdigest(),
        "files": source_files,
    }
    (out / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


def load_ims_prepared(folder="data/public/NASA_IMS/prepared"):
    """Verify the prepared CSV against its manifest before displaying trends."""
    folder = Path(folder)
    manifest = json.loads((folder / "dataset_manifest.json").read_text(encoding="utf-8"))
    path = folder / "features.csv"
    if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["features_sha256"]:
        raise ValueError("IMS feature CSV checksum does not match its manifest")
    frame = pd.read_csv(path)
    if len(frame) != manifest["feature_rows"]:
        raise ValueError("IMS feature row count does not match its manifest")
    return frame, manifest
