"""Generate controlled simulated waveform runs, not real hardware experiments.

From repository root:
    python scripts/run_virtual_lab.py --output artifacts/lab --duration 2 --seeds 2
    python scripts/run_virtual_lab.py --output artifacts/lab/raw_demo --save-waveforms
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from predictive.lab import create_lab_run, export_lab_run, extract_lab_features


TOOLS = ("drill", "grinder", "impact_wrench")
GRADES = ("normal", "mild", "moderate", "severe")
LOADS = (.25, .75)
GRADE_NAMES = dict(zip(GRADES, ("正常", "轻度", "中度", "重度")))


def _arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/lab"))
    parser.add_argument("--duration", type=float, default=2.0,
                        help="Seconds per simulated run, 0.5 to 120 (default: 2)")
    parser.add_argument("--seeds", type=int, default=2,
                        help="Number of consecutive generator seeds, 1 to 100 (default: 2)")
    parser.add_argument("--base-seed", type=int, default=42,
                        help="First seed; the default two seeds are 42 and 43")
    parser.add_argument("--save-waveforms", action="store_true",
                        help="Also export raw NPZ, three CSVs and metadata/features for every run")
    args = parser.parse_args(argv)
    if not math.isfinite(args.duration) or not .5 <= args.duration <= 120:
        parser.error("--duration must be finite and in [0.5, 120]")
    if not 1 <= args.seeds <= 100:
        parser.error("--seeds must be in [1, 100]")
    if args.base_seed < 0:
        parser.error("--base-seed must be non-negative")
    if args.output.exists() and not args.output.is_dir():
        parser.error("--output must be a directory path")
    return args


def _sanity_checks(rows):
    """Validate software outputs and generator trends, not diagnostic accuracy."""
    if len({row["run_id"] for row in rows}) != len(rows):
        raise RuntimeError("run IDs must be unique")
    feature_names = tuple(extract for extract in rows[0] if extract.startswith((
        "vibration_", "current_", "temperature_", "shaft_frequency_")))
    if not all(math.isfinite(float(row[key])) for row in rows for key in feature_names):
        raise RuntimeError("features and sampling rates must be finite")
    cohorts = {}
    for row in rows:
        key = (row["tool"], row["load"], row["rpm"], row["seed"], row["window_seconds"])
        cohorts.setdefault(key, {})[row["damage"]] = row
    trend_keys = ("vibration_raw_rms", "current_mean", "temperature_slope")
    for cohort in cohorts.values():
        if set(cohort) != set(GRADES):
            raise RuntimeError("each controlled cohort must contain all four generator grades")
        for feature in trend_keys:
            values = [cohort[grade][feature] for grade in GRADES]
            if not all(a < b for a, b in zip(values[:-1], values[1:])):
                raise RuntimeError(f"the intended generator trend failed for {feature}: {values}")
    return {
        "unique_run_ids": True,
        "finite_feature_values": True,
        "controlled_damage_comparisons": len(cohorts),
        "monotonic_generator_effects_checked": list(trend_keys),
        "interpretation": "Checks of this generator's construction; not independent fault/RUL validation.",
    }


def main(argv=None):
    args = _arguments(argv)
    seeds = list(range(args.base_seed, args.base_seed + args.seeds))
    rows = []
    tool_defaults = {}
    units = None
    sampling_rates = None
    waveform_directories = {}
    for tool in TOOLS:
        for damage in GRADES:
            for load in LOADS:
                for seed in seeds:
                    run = create_lab_run(tool=tool, damage=damage, load=load,
                                         duration_seconds=args.duration, seed=seed)
                    metadata = run.metadata
                    tool_defaults.setdefault(tool, {
                        "name": metadata["tool_name"], "rpm": metadata["rpm"],
                        "modeled_shaft_reference": metadata["rpm_reference"],
                        "resonance_hz": metadata["model_assumptions"]["vibration_resonance_hz"],
                        "bearing_excitation_to_shaft_ratio": metadata["model_assumptions"]["bearing_ratio"],
                        "thermal_time_constant_seconds": metadata["model_assumptions"]["thermal_time_constant_seconds"],
                        "healthy_operating_impacts": metadata["model_assumptions"]["operating_impacts_in_healthy_data"],
                    })
                    units, sampling_rates = metadata["units"], run.sampling_rates
                    run_id = f"{tool}-{damage}-load{int(round(load * 100)):03d}-seed{seed}"
                    row = {
                        "run_id": run_id,
                        "source": "synthetic_raw_waveform_lab",
                        "label_origin": "generator_setting",
                        "tool": tool, "tool_name": metadata["tool_name"],
                        "damage": damage, "damage_name": GRADE_NAMES[damage],
                        "load": load, "rpm": metadata["rpm"], "seed": seed,
                        "window_seconds": args.duration,
                        **{name + "_sample_rate_hz": run.sampling_rates[name]
                           for name in ("vibration", "current", "temperature")},
                        **extract_lab_features(run),
                    }
                    rows.append(row)
                    if args.save_waveforms:
                        relative_directory = Path("runs") / run_id
                        export_lab_run(run, args.output / relative_directory)
                        waveform_directories[run_id] = relative_directory.as_posix()
    sanity = _sanity_checks(rows)
    args.output.mkdir(parents=True, exist_ok=True)
    csv_path = args.output / "runs.csv"
    # UTF-8 BOM keeps Chinese labels readable in Windows Excel.
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    manifest = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": "synthetic_raw_waveform_lab", "is_simulated": True,
        "label_origin": "generator_setting",
        "run_count": len(rows),
        "experiment_grid": {"tools": list(TOOLS), "damage_grades": list(GRADES),
                            "loads": list(LOADS), "seeds": seeds,
                            "window_seconds": args.duration},
        "tool_defaults": tool_defaults,
        "counts": {"per_tool": dict(Counter(row["tool"] for row in rows)),
                   "per_damage": dict(Counter(row["damage"] for row in rows)),
                   "per_load": dict(Counter(str(row["load"]) for row in rows))},
        "sampling_rates_hz": sampling_rates, "signal_units": units,
        "time_reference": "per-channel seconds since a shared virtual t=0",
        "features": {
            "processing": "offline orthonormal Haar soft shrinkage; Hann FFT/PSD; temperature linear slope",
            "current_harmonic_reference": "modeled shaft orders 1,2,3, not mains/PWM harmonics",
            "temperature_slope_unit": "degC/second",
            "damage_label_used_to_extract_features": False,
            "compatible_with_original_six_feature_model_without_retraining": False,
        },
        "waveforms_saved": args.save_waveforms,
        "waveform_directories": waveform_directories,
        "summary_csv": "runs.csv",
        "csv_sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
        "implementation_sha256": {
            "predictive/lab.py": hashlib.sha256((ROOT / "predictive/lab.py").read_bytes()).hexdigest(),
            "scripts/run_virtual_lab.py": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
        "environment": {"python": sys.version.split()[0], "numpy": version("numpy"),
                        "scipy": version("scipy")},
        "sanity_checks": sanity,
        "interpretation_limits": [
            "These are generated condition runs, not independent tools or measured hardware experiments.",
            "No commercial tool, real bearing geometry, motor driver or sensor model was calibrated.",
            "The four grades are input settings, not public bearing dataset fault classes or verified wear labels.",
            "The batch contains no real failure endpoints, RUL ground truth or diagnostic accuracy evidence.",
            "48 default runs do not meet or verify the task book's 5000 real multi-condition samples target.",
        ],
        "independent_real_hardware_runs": 0,
        "taskbook_real_samples_target_verified": False,
    }
    manifest_path = args.output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"runs": len(rows), "controlled_cohorts": sanity["controlled_damage_comparisons"],
                      "csv": str(csv_path), "manifest": str(manifest_path),
                      "waveforms_saved": args.save_waveforms}, ensure_ascii=False))
    return manifest


if __name__ == "__main__":
    main()
