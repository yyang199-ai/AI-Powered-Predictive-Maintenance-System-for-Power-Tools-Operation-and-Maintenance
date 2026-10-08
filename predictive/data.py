"""Reproducible synthetic trajectories; labels are explicitly simulated."""
import numpy as np
import pandas as pd

FEATURES = ["vibration_rms", "vibration_kurtosis", "current_rms", "current_ripple", "temperature_rise", "temperature_slope"]
QUALITY = ["q_vibration", "q_current", "q_temperature"]
CONTEXT = ["rpm", "load", "ambient_c"]


def simulate_fleet(devices=30, windows=200, seed=42):
    """One-minute aggregate windows; effective lifetime is in operating hours.

    These are generated feature summaries, not acquired 12.8 kHz waveforms.
    Independent torque/load is available in this simulation.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for d in range(devices):
        lifetime = rng.uniform(2.6, 3.5)
        offset = rng.normal(0, 0.035, 6)
        wear_rate = rng.uniform(0.9, 1.1)
        for t in range(windows):
            hours = t / 60
            wear = min(hours / lifetime, 1.0)
            degradation = max((wear - 0.2) / 0.8, 0) ** wear_rate
            load = [0.15, 0.5, 0.85][(t // 20 + d) % 3]
            rpm = 3000 + 6000 * (1 - 0.35 * load) + rng.normal(0, 100)
            ambient = 23 + 3 * np.sin(d)
            # Variable load affects healthy amplitudes independently of wear.
            mu = np.array([0.7 + 2 * load + rpm / 20000, 3.0, 1 + 7 * load,
                           0.2 + 0.5 * load, 8 + 20 * load, 0.01 + 0.02 * load])
            effect = np.array([1.7, 4, 1, 0.65, 12, 0.08]) * degradation
            noise = rng.normal(0, [0.12, 0.18, 0.18, 0.06, 0.6, 0.008])
            values = mu + effect + offset * np.array([1, 1, 1, 0.2, 3, 0.03]) + noise
            quality = np.ones(3)
            fault_kind = "none"
            if rng.random() < 0.08:
                ch = int(rng.integers(3))
                quality[ch] = 0
                values[ch * 2:ch * 2 + 2] = np.nan
                fault_kind = ["vibration_missing", "current_missing", "temperature_missing"][ch]
            if rng.random() < 0.04 and fault_kind == "none":
                ch = int(rng.integers(3))
                quality[ch] = 0.35
                values[ch*2:ch*2+2] += rng.normal(0, [2, 2])
                fault_kind = "noisy_channel"
            if t % 83 == 80:
                quality[:] = 0
                values[:] = np.nan
                fault_kind = "all_missing"
            stage = int(np.digitize(wear, [0.35, 0.6, 0.85]))
            row = dict(zip(FEATURES, values))
            row.update(dict(zip(QUALITY, quality)))
            row.update(device_id=f"DRILL-{d+1:03d}", run_id="run-001", sequence=t,
                       timestamp=(pd.Timestamp("2026-01-01", tz="UTC") + pd.Timedelta(minutes=t)).isoformat(),
                       rpm=rpm, load=load, ambient_c=ambient, mode="drilling",
                       operating_hours=hours, stage=stage, hi_label=1 - wear,
                       rul_hours=max(lifetime - hours, 0), simulated_wear=wear,
                       sensor_fault=fault_kind, source="synthetic", failure_definition="simulated wear >= 1")
            rows.append(row)
    return pd.DataFrame(rows)


def grouped_split(data, seed=42):
    ids = np.array(sorted(data.device_id.unique()))
    if len(ids) < 10:
        raise ValueError("分组评估至少需要 10 台设备。")
    np.random.default_rng(seed).shuffle(ids)
    a, b = int(len(ids) * 0.7), int(len(ids) * 0.9)
    return {name: data[data.device_id.isin(group)].copy() for name, group in
            zip(["train", "validation", "test"], [ids[:a], ids[a:b], ids[b:]])}
