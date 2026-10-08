"""Synthetic sensor data and a baseline anomaly detector."""
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

FEATURES = ["temperature_c", "vibration_mm_s", "current_a"]

def simulate_data(samples=300, seed=42):
    rng = np.random.default_rng(seed)
    data = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=samples, freq="min"),
        "temperature_c": rng.normal(45, 2, samples),
        "vibration_mm_s": rng.normal(2, 0.2, samples),
        "current_a": rng.normal(5, 0.3, samples),
    })
    fault = np.arange(samples) >= int(samples * 0.85)
    data.loc[fault, FEATURES] += [20, 3, 2]
    data["simulated_fault"] = fault
    return data

def analyze(data, baseline=None):
    missing = set(FEATURES) - set(data.columns)
    if missing:
        raise ValueError("缺少数据列：" + ", ".join(sorted(missing)))
    values = data[FEATURES].apply(pd.to_numeric, errors="coerce")
    if len(values) < 20 or not np.isfinite(values.to_numpy()).all():
        raise ValueError("至少需要 20 行数据，且传感器值必须是有限数字。")
    # A separate healthy reference avoids training on the simulated faults.
    reference = simulate_data(1000, seed=7).iloc[:800][FEATURES] if baseline is None else baseline[FEATURES]
    model = IsolationForest(n_estimators=100, contamination=0.02, random_state=42)
    model.fit(reference)
    result = data.copy()
    result["anomaly"] = model.predict(values) == -1
    result["anomaly_score"] = -model.decision_function(values)
    result["maintenance_advice"] = np.where(result["anomaly"], "建议检查工具、传感器及运行负载", "继续监测")
    return result
