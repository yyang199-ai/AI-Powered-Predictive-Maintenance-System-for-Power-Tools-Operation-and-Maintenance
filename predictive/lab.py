"""Independent raw-waveform *simulation* bench; not measured power-tool data.

The offline Haar denoiser and FFT features do not alter the original six-feature
fleet simulator or its trained model. Damage labels are generator settings.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Mapping

import numpy as np
from scipy.signal import fftconvolve, periodogram


@dataclass
class LabRun:
    vibration: np.ndarray
    current: np.ndarray
    temperature: np.ndarray
    sampling_rates: dict[str, float]
    times: dict[str, np.ndarray]
    metadata: dict


_TOOLS = {
    "drill": dict(name="电钻", rpm=1800.0, max_rpm=3600.0,
                  vibration=1.2, resonance=2500.0, idle_current=2.0,
                  load_current=5.0, thermal_tau=45.0),
    "grinder": dict(name="角磨机", rpm=11000.0, max_rpm=16000.0,
                    vibration=2.0, resonance=4200.0, idle_current=4.0,
                    load_current=10.0, thermal_tau=60.0),
    "impact_wrench": dict(name="冲击扳手", rpm=2200.0, max_rpm=4500.0,
                          vibration=1.5, resonance=3000.0,
                          idle_current=3.0, load_current=12.0,
                          thermal_tau=50.0),
}
_DAMAGE = {"normal": 0.0, "mild": 0.3, "moderate": 0.7, "severe": 1.2}
_CHANNELS = ("vibration", "current", "temperature")
_DEFAULT_RATES = {"vibration": 25600.0, "current": 25600.0,
                  "temperature": 10.0}


def _number(value, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite number, not a boolean")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _ringing(times, sample_rate, frequency, resonance, amplitude, decay=700.0):
    """Regular simulated contacts excite a damped resonant response."""
    impulses = np.zeros(len(times), dtype=float)
    pulse_times = np.arange(0.0, len(times) / sample_rate, 1.0 / frequency)
    indices = np.floor(pulse_times * sample_rate).astype(int)
    indices = indices[indices < len(impulses)]
    impulses[indices] = amplitude
    kernel_time = np.arange(max(8, int(0.025 * sample_rate))) / sample_rate
    kernel = np.exp(-decay * kernel_time) * np.cos(2 * np.pi * resonance * kernel_time)
    return fftconvolve(impulses, kernel, mode="full")[:len(times)]


def create_lab_run(tool="drill", damage="normal", load=.5, rpm=None,
                   duration_seconds=5, seed=42, *, sampling_rates=None,
                   ambient_temperature_c=25.0) -> LabRun:
    """Generate a stationary operating condition with synthetic damage effects.

    ``rpm`` denotes the modeled rotating shaft, not a verified commercial tool's
    chuck/rotor speed. Additional keyword options select sampling rates and the
    ambient temperature. Three channel clocks share t=0; temperature is slower.
    """
    if tool not in _TOOLS:
        raise ValueError(f"tool must be one of {tuple(_TOOLS)}")
    if damage not in _DAMAGE:
        raise ValueError(f"damage must be one of {tuple(_DAMAGE)}")
    cfg = _TOOLS[tool]
    load = _number(load, "load")
    duration = _number(duration_seconds, "duration_seconds")
    rpm = cfg["rpm"] if rpm is None else _number(rpm, "rpm")
    ambient = _number(ambient_temperature_c, "ambient_temperature_c")
    if not 0 <= load <= 1:
        raise ValueError("load must be in [0, 1]")
    if not .5 <= duration <= 120:
        raise ValueError("duration_seconds must be in [0.5, 120]")
    if not 60 <= rpm <= cfg["max_rpm"]:
        raise ValueError(f"rpm must be in [60, {cfg['max_rpm']}]")
    if not -20 <= ambient <= 60:
        raise ValueError("ambient_temperature_c must be in [-20, 60]")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    rates = dict(_DEFAULT_RATES)
    if sampling_rates is not None:
        if not isinstance(sampling_rates, Mapping) or set(sampling_rates) - set(_CHANNELS):
            raise ValueError("sampling_rates must map vibration/current/temperature")
        rates.update(sampling_rates)
    rates = {name: _number(rates[name], f"sampling_rates[{name}]") for name in _CHANNELS}
    shaft_hz = rpm / 60.0
    # The synthetic resonant carrier and three modeled current harmonics must
    # lie comfortably below Nyquist. This validates the simulator, not a DAQ.
    if not 2.5 * cfg["resonance"] <= rates["vibration"] <= 100000:
        raise ValueError("vibration sampling rate must cover the modeled resonance with margin")
    if not max(128.0, 8.0 * shaft_hz) <= rates["current"] <= 100000:
        raise ValueError("current sampling rate must cover the modeled third harmonic with margin")
    if not 2 <= rates["temperature"] <= 100:
        raise ValueError("temperature sampling rate must be in [2, 100] Hz")
    counts = {name: int(round(duration * rates[name])) for name in _CHANNELS}
    if counts["temperature"] < 4 or min(counts["vibration"], counts["current"]) < 64:
        raise ValueError("at least four temperature and 64 waveform samples are required")
    if sum(counts.values()) > 8_000_000:
        raise ValueError("requested run exceeds the eight-million-sample memory limit")
    times = {name: np.arange(counts[name], dtype=float) / rates[name] for name in _CHANNELS}
    rng = np.random.default_rng(seed)
    severity = _DAMAGE[damage]
    tv = times["vibration"]
    base_amp = cfg["vibration"] * (.5 + load)
    vibration = base_amp * (np.sin(2 * np.pi * shaft_hz * tv)
                            + .25 * np.sin(2 * np.pi * 2 * shaft_hz * tv))
    bearing_hz = 4.6 * shaft_hz  # generator assumption, not an identified BPFO
    vibration += _ringing(tv, rates["vibration"], bearing_hz,
                          cfg["resonance"], 5.0 * severity * (1 + .3 * load))
    if tool == "impact_wrench":
        # Intended operating impacts also occur in healthy tools.
        vibration += _ringing(tv, rates["vibration"], 8 + 20 * load,
                              1500.0, 2.0 * load, decay=450.0)
    vibration += rng.normal(0, .12 * cfg["vibration"], len(tv))

    ti = times["current"]
    current_mean = cfg["idle_current"] + cfg["load_current"] * load + 1.5 * severity
    harmonic_amplitudes = (.18 + .25 * load + .18 * severity,
                           .06 + .08 * load + .20 * severity,
                           .03 + .05 * load + .15 * severity)
    current = np.full(len(ti), current_mean, dtype=float)
    for order, amplitude in enumerate(harmonic_amplitudes, start=1):
        current += amplitude * np.sin(2 * np.pi * order * shaft_hz * ti + order * .2)
    current += rng.normal(0, .035, len(ti))

    tt = times["temperature"]
    rise_equilibrium = 4.0 + 18.0 * load + 18.0 * severity
    temperature = ambient + rise_equilibrium * (1 - np.exp(-tt / cfg["thermal_tau"]))
    temperature += rng.normal(0, .015, len(tt))
    metadata = {
        "schema_version": 1,
        "data_source": "synthetic_raw_waveform_lab",
        "is_simulated": True,
        "tool": tool, "tool_name": cfg["name"], "damage": damage,
        "damage_label_source": "generator_setting_not_independent_measurement",
        "load": load, "rpm": rpm, "rpm_reference": "modeled_rotating_shaft",
        "duration_seconds": duration, "seed": int(seed),
        "ambient_temperature_c": ambient,
        "time_reference": "seconds_since_shared_virtual_start",
        "units": {"vibration": "m/s^2", "current": "A", "temperature": "degC"},
        "samples": counts,
        "model_assumptions": {
            "bearing_excitation_hz": bearing_hz,
            "bearing_ratio": 4.6,
            "vibration_resonance_hz": cfg["resonance"],
            "current_harmonic_reference_hz": shaft_hz,
            "current_harmonics": "shaft-frequency modulation, not mains/PWM harmonics",
            "thermal_time_constant_seconds": cfg["thermal_tau"],
            "thermal_equilibrium_rise_c": rise_equilibrium,
            "operating_impacts_in_healthy_data": tool == "impact_wrench",
            "stationary_load_and_rpm": True,
        },
        "limitations": [
            "Parameters are illustrative assumptions, not calibrated physical tool models.",
            "No real bearing geometry, PWM driver, gearing or sensor transfer function is identified.",
            "Damage classes do not establish real wear stages or remaining useful life.",
            "This raw-waveform bench is separate from public bearing data and the original fleet model.",
        ],
    }
    return LabRun(vibration, current, temperature, rates, times, metadata)


def haar_soft_denoise(signal, levels=6) -> np.ndarray:
    """Offline orthonormal Haar shrinkage with MAD universal soft threshold.

    Reflect-pad to the next power of two, use finest-scale details for sigma,
    apply sigma*sqrt(2*log(N)) to each detail level, and crop reconstruction.
    This is batch processing, not a causal streaming filter.
    """
    x = np.asarray(signal, dtype=float)
    if x.ndim != 1 or len(x) < 8 or not np.isfinite(x).all():
        raise ValueError("signal must contain at least eight finite one-dimensional samples")
    if isinstance(levels, bool) or not isinstance(levels, (int, np.integer)) or levels < 1:
        raise ValueError("levels must be a positive integer")
    n = len(x)
    padded_size = 1 << (n - 1).bit_length()
    approximation = np.pad(x, (0, padded_size - n), mode="reflect")
    details = []
    for _ in range(min(int(levels), padded_size.bit_length() - 1)):
        even, odd = approximation[::2], approximation[1::2]
        details.append((even - odd) / np.sqrt(2.0))
        approximation = (even + odd) / np.sqrt(2.0)
    fine = details[0]
    sigma = float(np.median(np.abs(fine - np.median(fine))) / .6744897501960817)
    threshold = sigma * np.sqrt(2 * np.log(n))
    for detail in reversed(details):
        shrunk = np.sign(detail) * np.maximum(np.abs(detail) - threshold, 0.0)
        rebuilt = np.empty(len(detail) * 2, dtype=float)
        rebuilt[::2] = (approximation + shrunk) / np.sqrt(2.0)
        rebuilt[1::2] = (approximation - shrunk) / np.sqrt(2.0)
        approximation = rebuilt
    return approximation[:n]


def _validate_run(run: LabRun):
    if not isinstance(run, LabRun):
        raise ValueError("run must be a LabRun")
    for name in _CHANNELS:
        values = np.asarray(getattr(run, name), dtype=float)
        try:
            rate = _number(run.sampling_rates[name], name + " sampling rate")
            times = np.asarray(run.times[name], dtype=float)
        except (KeyError, TypeError) as exc:
            raise ValueError("run is missing sampling rates or time axes") from exc
        minimum = 4 if name == "temperature" else 8
        if rate <= 0 or values.ndim != 1 or len(values) < minimum or not np.isfinite(values).all():
            raise ValueError(f"invalid {name} data or sampling rate")
        if times.shape != values.shape or not np.isfinite(times).all():
            raise ValueError(f"invalid {name} time axis")
        if not np.allclose(np.diff(times), 1 / rate, rtol=1e-6, atol=1e-10):
            raise ValueError(f"{name} time axis must be uniform at its declared rate")
    if not isinstance(run.metadata, dict):
        raise ValueError("metadata must be a dictionary")
    rpm = _number(run.metadata.get("rpm"), "metadata rpm")
    if rpm <= 0:
        raise ValueError("metadata rpm must be positive")


def _fft(signal, sample_rate):
    centered = signal - np.mean(signal)
    window = np.hanning(len(centered))
    amplitude = 2 * np.abs(np.fft.rfft(centered * window)) / window.sum()
    amplitude[0] *= .5
    if len(centered) % 2 == 0:
        amplitude[-1] *= .5
    frequency = np.fft.rfftfreq(len(centered), 1 / sample_rate)
    _, density = periodogram(centered, sample_rate, window=window,
                            detrend=False, scaling="density")
    return frequency, amplitude, density


def extract_lab_features(run: LabRun) -> dict[str, float]:
    """Extract window features after offline Haar denoising, preserving raw data.

    Temperature trend is in degC/second. Harmonic amplitudes use Hann FFT bins
    near modeled shaft orders; the distortion ratio is descriptive, not a
    calibrated electrical THD measurement or a fault probability.
    """
    _validate_run(run)
    vibration = haar_soft_denoise(run.vibration)
    current = haar_soft_denoise(run.current)
    centered = vibration - vibration.mean()
    rms = float(np.sqrt(np.mean(centered ** 2)))
    frequency, amplitude, density = _fft(vibration, run.sampling_rates["vibration"])
    df = frequency[1] - frequency[0]
    peak = 1 + int(np.argmax(amplitude[1:]))
    total_power = float(np.sum(density[1:]) * df)
    features = {
        "vibration_raw_rms": float(np.sqrt(np.mean(run.vibration ** 2))),
        "vibration_rms": rms,
        "vibration_kurtosis": float(np.mean(centered ** 4) / max(rms ** 4, 1e-20)),
        "vibration_peak_hz": float(frequency[peak]),
        "vibration_spectral_centroid_hz": float(np.sum(frequency * density) / max(np.sum(density), 1e-20)),
        "vibration_fft_resolution_hz": float(df),
        "vibration_low_band_power": float(np.sum(density[(frequency > 0) & (frequency < 1000)]) * df),
        "vibration_high_band_power": float(np.sum(density[frequency >= 1000]) * df),
        "vibration_total_spectral_power": total_power,
        "current_mean": float(current.mean()),
        "current_rms": float(np.sqrt(np.mean(current ** 2))),
        "current_ripple": float(np.std(current)),
        "temperature_rise": float(run.temperature[-1] - run.temperature[0]),
        "temperature_mean_c": float(np.mean(run.temperature)),
        "temperature_slope": float(np.polyfit(run.times["temperature"] - run.times["temperature"][0],
                                             run.temperature, 1)[0]),
        "shaft_frequency_hz": float(run.metadata["rpm"] / 60),
    }
    fi, ai, _ = _fft(current, run.sampling_rates["current"])
    harmonic_amplitudes = []
    for order in range(1, 4):
        target = order * features["shaft_frequency_hz"]
        if target >= fi[-1]:
            raise ValueError("current sampling rate does not cover modeled harmonic frequencies")
        nearest = int(np.argmin(np.abs(fi - target)))
        candidates = ai[max(1, nearest - 1):min(len(ai), nearest + 2)]
        value = float(candidates.max())
        harmonic_amplitudes.append(value)
        features[f"current_harmonic_{order}_amplitude"] = value
    features["current_harmonic_distortion_ratio"] = float(
        np.linalg.norm(harmonic_amplitudes[1:]) / max(harmonic_amplitudes[0], 1e-12))
    return features


def export_lab_run(run: LabRun, directory) -> dict[str, Path]:
    """Write lossless NPZ, one CSV per channel, metadata and feature JSON."""
    _validate_run(run)
    features = extract_lab_features(run)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    paths = {"waveforms": directory / "waveforms.npz",
             "metadata": directory / "metadata.json",
             "features": directory / "features.json"}
    arrays = {name: getattr(run, name) for name in _CHANNELS}
    arrays.update({name + "_time_seconds": run.times[name] for name in _CHANNELS})
    np.savez_compressed(paths["waveforms"], **arrays)
    for name in _CHANNELS:
        paths[name] = directory / (name + ".csv")
        unit = run.metadata.get("units", {}).get(name, "unknown")
        np.savetxt(paths[name], np.column_stack((run.times[name], getattr(run, name))),
                   delimiter=",", header=f"time_seconds,{name}_{unit}", comments="", fmt="%.12g")
    metadata = dict(run.metadata)
    metadata["sampling_rates_hz"] = run.sampling_rates
    metadata["feature_processing"] = {
        "denoising": "offline orthonormal Haar, up to six levels, MAD universal soft threshold",
        "spectrum": "demean, Hann-window rFFT amplitude and power spectral density",
        "temperature_slope_unit": "degC/second",
        "harmonic_reference": "modeled shaft frequency; not mains harmonics",
        "uses_damage_label_as_feature": False,
    }
    paths["metadata"].write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    paths["features"].write_text(json.dumps(features, ensure_ascii=False, indent=2), encoding="utf-8")
    return paths
