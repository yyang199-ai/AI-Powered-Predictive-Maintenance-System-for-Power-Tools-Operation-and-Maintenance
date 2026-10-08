import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from predictive.lab import (create_lab_run, extract_lab_features,
                            export_lab_run, haar_soft_denoise)


class VirtualLabTests(unittest.TestCase):
    def test_three_tools_have_distinct_clocks_and_explicit_simulated_source(self):
        for tool in ("drill", "grinder", "impact_wrench"):
            with self.subTest(tool=tool):
                run = create_lab_run(tool=tool, duration_seconds=1, seed=19)
                self.assertEqual(run.metadata["tool"], tool)
                self.assertTrue(run.metadata["is_simulated"])
                self.assertEqual(run.metadata["data_source"], "synthetic_raw_waveform_lab")
                self.assertEqual(run.metadata["units"]["vibration"], "m/s^2")
                self.assertEqual(run.metadata["units"]["current"], "A")
                self.assertEqual(run.metadata["units"]["temperature"], "degC")
                for name in ("vibration", "current", "temperature"):
                    values = getattr(run, name)
                    self.assertEqual(len(values), int(run.sampling_rates[name]))
                    self.assertTrue(np.isfinite(values).all())
                    self.assertEqual(run.times[name][0], 0)
                    self.assertLess(run.times[name][-1], 1)
                self.assertEqual(len(run.temperature), 10)
                self.assertEqual(len(run.vibration), 25600)
                self.assertEqual(run.metadata["model_assumptions"]["operating_impacts_in_healthy_data"],
                                 tool == "impact_wrench")

    def test_damage_and_load_change_physically_intended_quantities(self):
        damage_features = [extract_lab_features(create_lab_run(damage=level, seed=41))
                           for level in ("normal", "mild", "moderate", "severe")]
        for key in ("vibration_raw_rms", "current_mean", "temperature_slope"):
            values = [features[key] for features in damage_features]
            self.assertTrue(np.all(np.diff(values) > 0), (key, values))
        low = extract_lab_features(create_lab_run(load=.1, seed=41))
        high = extract_lab_features(create_lab_run(load=.9, seed=41))
        self.assertGreater(high["current_mean"], low["current_mean"])
        self.assertGreater(high["temperature_slope"], low["temperature_slope"])
        self.assertGreater(high["vibration_raw_rms"], low["vibration_raw_rms"])

    def test_fft_detects_shaft_frequency_and_current_harmonics(self):
        run = create_lab_run(rpm=1800, duration_seconds=2, seed=17)
        features = extract_lab_features(run)
        self.assertAlmostEqual(features["vibration_peak_hz"], 30, delta=.5)
        self.assertAlmostEqual(features["vibration_fft_resolution_hz"], .5)
        self.assertAlmostEqual(features["current_mean"], 4.5, delta=.01)
        self.assertAlmostEqual(features["current_harmonic_1_amplitude"], .305, delta=.025)
        self.assertGreater(features["current_harmonic_2_amplitude"], .04)
        self.assertGreater(features["current_harmonic_3_amplitude"], .02)
        self.assertTrue(all(np.isfinite(value) for value in features.values()))
        self.assertNotIn("damage", features)

    def test_haar_reduces_noise_and_preserves_constant_and_shape(self):
        rng = np.random.default_rng(11)
        times = np.arange(4096) / 4096
        clean = np.sin(2 * np.pi * 8 * times)
        noisy = clean + rng.normal(0, .22, len(times))
        denoised = haar_soft_denoise(noisy)
        self.assertLess(np.mean((denoised - clean) ** 2), np.mean((noisy - clean) ** 2))
        np.testing.assert_allclose(haar_soft_denoise(np.full(1001, 4.5)), 4.5, atol=1e-12)
        self.assertEqual(len(haar_soft_denoise(noisy[:4001])), 4001)

    def test_reproducible_export_preserves_raw_waveforms_and_metadata(self):
        run = create_lab_run(tool="grinder", damage="moderate", duration_seconds=1,
                             sampling_rates={"current": 4096, "temperature": 20}, seed=15)
        duplicate = create_lab_run(tool="grinder", damage="moderate", duration_seconds=1,
                                   sampling_rates={"current": 4096, "temperature": 20}, seed=15)
        np.testing.assert_array_equal(run.vibration, duplicate.vibration)
        np.testing.assert_array_equal(run.current, duplicate.current)
        np.testing.assert_array_equal(run.temperature, duplicate.temperature)
        with tempfile.TemporaryDirectory() as directory:
            paths = export_lab_run(run, Path(directory) / "run")
            self.assertEqual(set(paths), {"waveforms", "metadata", "features",
                                         "vibration", "current", "temperature"})
            with np.load(paths["waveforms"]) as arrays:
                np.testing.assert_array_equal(arrays["vibration"], run.vibration)
                np.testing.assert_array_equal(arrays["temperature_time_seconds"], run.times["temperature"])
            current_csv = np.loadtxt(paths["current"], delimiter=",", skiprows=1)
            np.testing.assert_allclose(current_csv[:, 1], run.current, atol=1e-9)
            metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
            self.assertTrue(metadata["is_simulated"])
            self.assertFalse(metadata["feature_processing"]["uses_damage_label_as_feature"])
            self.assertEqual(metadata["sampling_rates_hz"]["temperature"], 20)
            features = json.loads(paths["features"].read_text(encoding="utf-8"))
            self.assertEqual(features, extract_lab_features(run))

    def test_invalid_configuration_and_damaged_time_axis_are_rejected(self):
        for arguments in ({"tool": "saw"}, {"damage": "broken"}, {"load": -1},
                          {"load": np.nan}, {"rpm": 0}, {"rpm": 1e6},
                          {"duration_seconds": 0}, {"seed": -1}, {"seed": True},
                          {"sampling_rates": {"vibration": 500}},
                          {"sampling_rates": {"unknown": 10}},
                          {"sampling_rates": {"temperature": 2}, "duration_seconds": .5}):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                create_lab_run(**arguments)
        run = create_lab_run(duration_seconds=1)
        run.times["vibration"][100:] += .01
        with self.assertRaises(ValueError):
            extract_lab_features(run)
        with self.assertRaises(ValueError):
            haar_soft_denoise([1, 2, np.nan])


if __name__ == "__main__":
    unittest.main()
