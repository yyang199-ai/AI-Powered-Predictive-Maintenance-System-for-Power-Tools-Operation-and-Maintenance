import unittest
from pathlib import Path

import numpy as np

from predictive.public_data import (assign_record_splits, cwru_records, denoise_wavelet,
                                    extract_vibration_features, preprocess_waveform, resample_waveform)


class PublicPreparationTests(unittest.TestCase):
    def test_record_split_is_complete_disjoint_and_covers_classes(self):
        records = cwru_records()
        split = assign_record_splits(records)
        self.assertEqual(len(records), 40)
        self.assertEqual([sum(s == name for s in split.values()) for name in ("train", "validation", "test")], [28, 8, 4])
        for name in ("train", "validation", "test"):
            self.assertEqual({r["fault_label"] for r in records if split[r["record_id"]] == name}, {0, 1, 2, 3})
        self.assertTrue(all(r["sample_rate"] == 12000 for r in records))
        self.assertTrue(all(r["original_sample_rate"] == 48000 for r in records if r["fault_label"] == 0))
        self.assertTrue(all("current" not in r and "rul" not in r for r in records))

    def test_resampling_removes_out_of_band_alias(self):
        time = np.arange(48000) / 48000
        values = np.sin(2 * np.pi * 1000 * time) + np.sin(2 * np.pi * 10000 * time)
        output = resample_waveform(values, 48000)
        self.assertEqual(len(output), 12000)
        spectrum = np.abs(np.fft.rfft(output[200:-200]))
        peak = np.fft.rfftfreq(len(output[200:-200]), 1 / 12000)[np.argmax(spectrum)]
        self.assertAlmostEqual(peak, 1000, delta=2)
        self.assertLess(np.mean((output[200:-200] - np.sin(2 * np.pi * 1000 * np.arange(200, 11800) / 12000)) ** 2), .001)

    def test_wavelet_preserves_length_and_reduces_noise(self):
        clean = np.sin(2 * np.pi * 5 * np.arange(1024) / 1024)
        noisy = clean + np.random.default_rng(4).normal(0, .25, len(clean))
        output = denoise_wavelet(noisy)
        self.assertEqual(len(output), len(clean))
        self.assertLess(np.mean((output - clean) ** 2), np.mean((noisy - clean) ** 2))
        self.assertEqual(len(denoise_wavelet(noisy[:1019])), 1019)
        self.assertTrue(np.isfinite(extract_vibration_features(np.zeros(1024))["kurtosis"]))

    def test_preprocessing_is_window_local_and_keeps_amplitude(self):
        waveform = np.sin(2 * np.pi * 100 * np.arange(1024) / 12000)
        first = preprocess_waveform(waveform + 8)
        scaled = preprocess_waveform(waveform * 3)
        np.testing.assert_allclose(scaled, first * 3, atol=1e-6)
        self.assertAlmostEqual(float(first.mean()), 0, places=6)
        with self.assertRaises(ValueError):
            preprocess_waveform(np.full(1024, np.nan))

    def test_feature_frequency_uses_resampled_rate(self):
        values = np.sin(2 * np.pi * 375 * np.arange(4096) / 48000)
        features = extract_vibration_features(values, sample_rate=48000)
        self.assertAlmostEqual(features["peak_frequency_hz"], 375, delta=12)

    @unittest.skipUnless(Path("artifacts/bearing/data/dataset.npz").exists(), "prepared dataset not installed")
    def test_installed_dataset_has_no_record_cross_partition(self):
        with np.load("artifacts/bearing/data/dataset.npz", allow_pickle=False) as data:
            self.assertEqual(data["X"].shape[1:], (1, 1024))
            self.assertGreaterEqual(len(data["y"]), 5000)
            self.assertTrue(np.isfinite(data["X"]).all())
            for record_id in np.unique(data["record_id"]):
                indices = data["record_id"] == record_id
                self.assertEqual(len(np.unique(data["split"][indices])), 1)
                self.assertEqual(len(np.unique(data["y"][indices])), 1)
            for split in ("train", "validation", "test"):
                self.assertEqual(set(data["y"][data["split"] == split]), {0, 1, 2, 3})


if __name__ == "__main__":
    unittest.main()
