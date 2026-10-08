import json
import tempfile
import unittest
from pathlib import Path
import hashlib

import numpy as np
import torch
from torch.nn import functional as F

from predictive.bearing_model import (BearingClassifier, KernelLinearAttention,
                                     CLASS_LABELS, computation_profile,
                                     load_bearing_bundle, normalize_bearing,
                                     predict_bearing)
from predictive.public_data import preprocess_waveform
from scripts.train_bearing import fit_normalization, load_dataset


class BearingModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_linear_attention_matches_dense_positive_kernel(self):
        """Associative implementation must preserve the chosen normalized kernel."""
        torch.manual_seed(7)
        layer = KernelLinearAttention(dimension=12, heads=3)
        tokens = torch.randn(2, 9, 12)
        q, k, v = layer.qkv(tokens).reshape(2, 9, 3, 3, 4).permute(2, 0, 3, 1, 4)
        weights = (F.elu(q) + 1) @ (F.elu(k) + 1).transpose(-2, -1)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        expected = layer.project((weights @ v).transpose(1, 2).reshape(2, 9, 12))
        torch.testing.assert_close(layer(tokens), expected, atol=1e-6, rtol=1e-5)

    def test_all_modes_train_on_raw_window_shape_and_share_attention_budget(self):
        counts = {}
        for kind in ("cnn", "softmax", "linear"):
            model = BearingClassifier(kind)
            logits = model(torch.randn(4, 1, 1024))
            self.assertEqual(tuple(logits.shape), (4, 4))
            F.cross_entropy(logits, torch.tensor([0, 1, 2, 3])).backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                                for p in model.parameters()))
            counts[kind] = computation_profile(model)
        self.assertEqual(counts["softmax"]["parameter_count"], counts["linear"]["parameter_count"])
        self.assertEqual(counts["linear"]["token_count"], 64)
        self.assertLess(counts["linear"]["estimated_macs"], counts["softmax"]["estimated_macs"])
        self.assertLess(counts["cnn"]["parameter_count"], counts["linear"]["parameter_count"])

    def test_training_normalization_and_input_rejection(self):
        values = np.full(1024, 7, dtype=np.float32)
        normalized = normalize_bearing(values, {"mean": 3, "std": 2})
        self.assertEqual(tuple(normalized.shape), (1, 1, 1024))
        self.assertTrue(torch.equal(normalized, torch.full_like(normalized, 2)))
        for bad in (np.zeros(100), np.full(1024, np.nan), np.zeros((2, 2, 1024))):
            with self.assertRaises(ValueError):
                normalize_bearing(bad, {"mean": 0, "std": 1})
        with self.assertRaises(ValueError):
            normalize_bearing(values, {"mean": 0, "std": 0})
        with self.assertRaises(ValueError):
            BearingClassifier()(torch.zeros(1, 1024))

    def test_bundle_checksum_and_predictions_have_only_four_fault_classes(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            model = BearingClassifier("cnn").eval()
            torch.save(model.state_dict(), folder / "model.pt")
            manifest = {"task": "bearing_fault_classification", "class_labels": CLASS_LABELS,
                        "model_config": {"kind": "cnn", "dimension": 24, "heads": 4,
                                         "window_length": 1024},
                        "normalization": {"mean": 0, "std": 1},
                        "weights_sha256": hashlib.sha256((folder / "model.pt").read_bytes()).hexdigest()}
            (folder / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            loaded, saved = load_bearing_bundle(folder)
            result = predict_bearing(np.zeros(1024), loaded, saved, prepared=True)[0]
            self.assertEqual(set(result["scores"]), set(CLASS_LABELS))
            self.assertAlmostEqual(sum(result["scores"].values()), 1, places=6)
            self.assertNotIn("rul", result)
            (folder / "model.pt").write_bytes(b"corrupted")
            with self.assertRaises(ValueError):
                load_bearing_bundle(folder)

    def test_raw_and_prepared_inference_use_exactly_one_denoising_pass(self):
        rng = np.random.default_rng(19)
        raw = rng.normal(2, .2, 1024).astype(np.float32)
        prepared = preprocess_waveform(raw, sample_rate=12000, target_rate=12000)
        model = BearingClassifier("linear").eval()
        manifest = {"normalization": {"mean": 0, "std": .5}}
        from_raw = predict_bearing(raw, model, manifest, prepared=False)[0]
        from_prepared = predict_bearing(prepared, model, manifest, prepared=True)[0]
        np.testing.assert_allclose(list(from_raw["scores"].values()),
                                   list(from_prepared["scores"].values()), atol=1e-7)

    def test_record_leakage_rejected_and_standardization_fits_training_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            partitions = np.repeat(["train", "validation", "test"], 4)
            labels = np.tile(np.arange(4, dtype=np.int64), 3)
            records = np.array([f"{split}-{label}" for split, label in zip(partitions, labels)])
            waveform = np.tile(np.linspace(-1, 1, 1024, dtype=np.float32), (12, 1, 1))
            waveform[partitions != "train"] += 100
            path = folder / "dataset.npz"
            np.savez(path, X=waveform, y=labels, split=partitions, record_id=records)
            (folder / "dataset_manifest.json").write_text('{"source":"test fixture"}')
            X, _, split, protocol, _ = load_dataset(path)
            stats = fit_normalization(X, split)
            self.assertAlmostEqual(stats["mean"], 0, places=6)
            self.assertAlmostEqual(stats["std"], float(waveform[:4].std()), places=6)
            self.assertEqual(protocol["counts"]["test"]["records"], 4)
            records[4] = records[0]
            np.savez(path, X=waveform, y=labels, split=partitions, record_id=records)
            with self.assertRaisesRegex(ValueError, "record leakage"):
                load_dataset(path)


if __name__ == "__main__":
    unittest.main()
