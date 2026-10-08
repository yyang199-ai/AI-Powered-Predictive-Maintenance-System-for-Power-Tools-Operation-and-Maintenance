from datetime import timedelta
from pathlib import Path
import tempfile
import unittest

import numpy as np

from predictive.ims_data import (IMS_RUN2_COUNT, IMS_RUN2_START, check_complete_run2,
                                 ims_vibration_features, load_ims_prepared,
                                 prepare_ims_run2, read_ims_snapshot)


class IMSDataTests(unittest.TestCase):
    def test_complete_timeline_cannot_be_claimed_from_endpoints(self):
        paths = [Path((IMS_RUN2_START + timedelta(minutes=10 * i)).strftime("%Y.%m.%d.%H.%M.%S"))
                 for i in range(IMS_RUN2_COUNT)]
        self.assertTrue(check_complete_run2(paths))
        self.assertFalse(check_complete_run2([paths[0], paths[-1]]))
        self.assertFalse(check_complete_run2(paths[:10] + paths[11:]))

    def test_partial_import_never_fabricates_lifetime_labels(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            raw = folder / "raw"
            raw.mkdir()
            t = np.arange(20480) / 20000
            values = np.sin(2 * np.pi * 200 * t)
            snapshot = raw / "2004.02.12.10.32.39"
            np.savetxt(snapshot, np.repeat(values[:, None], 4, axis=1))
            with self.assertRaisesRegex(ValueError, "incomplete"):
                prepare_ims_run2(raw, folder / "complete")
            manifest = prepare_ims_run2(raw, folder / "prepared", require_complete=False)
            frame, _ = load_ims_prepared(folder / "prepared")
            self.assertEqual(manifest["download_status"], "partial_import")
            self.assertEqual(manifest["unique_failure_events"], 0)
            self.assertTrue(frame["endpoint_proxy_rul_hours"].isna().all())
            self.assertTrue(frame["is_censored"].all())
            self.assertFalse(frame["event_observed"].any())
            self.assertFalse(manifest["labels"]["trained_rul_predictor"])
            (folder / "prepared/features.csv").write_text("corrupt")
            with self.assertRaisesRegex(ValueError, "checksum"):
                load_ims_prepared(folder / "prepared")

    def test_snapshot_channel_shape_and_nonfinite_values_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "2004.02.12.10.32.39"
            np.savetxt(path, np.ones((20480, 3)))
            with self.assertRaisesRegex(ValueError, "four channels"):
                read_ims_snapshot(path)
            values = np.ones((20480, 4))
            values[0, 0] = np.nan
            np.savetxt(path, values)
            with self.assertRaisesRegex(ValueError, "finite"):
                read_ims_snapshot(path)

    def test_frequency_features_use_actual_20khz_clock(self):
        t = np.arange(20480) / 20000
        result = ims_vibration_features(np.sin(2 * np.pi * 200 * t))
        self.assertGreater(result["band_low_fraction"], .98)
        self.assertLess(abs(result["spectral_peak_hz"] - 200), 2)
        self.assertTrue(np.isfinite(list(result.values())).all())

    def test_shipped_complete_run_preserves_censoring_and_endpoint_scope(self):
        prepared = Path(__file__).resolve().parents[1] / "data/public/NASA_IMS/prepared"
        frame, manifest = load_ims_prepared(prepared)
        self.assertTrue(manifest["complete_timeline_verified"])
        self.assertEqual(len(frame), 4 * IMS_RUN2_COUNT)
        self.assertTrue((frame.groupby("bearing_id").size() == IMS_RUN2_COUNT).all())
        self.assertEqual(set(frame.loc[frame.endpoint_proxy_rul_hours.notna(), "bearing_id"]),
                         {"bearing_1"})
        other_bearings = frame[frame.bearing_id != "bearing_1"]
        self.assertTrue(other_bearings.endpoint_proxy_rul_hours.isna().all())
        self.assertTrue(other_bearings.is_censored.all())
        bearing1 = frame[frame.bearing_id == "bearing_1"]
        self.assertEqual(bearing1.endpoint_proxy_rul_hours.iloc[-1], 0.0)
        self.assertTrue((np.diff(bearing1.endpoint_proxy_rul_hours) < 0).all())
        self.assertFalse(manifest["labels"]["trained_rul_predictor"])


if __name__ == "__main__":
    unittest.main()
