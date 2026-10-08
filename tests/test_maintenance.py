import unittest
import numpy as np
from maintenance import analyze, simulate_data

class MaintenanceTests(unittest.TestCase):
    def test_detects_simulated_faults(self):
        result = analyze(simulate_data())
        self.assertGreater(result.loc[result.simulated_fault, "anomaly"].mean(), 0.9)
        self.assertLess(result.loc[~result.simulated_fault, "anomaly"].mean(), 0.1)
        self.assertTrue(result.anomaly_score.notna().all())

    def test_rejects_invalid_sensor_data(self):
        for value in [np.nan, np.inf, "invalid"]:
            data = simulate_data().astype({"current_a": object})
            data.loc[0, "current_a"] = value
            with self.assertRaises(ValueError):
                analyze(data)

    def test_requires_columns_and_sample_count(self):
        with self.assertRaises(ValueError):
            analyze(simulate_data().drop(columns="current_a"))
        with self.assertRaises(ValueError):
            analyze(simulate_data(10))

if __name__ == "__main__":
    unittest.main()
