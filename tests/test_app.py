import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from predictive.store import Store


class AppTests(unittest.TestCase):
    def test_public_default_and_readable_pages(self):
        app = AppTest.from_file("app.py").run(timeout=60)
        self.assertFalse(app.exception)
        self.assertEqual(app.sidebar.radio[0].value, "公开轴承预验证")
        self.assertTrue(any("历史振动数据" in item.value for item in app.info))
        if not Path("artifacts/bearing/data/dataset.npz").exists():
            self.assertTrue(any("公开数据尚未准备完成" in item.value for item in app.error))
            self.assertFalse(any(button.label == "生成虚拟实验记录" for button in app.button))
        app.sidebar.radio[0].set_value("模型与使用说明").run(timeout=60)
        self.assertFalse(app.exception)
        self.assertTrue(any("任务书的准确率" in item.value for item in app.markdown))

    def test_virtual_lab_keeps_simulation_label_and_units(self):
        app = AppTest.from_file("app.py").run(timeout=60)
        app.sidebar.radio[0].set_value("虚拟电动工具实验台").run(timeout=60)
        self.assertFalse(app.exception)
        create = next(button for button in app.button if button.label == "生成虚拟实验记录")
        create.click().run(timeout=60)
        self.assertFalse(app.exception)
        self.assertTrue(any(metric.value == "已生成" for metric in app.metric))
        run = app.session_state["lab_run"]
        self.assertTrue(run.metadata["is_simulated"])
        self.assertEqual(set(run.sampling_rates), {"vibration", "current", "temperature"})
        self.assertEqual(len(app.tabs), 4)
        self.assertTrue(any("不能直接输入" in item.value for item in app.warning))

    def test_public_inference_and_manual_replay(self):
        if not all(Path(path).exists() for path in ["artifacts/bearing/data/dataset.npz", "artifacts/bearing/manifest.json"]):
            self.skipTest("Public dataset and trained model are not yet prepared.")
        app = AppTest.from_file("app.py").run(timeout=60)
        self.assertFalse(app.exception)
        next(button for button in app.button if button.label == "开始分析当前窗口").click().run(timeout=60)
        self.assertFalse(app.exception)
        prediction = app.session_state["bearing_result"]
        self.assertIn(prediction["class_index"], range(4))
        self.assertAlmostEqual(sum(prediction["scores"].values()), 1, places=5)
        self.assertTrue(any(metric.value == "已完成" for metric in app.metric))
        first = app.session_state["bearing_result_identity"]
        next(button for button in app.button if button.label == "下一窗口并分析").click().run(timeout=60)
        self.assertFalse(app.exception)
        self.assertNotEqual(first, app.session_state["bearing_result_identity"])
        next(button for button in app.button if button.label == "重置").click().run(timeout=60)
        self.assertIsNone(app.session_state["bearing_result"])
        self.assertEqual(app.session_state["bearing_window_number"], 1)
        self.assertFalse(app.exception)

    def test_legacy_simulation_persistence_and_upload_prompt(self):
        with tempfile.TemporaryDirectory() as folder:
            store = Store(Path(folder) / "ui.sqlite3")
            try:
                with patch("predictive.store.Store", return_value=store):
                    app = AppTest.from_file("legacy_app.py").run(timeout=60)
                    self.assertFalse(app.exception)
                    self.assertEqual(app.metric[0].value, "200")
                    next(button for button in app.button if button.label == "绑定仿真设备并保存分析记录").click().run(timeout=60)
                    self.assertFalse(app.exception)
                    self.assertEqual(len(store.devices()), 4)
                    self.assertTrue(store.events())
                    app.sidebar.radio[0].set_value("上传 CSV").run()
                    self.assertFalse(app.exception)
                    self.assertTrue(app.info)
            finally:
                store.db.close()
