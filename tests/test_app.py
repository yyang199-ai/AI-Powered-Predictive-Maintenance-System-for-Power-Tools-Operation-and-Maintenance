import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
from streamlit.testing.v1 import AppTest
from predictive.store import Store

class AppTests(unittest.TestCase):
    def test_simulation_persistence_and_upload_prompt(self):
        with tempfile.TemporaryDirectory() as folder:
            store=Store(Path(folder)/'ui.sqlite3')
            with patch('predictive.store.Store',return_value=store):
                app=AppTest.from_file('app.py').run(timeout=60)
                self.assertFalse(app.exception)
                self.assertEqual(app.metric[0].value,'200')
                self.assertEqual(len(app.tabs),3)
                save=next(b for b in app.button if b.label=='绑定仿真设备并保存分析记录')
                save.click().run(timeout=60)
                self.assertFalse(app.exception)
                self.assertEqual(len(store.devices()),4)
                self.assertTrue(store.events())
                app.sidebar.radio[0].set_value('上传 CSV').run()
                self.assertFalse(app.exception)
                self.assertTrue(app.info)
            store.db.close()
