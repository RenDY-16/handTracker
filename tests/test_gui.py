import unittest
import sys
import os

# Add parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from gui_control import ControlPanelGUI

class TestControlPanelGUI(unittest.TestCase):
    def test_gui_initialization(self):
        app_ctx = {
            "filters": ["VENOM-VISION", "SPIDER-MAN"],
            "show_telemetry_hud": False,
            "show_help_overlay": False,
            "show_hud_buttons": True,
            "sci_fi_glow": True,
        }
        panel = ControlPanelGUI(app_ctx)
        self.assertIsNotNone(panel)
        self.assertEqual(panel.app_context["filters"], ["VENOM-VISION", "SPIDER-MAN"])

    def test_get_camera_list(self):
        panel = ControlPanelGUI()
        cams = panel._get_camera_list()
        self.assertIsInstance(cams, list)
        self.assertGreater(len(cams), 0)

if __name__ == '__main__':
    unittest.main()
