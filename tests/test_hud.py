import unittest
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# Test HUD layout calculations

class TestHUDLayout(unittest.TestCase):
    def test_pill_button_bounds_calculation(self):
        btn_w, btn_h = 145, 42
        gap = 14
        buttons_count = 4
        w, h = 1920, 1080
        total_w = buttons_count * btn_w + (buttons_count - 1) * gap
        start_x = max(10, (w - total_w) // 2)
        start_y = h - 65
        
        self.assertGreater(start_x, 0)
        self.assertLess(start_x + total_w, w)
        self.assertLess(start_y + btn_h, h)

if __name__ == "__main__":
    unittest.main()
