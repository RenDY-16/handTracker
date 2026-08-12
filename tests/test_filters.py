import unittest
import numpy as np
import sys
import os

# Import main filter function
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from main import apply_filter, filters

class TestVisualFilters(unittest.TestCase):
    def setUp(self):
        # Create a synthetic 100x100 BGR test image frame
        self.sample_roi = np.zeros((100, 100, 3), dtype=np.uint8)
        # Fill gradient colors
        for y in range(100):
            for x in range(100):
                self.sample_roi[y, x] = [x * 2, y * 2, (x + y) % 255]

    def test_all_filters_produce_valid_frame(self):
        for f_name in filters:
            with self.subTest(filter_name=f_name):
                result = apply_filter(self.sample_roi, f_name)
                self.assertIsNotNone(result, f"Filter {f_name} returned None")
                self.assertEqual(result.shape, self.sample_roi.shape, f"Filter {f_name} changed ROI dimensions")
                self.assertEqual(result.dtype, np.uint8, f"Filter {f_name} output dtype is not uint8")

if __name__ == "__main__":
    unittest.main()
