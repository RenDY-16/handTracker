import unittest
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from main import check_fist_gesture, check_claw_gesture, check_metal_gesture, check_peace_gesture, check_ok_gesture, check_finger_gun_gesture

class MockLandmark:
    def __init__(self, x, y, z=0.0):
        self.x = x
        self.y = y
        self.z = z

class TestGestureDetectors(unittest.TestCase):
    def setUp(self):
        self.w = 640
        self.h = 480
        # Create a standard mock hand landmarks list (21 points)
        self.hand_lms = [MockLandmark(0.5, 0.5) for _ in range(21)]

    def test_gesture_functions_return_bool(self):
        self.assertIsInstance(check_fist_gesture(self.hand_lms, self.w, self.h), bool)
        self.assertIsInstance(check_claw_gesture(self.hand_lms, self.w, self.h), bool)
        self.assertIsInstance(check_metal_gesture(self.hand_lms, self.w, self.h), bool)
        self.assertIsInstance(check_peace_gesture(self.hand_lms, self.w, self.h), bool)
        self.assertIsInstance(check_ok_gesture(self.hand_lms, self.w, self.h), bool)
        self.assertIsInstance(check_finger_gun_gesture(self.hand_lms, self.w, self.h), bool)

if __name__ == "__main__":
    unittest.main()
