import unittest
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sound_manager import play_sound_touch, play_sound_lock, play_sound_unlock, play_sound_snap, play_sound_thwip, play_sound_roar, play_sound_glitch

class TestAudioSynthesizer(unittest.TestCase):
    def test_audio_functions_execute_without_blocking(self):
        try:
            play_sound_touch()
            play_sound_lock()
            play_sound_unlock()
            play_sound_snap()
            play_sound_thwip()
            play_sound_roar()
            play_sound_glitch()
        except Exception as e:
            self.fail(f"Audio sound synthesis raised exception: {e}")

if __name__ == "__main__":
    unittest.main()
