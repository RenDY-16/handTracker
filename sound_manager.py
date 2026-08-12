"""
Retrolens Sound Manager & Procedural Sound Synthesizer
Uses pygame.mixer for zero-latency, high-definition sound effects.
"""
import os
import math
import struct
import wave
import threading

# Directory for cached sound assets
SOUNDS_DIR = os.path.join(os.path.dirname(__file__), "assets", "sounds")
os.makedirs(SOUNDS_DIR, exist_ok=True)

# Pygame Mixer State
HAS_PYGAME = False
SOUND_CACHE = {}

def _generate_wav_files():
    """Procedurally generates high-quality 16-bit 44.1kHz stereo WAV files."""
    sample_rate = 44100

    def write_wav(filename, samples):
        path = os.path.join(SOUNDS_DIR, filename)
        if os.path.exists(path):
            return path
        try:
            with wave.open(path, 'w') as wf:
                wf.setnchannels(2)  # Stereo
                wf.setsampwidth(2)  # 16-bit
                wf.setframerate(sample_rate)
                
                raw_data = bytearray()
                for left, right in samples:
                    il = max(-32768, min(32767, int(left * 32767)))
                    ir = max(-32768, min(32767, int(right * 32767)))
                    raw_data.extend(struct.pack('<hh', il, ir))
                wf.writeframes(raw_data)
        except Exception as e:
            print(f"Error generating {filename}: {e}")
        return path

    # 1. THWIP (Spider-Man Web Shot: High pitch sweep 900Hz -> 1800Hz -> 300Hz)
    num_samples = int(sample_rate * 0.12)
    samples_thwip = []
    for i in range(num_samples):
        t = i / sample_rate
        freq = 900 + 1200 * math.sin(t * 30) * math.exp(-t * 15) - t * 2000
        freq = max(200, freq)
        env = math.exp(-t * 22)
        val = math.sin(2 * math.pi * freq * t) * env * 0.7
        # Slight stereo separation
        samples_thwip.append((val * 0.9, val * 1.0))
    write_wav("thwip.wav", samples_thwip)

    # 2. ROAR (Venom Symbiote Deep Alien Growl: 70Hz sub-bass + pitch modulation + distortion)
    num_samples = int(sample_rate * 0.35)
    samples_roar = []
    import random
    rng = random.Random(42)
    for i in range(num_samples):
        t = i / sample_rate
        freq = 65 + 110 * math.sin(2 * math.pi * 8 * t) * (1.0 - t * 2)
        freq = max(40, freq)
        env = math.sin(math.pi * (i / num_samples))  # Fade in & out
        noise = (rng.random() * 2 - 1) * 0.2
        val = (math.sin(2 * math.pi * freq * t) * 0.6 + math.sin(2 * math.pi * (freq * 1.5) * t) * 0.3 + noise) * env * 0.8
        # Soft clipping overdrive
        val = math.tanh(val * 1.8)
        samples_roar.append((val, val * 0.85))
    write_wav("roar.wav", samples_roar)

    # 3. SHUTTER (Camera Snap Click: White noise burst + 2 sharp transients)
    num_samples = int(sample_rate * 0.08)
    samples_shutter = []
    for i in range(num_samples):
        t = i / sample_rate
        noise = (rng.random() * 2 - 1)
        click1 = math.exp(-t * 120) * math.sin(2 * math.pi * 1800 * t)
        t2 = max(0.0, t - 0.03)
        click2 = math.exp(-t2 * 100) * math.sin(2 * math.pi * 1200 * t2) if t > 0.03 else 0.0
        val = (click1 * 0.6 + click2 * 0.5 + noise * math.exp(-t * 80) * 0.3)
        val = max(-1.0, min(1.0, val))
        samples_shutter.append((val, val))
    write_wav("shutter.wav", samples_shutter)

    # 4. GLITCH (Multiverse Digital Cyber Zap: Rapid 3-tone arpeggio)
    num_samples = int(sample_rate * 0.15)
    samples_glitch = []
    for i in range(num_samples):
        t = i / sample_rate
        stage = int(t * 25) % 3
        freqs = [1050, 480, 1420]
        freq = freqs[stage]
        sq = 0.5 if (math.sin(2 * math.pi * freq * t) > 0) else -0.5
        env = math.exp(-t * 12)
        val = sq * env * 0.5
        samples_glitch.append((val * 0.8, val * 1.0))
    write_wav("glitch.wav", samples_glitch)

    # 5. LOCK (Sci-Fi Engage Chime: Rising 2-tone chime 320Hz -> 640Hz)
    num_samples = int(sample_rate * 0.10)
    samples_lock = []
    for i in range(num_samples):
        t = i / sample_rate
        freq = 320 if t < 0.04 else 640
        env = math.exp(-(t % 0.04) * 35)
        val = math.sin(2 * math.pi * freq * t) * env * 0.6
        samples_lock.append((val, val * 0.9))
    write_wav("lock.wav", samples_lock)

    # 6. UNLOCK (Sci-Fi Disengage Chime: Falling 2-tone chime 640Hz -> 320Hz)
    num_samples = int(sample_rate * 0.10)
    samples_unlock = []
    for i in range(num_samples):
        t = i / sample_rate
        freq = 640 if t < 0.04 else 320
        env = math.exp(-(t % 0.04) * 35)
        val = math.sin(2 * math.pi * freq * t) * env * 0.6
        samples_unlock.append((val * 0.9, val))
    write_wav("unlock.wav", samples_unlock)

    # 7. CLICK (Tactile UI Touch Squelch: Subtle low click)
    num_samples = int(sample_rate * 0.04)
    samples_click = []
    for i in range(num_samples):
        t = i / sample_rate
        val = math.sin(2 * math.pi * 320 * t) * math.exp(-t * 90) * 0.4
        samples_click.append((val, val))
    write_wav("click.wav", samples_click)

def _init_sound_engine():
    global HAS_PYGAME, SOUND_CACHE
    try:
        import pygame
        pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
        HAS_PYGAME = True
        
        # Pre-generate & Load Sounds into Mixer Cache
        _generate_wav_files()
        
        for fname in ["thwip.wav", "roar.wav", "shutter.wav", "glitch.wav", "lock.wav", "unlock.wav", "click.wav"]:
            path = os.path.join(SOUNDS_DIR, fname)
            if os.path.exists(path):
                key = os.path.splitext(fname)[0]
                SOUND_CACHE[key] = pygame.mixer.Sound(path)
    except Exception as e:
        HAS_PYGAME = False
        print(f"Pygame Mixer Sound Engine fallback (winsound mode): {e}")

# Initialize on module import
_init_sound_engine()

def play_sound(name):
    """Asynchronously plays a cached HD sound effect."""
    def _play():
        if HAS_PYGAME and name in SOUND_CACHE:
            try:
                SOUND_CACHE[name].play()
                return
            except Exception:
                pass
        
        # Fallback to winsound if Pygame mixer unavailable
        try:
            import winsound
            fallback_map = {
                "thwip": (520, 20),
                "roar": (120, 40),
                "shutter": (1400, 25),
                "glitch": (980, 20),
                "lock": (320, 25),
                "unlock": (260, 25),
                "click": (260, 20)
            }
            if name in fallback_map:
                freq, dur = fallback_map[name]
                winsound.Beep(freq, dur)
        except Exception:
            pass

    threading.Thread(target=_play, daemon=True).start()

# Convenient API aliases matching main.py conventions
def play_sound_touch():
    play_sound("click")

def play_sound_lock():
    play_sound("lock")

def play_sound_unlock():
    play_sound("unlock")

def play_sound_snap():
    play_sound("shutter")

def play_sound_thwip():
    play_sound("thwip")

def play_sound_roar():
    play_sound("roar")

def play_sound_glitch():
    play_sound("glitch")
