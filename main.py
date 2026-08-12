# pyrefly: ignore [missing-import]
import cv2
import mediapipe as mp
import time
import math
import numpy as np
import pyvirtualcam
import threading
import os

# Pastikan folder captures/ tersedia untuk foto Air-Snap
os.makedirs("captures", exist_ok=True)

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.7,
    min_tracking_confidence=0.7
)
landmarker = HandLandmarker.create_from_options(options)

# --- Class Threaded Camera (Mencegah Read Latency / Stutter) ---
class ThreadedCamera:
    def __init__(self, src=0, w=1920, h=1080, fps=30):
        self.cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        self.grabbed, self.frame = self.cap.read()
        self.started = False
        self.read_lock = threading.Lock()

    def start(self):
        if self.started:
            return self
        self.started = True
        self.thread = threading.Thread(target=self.update, args=(), daemon=True)
        self.thread.start()
        return self

    def update(self):
        while self.started:
            grabbed, frame = self.cap.read()
            if not grabbed:
                continue
            with self.read_lock:
                self.grabbed, self.frame = grabbed, frame

    def read(self):
        with self.read_lock:
            if not self.grabbed or self.frame is None:
                return False, None
            return True, self.frame.copy()

    def get(self, prop_id):
        return self.cap.get(prop_id)

    def stop(self):
        self.started = False
        if hasattr(self, 'thread') and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        self.cap.release()

PROCESS_W, PROCESS_H = 1920, 1080

# --- State Variables ---

# --- HD Sound Engine (pygame.mixer) ---
from sound_manager import (
    play_sound_touch, play_sound_lock, play_sound_unlock,
    play_sound_snap, play_sound_thwip, play_sound_roar, play_sound_glitch
)



filters = ["VENOM-VISION", "SPIDER-MAN", "SPIDER-VENOM-UNITE", "VENOM-CARNAGE", "CARNAGE-VORTEX", "SPIDER-2099", "CYBER-VORTEX", "SYMBIOTE-VORTEX", "SYMBIOTE-BURST", "MULTIVERSE-GLITCH", "NIGHT-VISION", "HOLOGRAM", "THERMAL-SCAN", "SKETCH-NOIR", "SYMBIOTE-RED", "SPIDER-NOIR", "TOXIC-SYMBIOTE", "ANTI-VENOM", "EMBOSS", "GLITCH", "NEON", "MONO", "PIXELATE", "INVERT"]
current_filter = 0
gesture_triggered = False
frame_count = 0
last_results = None
sci_fi_glow = True
zoom_mode = True        # Toggle mode Dynamic Finger Zoom
zoom_locked = False     # State Lock Zoom (True = Terkunci, False = Bebas)
current_zoom = 1.0      # Level zoom saat ini (1.0 = normal, max = 2.5)
zoom_cx = PROCESS_W // 2
zoom_cy = PROCESS_H // 2
show_hud_buttons = True
was_portal_active = False
show_help_overlay = False
show_telemetry_hud = False

# Variables for Hologram HUD & Air-Snap / Recording / Combo / Telemetry
last_touch_time = 0
touch_cooldown = 0.5
last_ok_gesture_time = 0
last_combo_time = 0
last_gun_gesture_time = 0
previous_filter = 0
snap_notification = ""
snap_notification_time = 0
snap_flash_alpha = 0.0
fps_ema = 30.0
last_fps_time = time.time()

def draw_telemetry_hud(img, w, h, hands_count, fps_val):
    rx1, ry1 = 15, 85
    rx2, ry2 = 295, 175
    
    overlay = img.copy()
    cv2.rectangle(overlay, (rx1, ry1), (rx2, ry2), (10, 8, 18), -1)
    cv2.rectangle(overlay, (rx1, ry1), (rx2, ry2), (35, 30, 60), 1, cv2.LINE_AA)
    img = cv2.addWeighted(overlay, 0.88, img, 0.12, 0)
    
    latency = max(8.0, 1000.0 / max(1.0, fps_val))
    cv2.putText(img, "AI TELEMETRY DIAGNOSTICS", (rx1 + 12, ry1 + 22),
                cv2.FONT_HERSHEY_DUPLEX, 0.38, (240, 240, 255), 1, cv2.LINE_AA)
    cv2.line(img, (rx1 + 12, ry1 + 28), (rx2 - 12, ry1 + 28), (20, 20, 220), 1, cv2.LINE_AA)
    
    cv2.putText(img, f"• LATENCY: {latency:.1f} ms", (rx1 + 12, ry1 + 46),
                cv2.FONT_HERSHEY_DUPLEX, 0.35, (0, 255, 160), 1, cv2.LINE_AA)
    cv2.putText(img, f"• HANDS DETECTED: {hands_count}/2", (rx1 + 12, ry1 + 62),
                cv2.FONT_HERSHEY_DUPLEX, 0.35, (220, 220, 240), 1, cv2.LINE_AA)
    cv2.putText(img, f"• AI MODEL: MEDIAPIPE TASK", (rx1 + 12, ry1 + 78),
                cv2.FONT_HERSHEY_DUPLEX, 0.35, (180, 180, 200), 1, cv2.LINE_AA)
    return img

# Video Recording State
is_recording = False
rec_writer = None
rec_start_time = 0
rec_filename = ""
rec_max_duration = 5.0 # 5 detik clip

def start_recording(img_shape, fps=30):
    global is_recording, rec_writer, rec_start_time, rec_filename
    if is_recording:
        return
    h, w = img_shape[:2]
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    os.makedirs("captures/recordings", exist_ok=True)
    rec_filename = f"captures/recordings/Venom_Clip_{timestamp}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    rec_writer = cv2.VideoWriter(rec_filename, fourcc, fps, (w, h))
    if not rec_writer.isOpened():
        rec_filename = f"captures/recordings/Venom_Clip_{timestamp}.avi"
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        rec_writer = cv2.VideoWriter(rec_filename, fourcc, fps, (w, h))
    is_recording = True
    rec_start_time = time.time()
    play_sound_lock()

def stop_recording():
    global is_recording, rec_writer
    if not is_recording:
        return
    is_recording = False
    if rec_writer is not None:
        rec_writer.release()
        rec_writer = None
    play_sound_unlock()

def check_fist_gesture(hand_lms, w, h):
    wrist = hand_lms[0]
    def d_wrist(idx):
        return math.hypot((hand_lms[idx].x - wrist.x) * w, (hand_lms[idx].y - wrist.y) * h)
    
    fist = True
    for tip_id, mcp_id in [(4, 2), (8, 5), (12, 9), (16, 13), (20, 17)]:
        if d_wrist(tip_id) > d_wrist(mcp_id) * 1.05:
            fist = False
            break
    return fist

def draw_help_overlay(img, w, h):
    pad_x, pad_y = 60, 40
    x1, y1 = pad_x, pad_y
    x2, y2 = w - pad_x, h - pad_y
    
    overlay = img.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (8, 6, 16), -1)
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (40, 35, 80), 2, cv2.LINE_AA)
    img = cv2.addWeighted(overlay, 0.92, img, 0.08, 0)
    
    cv2.putText(img, "VENOM RETROLENS - GESTURE CONTROL GUIDE", (x1 + 30, y1 + 45),
                cv2.FONT_HERSHEY_DUPLEX, 0.62, (240, 240, 255), 1, cv2.LINE_AA)
    cv2.line(img, (x1 + 30, y1 + 58), (x2 - 30, y1 + 58), (20, 20, 220), 2, cv2.LINE_AA)
    
    items = [
        ("Gestur Peace (✌️) / Shortcut 's'", "AIR-SNAP PHOTO CAPTURE (Simpan Foto HD ke captures/)"),
        ("Gestur Fist (✊) / Shortcut 'r'", "VIDEO AIR-CLIP RECORDER (Rekam 5 detik ke captures/recordings/)"),
        ("Gestur Symbiote Claw (🖐️)", "INSTANT TRIGGER VECTORS -> VENOM-VISION 🕷️"),
        ("Gestur OK (👌)", "LOCK / UNLOCK SYMBIOTE ZOOM (Kunci/Buka Zoom Kamera)"),
        ("Tangan Kanan ✋ (Jempol + Manis)", "NEXT FILTER ⏭️ (Pindah ke filter berikutnya)"),
        ("Tangan Kiri 🤚 (Jempol + Manis)", "PREVIOUS FILTER ⏮️ (Pindah ke filter sebelumnya)"),
        ("1 Tangan Metal (🤟)", "MUNCULKAN MENU HUD (Tampilkan tombol kontrol bawah)"),
        ("2 Tangan Metal (🤟 🤟)", "DUAL COMBO: MULTIVERSE GLITCH BURST + ALIEN ROAR"),
        ("Portal 4 Jari", "Membuka Portal Filter dengan Jempol & Telunjuk kedua tangan"),
        ("Shortcut 'd'", "Toggle AI Telemetry Diagnostics HUD"),
        ("Shortcut 'm'", "Toggle Show / Hide HUD Menu"),
        ("Shortcut 'h'", "Buka / Tutup Panduan Gestur Ini"),
        ("Shortcut '1'-'8'", "Direct Theme Jump (Venom/Spidey/Vortex/Glitch/Anti/Unite/Cyber/Carnage)"),
        ("Shortcut '9'-'0'", "Direct Theme Jump (Night-Vision / Hologram)"),
        ("Shortcut 'q'", "Keluar Aplikasi")
    ]
    
    start_y = y1 + 95
    for i, (gest, desc) in enumerate(items):
        cy = start_y + i * 28
        cv2.putText(img, f"• {gest}:", (x1 + 35, cy), cv2.FONT_HERSHEY_DUPLEX, 0.38, (20, 20, 240), 1, cv2.LINE_AA)
        cv2.putText(img, desc, (x1 + 320, cy), cv2.FONT_HERSHEY_DUPLEX, 0.36, (220, 220, 235), 1, cv2.LINE_AA)
        
    cv2.putText(img, "Tekan 'h' atau 'ESC' untuk menutup panduan ini", (x1 + 35, y2 - 20),
                cv2.FONT_HERSHEY_DUPLEX, 0.42, (150, 150, 175), 1, cv2.LINE_AA)
    return img

def check_peace_gesture(hand_lms, w, h):
    wrist = hand_lms[0]
    
    def d_wrist(idx):
        return math.hypot((hand_lms[idx].x - wrist.x) * w, (hand_lms[idx].y - wrist.y) * h)
    
    index_open   = d_wrist(8)  > d_wrist(6) * 1.15
    middle_open  = d_wrist(12) > d_wrist(10) * 1.15
    ring_closed  = d_wrist(16) < d_wrist(14) * 1.1
    pinky_closed = d_wrist(20) < d_wrist(18) * 1.1
    
    pt8 = (hand_lms[8].x * w, hand_lms[8].y * h)
    pt12 = (hand_lms[12].x * w, hand_lms[12].y * h)
    fingers_separated = math.hypot(pt8[0] - pt12[0], pt8[1] - pt12[1]) > 25
    
    return index_open and middle_open and ring_closed and pinky_closed and fingers_separated

def check_ok_gesture(hand_lms, w, h):
    thumb = hand_lms[4]
    index = hand_lms[8]
    wrist = hand_lms[0]
    
    def d_wrist(idx):
        return math.hypot((hand_lms[idx].x - wrist.x) * w, (hand_lms[idx].y - wrist.y) * h)
    
    pinch_dist = math.hypot((thumb.x - index.x) * w, (thumb.y - index.y) * h)
    is_pinch = pinch_dist < 34
    
    middle_open = d_wrist(12) > d_wrist(10) * 1.12
    ring_open   = d_wrist(16) > d_wrist(14) * 1.12
    pinky_open  = d_wrist(20) > d_wrist(18) * 1.12
    
    return is_pinch and middle_open and ring_open and pinky_open

def check_claw_gesture(hand_lms, w, h):
    wrist = hand_lms[0]
    
    def d_wrist(idx):
        return math.hypot((hand_lms[idx].x - wrist.x) * w, (hand_lms[idx].y - wrist.y) * h)
    
    # Kelima jari setengah tertekuk (curled/bent forward)
    curled = True
    for tip_id, mcp_id in [(8, 5), (12, 9), (16, 13), (20, 17)]:
        dist_tip = d_wrist(tip_id)
        dist_mcp = d_wrist(mcp_id)
        if dist_tip > dist_mcp * 1.35 or dist_tip < dist_mcp * 0.8:
            curled = False
            break
            
    return curled

def check_metal_gesture(hand_lms, w, h):
    wrist = hand_lms[0]
    
    def d_wrist(idx):
        return math.hypot((hand_lms[idx].x - wrist.x) * w, (hand_lms[idx].y - wrist.y) * h)
    
    index_open = d_wrist(8) > d_wrist(6) * 1.12
    pinky_open = d_wrist(20) > d_wrist(18) * 1.12
    
    middle_closed = d_wrist(12) < d_wrist(10) * 1.1 or d_wrist(12) < d_wrist(9) * 1.15
    ring_closed   = d_wrist(16) < d_wrist(14) * 1.1 or d_wrist(16) < d_wrist(13) * 1.15
    
    return index_open and pinky_open and middle_closed and ring_closed

def check_finger_gun_gesture(hand_lms, w, h):
    wrist = hand_lms[0]
    
    def d_wrist(idx):
        return math.hypot((hand_lms[idx].x - wrist.x) * w, (hand_lms[idx].y - wrist.y) * h)
    
    # Index extended, thumb up, middle/ring/pinky closed
    index_open    = d_wrist(8)  > d_wrist(6)  * 1.15
    thumb_open    = d_wrist(4)  > d_wrist(3)  * 1.10
    middle_closed = d_wrist(12) < d_wrist(10) * 1.1
    ring_closed   = d_wrist(16) < d_wrist(14) * 1.1
    pinky_closed  = d_wrist(20) < d_wrist(18) * 1.1
    
    return index_open and thumb_open and middle_closed and ring_closed and pinky_closed


def apply_filter(roi, filter_name, x=0, y=0):
    if filter_name == "VENOM-VISION":
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        contrast = cv2.addWeighted(gray, 1.35, gray, 0, -20)
        bgr = cv2.cvtColor(contrast, cv2.COLOR_GRAY2BGR)
        mask_bright = contrast > 175
        mask_mid = (contrast > 80) & (contrast <= 175)
        bgr[mask_bright] = [250, 250, 255]
        bgr[mask_mid] = [25, 20, 210]
        return cv2.addWeighted(roi, 0.25, bgr, 0.75, 0)
    elif filter_name == "SPIDER-MAN":
        # Marvel Comic Book Vibrant Color + Halftone Dots Filter (Cinematic Blend)
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        h_r, w_r = roi.shape[:2]
        
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        hsv[:, :, 1] = cv2.add(hsv[:, :, 1], 45) # Subtle Saturation Boost
        vibrant = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        dot_spacing = 7
        grid_y, grid_x = np.mgrid[0:h_r, 0:w_r]
        dot_mask = ((grid_x % dot_spacing == 0) & (grid_y % dot_spacing == 0))
        
        edges = cv2.Canny(gray, 80, 160)
        vibrant[edges > 0] = [15, 10, 25] # Comic Ink Line
        
        # Alpha blended comic dots (Rule #3: 0.3 opacity blend)
        dots_layer = vibrant.copy()
        dots_layer[dot_mask & (gray < 160)] = [200, 30, 45]
        return cv2.addWeighted(vibrant, 0.70, dots_layer, 0.30, 0)
    elif filter_name == "VENOM-CARNAGE":
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        contrast = cv2.addWeighted(gray, 1.45, gray, 0, -30)
        bgr = cv2.cvtColor(contrast, cv2.COLOR_GRAY2BGR)
        mask_bright = contrast > 160
        bgr[mask_bright] = [20, 20, 240]
        return cv2.addWeighted(roi, 0.30, bgr, 0.70, 0)
    elif filter_name == "SPIDER-2099":
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 60, 140)
        bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        bgr[:, :, 0] = cv2.add(bgr[:, :, 0], 80)
        bgr[edges > 0] = [240, 220, 30]
        return cv2.addWeighted(roi, 0.30, bgr, 0.70, 0)
    elif filter_name == "SYMBIOTE-VORTEX":
        h_r, w_r = roi.shape[:2]
        shift = max(6, w_r // 16)
        vortex_roi = roi.copy()
        if w_r > shift:
            vortex_roi[:, :-shift, 2] = cv2.addWeighted(roi[:, :-shift, 2], 0.7, roi[:, shift:, 2], 0.3, 0)
            vortex_roi[:, shift:, 0] = cv2.addWeighted(roi[:, shift:, 0], 0.7, roi[:, :-shift, 0], 0.3, 0)
        
        X = np.linspace(-1, 1, w_r)
        Y = np.linspace(-1, 1, h_r)
        xx, yy = np.meshgrid(X, Y)
        radius = np.sqrt(xx**2 + yy**2)
        vignette = np.clip(1.0 - radius * 0.65, 0.2, 1.0).astype(np.float32)
        vignette_3ch = cv2.merge([vignette, vignette, vignette])
        
        return (vortex_roi.astype(np.float32) * vignette_3ch).astype(np.uint8)
    elif filter_name == "SPIDER-VENOM-UNITE":
        h_r, w_r = roi.shape[:2]
        mid_h = h_r // 2
        spidey_top = apply_filter(roi[:mid_h, :], "SPIDER-MAN")
        venom_bot = apply_filter(roi[mid_h:, :], "VENOM-VISION")
        united = np.vstack([spidey_top, venom_bot])
        cv2.line(united, (0, mid_h), (w_r, mid_h), (240, 240, 255), 1, cv2.LINE_AA)
        return united
    elif filter_name == "CYBER-VORTEX":
        c_2099 = apply_filter(roi, "SPIDER-2099")
        vortex = apply_filter(c_2099, "SYMBIOTE-VORTEX")
        return cv2.addWeighted(c_2099, 0.40, vortex, 0.60, 0)
    elif filter_name == "CARNAGE-VORTEX":
        carnage = apply_filter(roi, "VENOM-CARNAGE")
        vortex = apply_filter(carnage, "SYMBIOTE-VORTEX")
        return cv2.addWeighted(carnage, 0.40, vortex, 0.60, 0)
    elif filter_name == "SYMBIOTE-BURST":
        h_r, w_r = roi.shape[:2]
        burst_roi = roi.copy()
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 70, 150)
        burst_roi[edges > 0] = [220, 20, 240]
        X = np.linspace(-1, 1, w_r)
        Y = np.linspace(-1, 1, h_r)
        xx, yy = np.meshgrid(X, Y)
        radius = np.sqrt(xx**2 + yy**2)
        vignette = np.clip(1.0 - radius * 0.50, 0.3, 1.0).astype(np.float32)
        vignette_3ch = cv2.merge([vignette, vignette, vignette])
        b_frame = (burst_roi.astype(np.float32) * vignette_3ch).astype(np.uint8)
        return cv2.addWeighted(roi, 0.30, b_frame, 0.70, 0)
    elif filter_name == "MULTIVERSE-GLITCH":
        h_r, w_r = roi.shape[:2]
        glitch_roi = roi.copy()
        if w_r > 20:
            s1 = w_r // 10
            s2 = w_r // 15
            glitch_roi[:, :-s1, 2] = roi[:, s1:, 2]
            glitch_roi[:, s2:, 1] = roi[:, :-s2, 1]
            glitch_roi[::4, :, :] = cv2.addWeighted(glitch_roi[::4, :, :], 0.4, np.full_like(glitch_roi[::4, :, :], 240), 0.6, 0)
        return glitch_roi
    elif filter_name == "SYMBIOTE-RED":
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, mask_c = cv2.threshold(gray, 110, 255, cv2.THRESH_BINARY)
        filtered = np.zeros_like(roi)
        filtered[mask_c == 255] = [20, 20, 240]   # Blood Crimson Red
        filtered[mask_c == 0] = [10, 10, 15]     # Pitch Black Symbiote
        return filtered
    elif filter_name == "SPIDER-NOIR":
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        return cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
    elif filter_name == "TOXIC-SYMBIOTE":
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, mask_c = cv2.threshold(gray, 110, 255, cv2.THRESH_BINARY)
        filtered = np.zeros_like(roi)
        filtered[mask_c == 255] = [40, 240, 60]   # Toxic Green
        filtered[mask_c == 0] = [10, 10, 15]     # Pitch Black
        return filtered
    elif filter_name == "ANTI-VENOM":
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        inv = cv2.bitwise_not(gray)
        contrast = cv2.addWeighted(inv, 1.5, inv, 0, -20)
        return cv2.cvtColor(contrast, cv2.COLOR_GRAY2BGR)
    elif filter_name == "MONO":
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    elif filter_name == "INVERT":
        return cv2.bitwise_not(roi)
    elif filter_name == "BLUR":
        return cv2.GaussianBlur(roi, (25, 25), 0)
    elif filter_name == "SEPIA":
        kernel = np.array([[0.272, 0.534, 0.131],
                           [0.349, 0.686, 0.168],
                           [0.393, 0.769, 0.189]])
        filtered = cv2.transform(roi, kernel)
        return np.clip(filtered, 0, 255).astype(np.uint8)
    elif filter_name == "DUAL-TONE":
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, mask_c = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        filtered = np.zeros_like(roi)
        filtered[mask_c == 255] = [20, 20, 240]
        filtered[mask_c == 0] = [240, 240, 255]
        return filtered
    elif filter_name == "PIXELATE":
        h_r, w_r = roi.shape[:2]
        if h_r > 10 and w_r > 10:
            small = cv2.resize(roi, (w_r//10, h_r//10), interpolation=cv2.INTER_LINEAR)
            return cv2.resize(small, (w_r, h_r), interpolation=cv2.INTER_NEAREST)
    elif filter_name == "THERMAL":
        return cv2.applyColorMap(roi, cv2.COLORMAP_JET)
    elif filter_name == "SKETCH":
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        inv = cv2.bitwise_not(gray)
        blur = cv2.GaussianBlur(inv, (21, 21), 0)
        sketch = cv2.divide(gray, 255 - blur, scale=256)
        return cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)
    elif filter_name == "GLITCH":
        h_r, w_r = roi.shape[:2]
        shift = max(5, w_r // 20)
        glitch_roi = roi.copy()
        if w_r > shift:
            glitch_roi[:, :-shift, 2] = roi[:, shift:, 2]
            glitch_roi[:, shift:, 0] = roi[:, :-shift, 0]
        return glitch_roi
    elif filter_name == "NEON":
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)
        edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        edges_bgr[np.where((edges_bgr == [255, 255, 255]).all(axis=2))] = [20, 20, 240] # Crimson Red Neon
        kernel = np.ones((3,3), np.uint8)
        return cv2.dilate(edges_bgr, kernel, iterations=1)
    elif filter_name == "NIGHT-VISION":
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        bright = cv2.addWeighted(gray, 1.6, gray, 0, 15)
        green_frame = np.zeros_like(roi)
        green_frame[:, :, 1] = bright  # Green channel only
        green_frame[:, :, 0] = (bright * 0.15).astype(np.uint8)  # Slight blue tint
        noise = np.random.randint(0, 12, gray.shape, dtype=np.uint8)
        green_frame[:, :, 1] = cv2.add(green_frame[:, :, 1], noise)
        return green_frame
    elif filter_name == "HOLOGRAM":
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 40, 120)
        h_r, w_r = roi.shape[:2]
        holo = np.zeros_like(roi)
        holo[:, :, 0] = (edges * 0.9).astype(np.uint8)   # Blue wireframe
        holo[:, :, 1] = (edges * 0.4).astype(np.uint8)   # Slight green
        # Scanline effect
        holo[::3, :, 0] = np.clip(holo[::3, :, 0].astype(np.int16) + 25, 0, 255).astype(np.uint8)
        return cv2.addWeighted(roi, 0.15, holo, 0.85, 0)
    elif filter_name == "THERMAL-SCAN":
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        thermal = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
        return cv2.addWeighted(roi, 0.20, thermal, 0.80, 0)
    elif filter_name == "SKETCH-NOIR":
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        inv = cv2.bitwise_not(gray)
        blur = cv2.GaussianBlur(inv, (21, 21), 0)
        sketch = cv2.divide(gray, 255 - blur, scale=256)
        sketch_bgr = cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)
        # Desaturated warm noir tint
        tint = np.full_like(sketch_bgr, (15, 12, 25))
        return cv2.addWeighted(sketch_bgr, 0.85, tint, 0.15, 0)
    elif filter_name == "EMBOSS":
        kernel = np.array([[-2, -1, 0],
                           [-1,  1, 1],
                           [ 0,  1, 2]])
        embossed = cv2.filter2D(roi, -1, kernel) + 128
        return cv2.addWeighted(roi, 0.25, embossed, 0.75, 0)
        
    return roi

def run_app():
    global PROCESS_W, PROCESS_H, current_filter, gesture_triggered, frame_count, last_results, sci_fi_glow, zoom_mode, zoom_locked, current_zoom, zoom_cx, zoom_cy, show_hud_buttons, was_portal_active, show_help_overlay, show_telemetry_hud, last_touch_time, last_ok_gesture_time, last_combo_time, snap_notification, snap_notification_time, snap_flash_alpha, fps_ema, last_fps_time, is_recording, rec_writer, rec_start_time, rec_filename, last_gun_gesture_time, previous_filter

    from pygrabber.dshow_graph import FilterGraph

    graph = FilterGraph()
    device_names = graph.get_input_devices()

    selected_idx = 0
    if device_names:
        for i, name in enumerate(device_names):
            if "nvidia broadcast" in name.lower():
                selected_idx = i
                break

    camera = ThreadedCamera(selected_idx, 1920, 1080, 30).start()

    actual_w = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
    PROCESS_W, PROCESS_H = actual_w, actual_h

    # --- Launch GUI Control Panel Thread ---
    gui_snap_requested = False

    from gui_control import ControlPanelGUI

    def set_filter_cb(f_name):
        global current_filter
        if f_name in filters:
            current_filter = filters.index(f_name)

    def set_camera_cb(cam_str):
        pass

    def sync_toggles_cb(toggles):
        global show_telemetry_hud, show_help_overlay, show_hud_buttons, sci_fi_glow
        show_telemetry_hud = toggles.get("show_telemetry_hud", show_telemetry_hud)
        show_help_overlay = toggles.get("show_help_overlay", show_help_overlay)
        show_hud_buttons = toggles.get("show_hud_buttons", show_hud_buttons)
        sci_fi_glow = toggles.get("sci_fi_glow", sci_fi_glow)

    def trigger_snap_cb():
        nonlocal gui_snap_requested
        gui_snap_requested = True

    def trigger_rec_cb():
        global is_recording
        if not is_recording:
            start_recording((PROCESS_H, PROCESS_W, 3))

    def trigger_reset_zoom_cb():
        global zoom_locked, current_zoom
        zoom_locked = False
        current_zoom = 1.0

    app_ctx = {
        "filters": filters,
        "get_current_filter": lambda: filters[current_filter],
        "get_toggles": lambda: {
            "show_telemetry_hud": show_telemetry_hud,
            "show_help_overlay": show_help_overlay,
            "show_hud_buttons": show_hud_buttons,
            "sci_fi_glow": sci_fi_glow
        },
        "set_filter_callback": set_filter_cb,
        "set_camera_callback": set_camera_cb,
        "sync_toggles_callback": sync_toggles_cb,
        "trigger_snap_callback": trigger_snap_cb,
        "trigger_rec_callback": trigger_rec_cb,
        "trigger_reset_zoom_callback": trigger_reset_zoom_cb
    }

    gui = ControlPanelGUI(app_ctx)
    gui.start_in_thread()

    cv2.namedWindow('RETROLENS Pake Python', cv2.WINDOW_NORMAL)

    while True:
        success, img = camera.read()
        if not success or img is None:
            time.sleep(0.005)
            continue

        img = cv2.flip(img, 1)
        h, w = img.shape[:2]

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        timestamp_ms = time.time_ns() // 1_000_000
        
        frame_count += 1
        if frame_count % 2 == 0 or last_results is None:
            results = landmarker.detect_for_video(mp_image, timestamp_ms)
            last_results = results
        else:
            results = last_results
        
        filter_name = filters[current_filter]
        
        pts_portal = []
        change_filter = False
        metal_count = 0
        index_tips = []
        target_zoom = 1.0
    
        if results.hand_landmarks:
            for i, hand_lms in enumerate(results.hand_landmarks):
                idx_pt = (int(hand_lms[8].x * w), int(hand_lms[8].y * h))
                index_tips.append(idx_pt)
                
                if check_metal_gesture(hand_lms, w, h):
                    metal_count += 1
    
            for hand_lms in results.hand_landmarks:
                for id, lm in enumerate(hand_lms):
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    if id in [4, 8]:
                        pts_portal.append([cx, cy])
    
            is_forming_portal = (len(pts_portal) == 4)
    
            if is_forming_portal:
                if not was_portal_active:
                    play_sound_portal_open()
                    was_portal_active = True
            else:
                was_portal_active = False
    
            for pt in pts_portal:
                draw_venom_fingertip(img, pt[0], pt[1], active=is_forming_portal)
    
            target_cx = w // 2
            target_cy = h // 2
    
            # --- HITUNG TARGET FINGER ZOOM & LOKASI TANGAN ---
            if zoom_mode and not is_forming_portal:
                if len(results.hand_landmarks) >= 1:
                    hand_lms = results.hand_landmarks[0]
                    
                    # Cek Gestur Fist (✊ Kepalan Tangan) -> Start 5-Second Video Recording Clip
                    if check_fist_gesture(hand_lms, w, h):
                        curr_t = time.time()
                        if curr_t - last_rec_gesture_time > 2.0 and not is_recording:
                            last_rec_gesture_time = curr_t
                            start_recording(img.shape)
    
                    # Cek Gestur Symbiote Claw (🖐️ cakar) -> Instant Switch ke VENOM-VISION
                    if check_claw_gesture(hand_lms, w, h):
                        curr_t = time.time()
                        if curr_t - last_theme_gesture_time > 0.8:
                            last_theme_gesture_time = curr_t
                            current_filter = filters.index("VENOM-VISION")
                            play_sound_roar()
                            snap_notification = "THEME ACTIVATED -> VENOM-VISION 🕷️"
                            snap_notification_time = curr_t + 1.8
    
                    # Cek Gestur Peace (✌️) untuk Air-Snap Photo Capture
                    if check_peace_gesture(hand_lms, w, h):
                        curr_t = time.time()
                        if curr_t - last_snap_time > 1.2:
                            last_snap_time = curr_t
                            snap_filename = f"captures/Venom_Capture_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
                            cv2.imwrite(snap_filename, img)
                            play_sound_snap()
                            snap_notification = f"PHOTO SAVED -> {os.path.basename(snap_filename)}"
                            snap_notification_time = curr_t + 2.5
                            snap_flash_alpha = 0.65
    
                    # Cek Gestur OK (👌) untuk Toggle LOCK / UNLOCK Zoom
                    if check_ok_gesture(hand_lms, w, h):
                        curr_t = time.time()
                        if curr_t - last_ok_gesture_time > 0.6:
                            last_ok_gesture_time = curr_t
                            zoom_locked = not zoom_locked
                            if zoom_locked:
                                play_sound_lock()
                            else:
                                play_sound_unlock()

                    # Cek Gestur Finger Gun (👉) untuk Quick-Switch ke filter terakhir
                    if check_finger_gun_gesture(hand_lms, w, h):
                        curr_t = time.time()
                        if curr_t - last_gun_gesture_time > 1.0:
                            last_gun_gesture_time = curr_t
                            old_filter = current_filter
                            current_filter = previous_filter
                            previous_filter = old_filter
                            play_sound_thwip()
                            snap_notification = f"QUICK-SWITCH -> {filters[current_filter]} 👉"
                            snap_notification_time = curr_t + 1.8
                    
                    if not zoom_locked:
                        thumb = (int(hand_lms[4].x * w), int(hand_lms[4].y * h))
                        index = (int(hand_lms[8].x * w), int(hand_lms[8].y * h))
                        dist_pinch = math.hypot(thumb[0] - index[0], thumb[1] - index[1])
                        target_zoom = 1.0 + max(0.0, min(1.5, (dist_pinch - 35) / 110.0))
                        
                        # Posisi pusat zoom mengikuti titik tengah antara jempol & telunjuk tangan!
                        target_cx = (thumb[0] + index[0]) // 2
                        target_cy = (thumb[1] + index[1]) // 2
    
                        if target_zoom > 1.05:
                            conn_layer = img.copy()
                            cv2.line(conn_layer, thumb, index, (20, 20, 210), 1, cv2.LINE_AA)
                            img = cv2.addWeighted(img, 0.65, conn_layer, 0.35, 0)
    
            filter_step = 0
    
            if len(results.hand_landmarks) >= 2:
                pt0 = index_tips[0]
                pt1 = index_tips[1]
                dist_tips = math.hypot(pt0[0]-pt1[0], pt0[1]-pt1[1])
                if dist_tips < 55:
                    mid_x = (pt0[0] + pt1[0]) // 2
                    mid_y = (pt0[1] + pt1[1]) // 2 - 25
                    cv2.putText(img, "⚡ SPIDER-SENSE ⚡", (mid_x - 65, mid_y),
                                cv2.FONT_HERSHEY_DUPLEX, 0.42, (200, 30, 45), 1, cv2.LINE_AA)
                if dist_tips < 40:
                    filter_step = 1
                    
            for i, hand_lms in enumerate(results.hand_landmarks):
                thumb = hand_lms[4]
                pinky = hand_lms[20]
                ring  = hand_lms[16]
                tx, ty = int(thumb.x * w), int(thumb.y * h)
                px, py = int(pinky.x * w), int(pinky.y * h)
                rx, ry = int(ring.x * w), int(ring.y * h)
                
                # Ganti filter: Tangan Kanan = NEXT (+1), Tangan Kiri = PREVIOUS (-1)
                if math.hypot(tx - px, ty - py) < 40 or math.hypot(tx - rx, ty - ry) < 40:
                    hand_label = "Right"
                    if results.handedness and i < len(results.handedness):
                        hand_label = results.handedness[i][0].category_name
                    
                    if hand_label == "Right":
                        filter_step = 1   # Tangan Kanan = NEXT FILTER
                    else:
                        filter_step = -1  # Tangan Kiri = PREVIOUS FILTER
                    
            if filter_step != 0:
                if not gesture_triggered:
                    previous_filter = current_filter
                    current_filter = (current_filter + filter_step) % len(filters)
                    gesture_triggered = True
                    play_sound_touch()
            else:
                gesture_triggered = False
    
            if len(pts_portal) == 4:
                pts_portal.sort(key=lambda p: p[1])
                top_pts = pts_portal[:2]
                bottom_pts = pts_portal[2:]
                top_pts.sort(key=lambda p: p[0])
                bottom_pts.sort(key=lambda p: p[0])
                
                poly_pts = np.array([top_pts[0], top_pts[1], bottom_pts[1], bottom_pts[0]], dtype=np.int32)
                
                x, y, bw, bh = cv2.boundingRect(poly_pts)
                x, y = max(0, x), max(0, y)
                bw, bh = min(w - x, bw), min(h - y, bh)
                
                if bw > 0 and bh > 0:
                    roi = img[y:y+bh, x:x+bw].copy()
                    filtered_roi = apply_filter(roi, filter_name, x, y)
                    
                    mask = np.zeros((bh, bw), dtype=np.float32)
                    poly_roi = poly_pts - [x, y]
                    cv2.fillPoly(mask, [poly_roi], 1.0)
                    mask_blurred = cv2.GaussianBlur(mask, (7, 7), 0)
                    mask_3ch = cv2.merge([mask_blurred, mask_blurred, mask_blurred])
                    
                    blended = roi.astype(np.float32) * (1.0 - mask_3ch) + filtered_roi.astype(np.float32) * mask_3ch
                    img[y:y+bh, x:x+bw] = np.clip(blended, 0, 255).astype(np.uint8)
                    
                    if sci_fi_glow:
                        glow_layer = np.zeros_like(img)
                        cv2.polylines(glow_layer, [poly_pts], True, (10, 10, 25), 18, cv2.LINE_AA)
                        cv2.polylines(glow_layer, [poly_pts], True, (20, 20, 240), 8, cv2.LINE_AA)
                        cv2.polylines(glow_layer, [poly_pts], True, (255, 255, 255), 2, cv2.LINE_AA)
                        glow_blurred = cv2.GaussianBlur(glow_layer, (21, 21), 0)
                        img = cv2.addWeighted(img, 1.0, glow_blurred, 1.0, 0)
                    
                    cv2.polylines(img, [poly_pts], True, (255, 255, 255), 2, cv2.LINE_AA)
    
        # --- SMOOTH ZOOM & TRACKING INTERPOLATION ---
        if zoom_locked:
            pass # Tahan posisi zoom & lokasi tetap terkunci
        elif zoom_mode and target_zoom > 1.02:
            current_zoom = current_zoom * 0.82 + target_zoom * 0.18
            zoom_cx = zoom_cx * 0.82 + target_cx * 0.18
            zoom_cy = zoom_cy * 0.82 + target_cy * 0.18
        else:
            current_zoom = current_zoom * 0.82 + 1.0 * 0.18
            zoom_cx = zoom_cx * 0.85 + (w // 2) * 0.15
            zoom_cy = zoom_cy * 0.85 + (h // 2) * 0.15
    
        # --- EKSEKUSI KAMERA ZOOM & TRACKING TANGAN BILA ZOOM > 1.02 ---
        if current_zoom > 1.02:
            crop_w = int(w / current_zoom)
            crop_h = int(h / current_zoom)
            
            half_w = crop_w // 2
            half_h = crop_h // 2
            
            # Batasi pusat zoom agar area crop tetap berada di dalam frame kamera
            cx_clamped = max(half_w, min(w - half_w, int(zoom_cx)))
            cy_clamped = max(half_h, min(h - half_h, int(zoom_cy)))
            
            x1 = cx_clamped - half_w
            y1 = cy_clamped - half_h
            x2 = x1 + crop_w
            y2 = y1 + crop_h
            
            cropped_roi = img[y1:y2, x1:x2]
            img = cv2.resize(cropped_roi, (w, h), interpolation=cv2.INTER_LINEAR)
            
            # Badge Zoom Indicator Venom di Kanan Atas
            if zoom_locked:
                cv2.putText(img, f"VENOM LOCKED {current_zoom:.1f}x", (w - 230, 40),
                            cv2.FONT_HERSHEY_DUPLEX, 0.45, (20, 20, 240), 1, cv2.LINE_AA)
            else:
                cv2.putText(img, f"VENOM ZOOM {current_zoom:.1f}x", (w - 210, 40),
                            cv2.FONT_HERSHEY_DUPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    
        # --- MENU TOMBOL BAWAH (VENOM SYMBIOTE HUD) & DUAL COMBO ---
        if metal_count >= 2:
            show_hud_buttons = False
            curr_t = time.time()
            if curr_t - last_combo_time > 1.8:
                last_combo_time = curr_t
                if "MULTIVERSE-GLITCH" in filters:
                    current_filter = filters.index("MULTIVERSE-GLITCH")
                play_sound_roar()
                snap_notification = "💥 DUAL COMBO: MULTIVERSE GLITCH ACTIVATED! ⚡"
                snap_notification_time = curr_t + 2.5
        elif metal_count == 1:
            show_hud_buttons = True
    
        if show_hud_buttons:
            zoom_title = "ZOOM (LOCKED)" if zoom_locked else "VENOM ZOOM"
            buttons = [
                {"id": "glow",  "title": "GLOW",     "is_on": sci_fi_glow,       "color": (20, 20, 220)},
                {"id": "zoom",  "title": zoom_title, "is_on": zoom_mode,         "color": (20, 20, 220) if zoom_locked else (255, 255, 255)},
                {"id": "telem", "title": "DIAGS",    "is_on": show_telemetry_hud, "color": (40, 200, 120)},
                {"id": "next",  "title": f"{filter_name}", "is_on": None,       "color": (200, 200, 220)}
            ]
            
            btn_w, btn_h = 145, 42
            gap = 14
            total_w = len(buttons) * btn_w + (len(buttons) - 1) * gap
            start_x = max(10, (w - total_w) // 2)
            start_y = h - 65
            
            pad_x, pad_y = 14, 10
            rx1, ry1 = start_x - pad_x, start_y - pad_y
            rx2, ry2 = start_x + total_w + pad_x, start_y + btn_h + pad_y
            r_bar = 16
            
            bar_overlay = img.copy()
            cv2.rectangle(bar_overlay, (rx1 + r_bar, ry1), (rx2 - r_bar, ry2), (8, 6, 14), -1)
            cv2.rectangle(bar_overlay, (rx1, ry1 + r_bar), (rx2, ry2 - r_bar), (8, 6, 14), -1)
            cv2.circle(bar_overlay, (rx1 + r_bar, ry1 + r_bar), r_bar, (8, 6, 14), -1, cv2.LINE_AA)
            cv2.circle(bar_overlay, (rx2 - r_bar, ry1 + r_bar), r_bar, (8, 6, 14), -1, cv2.LINE_AA)
            cv2.circle(bar_overlay, (rx1 + r_bar, ry2 - r_bar), r_bar, (8, 6, 14), -1, cv2.LINE_AA)
            cv2.circle(bar_overlay, (rx2 - r_bar, ry2 - r_bar), r_bar, (8, 6, 14), -1, cv2.LINE_AA)
            img = cv2.addWeighted(bar_overlay, 0.85, img, 0.15, 0)
            
            curr_time = time.time()
            
            for idx, btn in enumerate(buttons):
                bx1 = start_x + idx * (btn_w + gap)
                by1 = start_y
                bx2 = bx1 + btn_w
                by2 = by1 + btn_h
                
                touched = False
                for tip in index_tips:
                    if bx1 <= tip[0] <= bx2 and by1 <= tip[1] <= by2:
                        touched = True
                        break
                
                if touched and (curr_time - last_touch_time > touch_cooldown):
                    last_touch_time = curr_time
                    play_sound_touch()
                    if btn["id"] == "glow":
                        sci_fi_glow = not sci_fi_glow
                    elif btn["id"] == "zoom":
                        if zoom_locked:
                            zoom_locked = False
                            play_sound_unlock()
                        else:
                            zoom_mode = not zoom_mode
                    elif btn["id"] == "telem":
                        show_telemetry_hud = not show_telemetry_hud
                    elif btn["id"] == "next":
                        current_filter = (current_filter + 1) % len(filters)
    
                # Render Pill Card Tombol (Venom Symbiote Dark Style)
                r_pill = 12
                btn_bg = btn["color"] if touched else (18, 14, 28)
                btn_border = (255, 255, 255) if touched else (20, 20, 200)
                
                cv2.rectangle(img, (bx1 + r_pill, by1), (bx2 - r_pill, by2), btn_bg, -1)
                cv2.rectangle(img, (bx1, by1 + r_pill), (bx2, by2 - r_pill), btn_bg, -1)
                cv2.circle(img, (bx1 + r_pill, by1 + r_pill), r_pill, btn_bg, -1, cv2.LINE_AA)
                cv2.circle(img, (bx2 - r_pill, by1 + r_pill), r_pill, btn_bg, -1, cv2.LINE_AA)
                cv2.circle(img, (bx1 + r_pill, by2 - r_pill), r_pill, btn_bg, -1, cv2.LINE_AA)
                cv2.circle(img, (bx2 - r_pill, by2 - r_pill), r_pill, btn_bg, -1, cv2.LINE_AA)
                
                cv2.line(img, (bx1 + r_pill, by1), (bx2 - r_pill, by1), btn_border, 1, cv2.LINE_AA)
                cv2.line(img, (bx1 + r_pill, by2), (bx2 - r_pill, by2), btn_border, 1, cv2.LINE_AA)
                cv2.line(img, (bx1, by1 + r_pill), (bx1, by2 - r_pill), btn_border, 1, cv2.LINE_AA)
                cv2.line(img, (bx2, by1 + r_pill), (bx2, by2 - r_pill), btn_border, 1, cv2.LINE_AA)
                cv2.ellipse(img, (bx1 + r_pill, by1 + r_pill), (r_pill, r_pill), 180, 0, 90, btn_border, 1, cv2.LINE_AA)
                cv2.ellipse(img, (bx2 - r_pill, by1 + r_pill), (r_pill, r_pill), 270, 0, 90, btn_border, 1, cv2.LINE_AA)
                cv2.ellipse(img, (bx1 + r_pill, by2 - r_pill), (r_pill, r_pill), 90, 0, 90, btn_border, 1, cv2.LINE_AA)
                cv2.ellipse(img, (bx2 - r_pill, by2 - r_pill), (r_pill, r_pill), 0, 0, 90, btn_border, 1, cv2.LINE_AA)
    
                text_color = (0, 0, 0) if touched else (240, 240, 255)
                
                if btn["is_on"] is not None:
                    dot_color = (20, 20, 240) if btn["is_on"] else (60, 60, 75)
                    cv2.circle(img, (bx1 + 18, by1 + 21), 4, dot_color, -1, cv2.LINE_AA)
                    if btn["is_on"]:
                        cv2.circle(img, (bx1 + 18, by1 + 21), 6, (255, 255, 255), 1, cv2.LINE_AA)
                    cv2.putText(img, btn["title"], (bx1 + 30, by1 + 26), 
                                cv2.FONT_HERSHEY_DUPLEX, 0.38, text_color, 1, cv2.LINE_AA)
                else:
                    cv2.putText(img, btn["title"], (bx1 + 14, by1 + 26), 
                                cv2.FONT_HERSHEY_DUPLEX, 0.38, text_color, 1, cv2.LINE_AA)
    
        # --- STREAMER BROADCAST STATUS BADGE (Kiri Atas) ---
        curr_frame_time = time.time()
        dt = curr_frame_time - last_fps_time
        last_fps_time = curr_frame_time
        if dt > 0:
            fps_ema = fps_ema * 0.9 + (1.0 / dt) * 0.1
    
        badge_text = f"LIVE | {fps_ema:.0f} FPS | OBS VCAM"
        cv2.rectangle(img, (15, 15), (225, 45), (10, 8, 18), -1)
        cv2.rectangle(img, (15, 15), (225, 45), (35, 30, 60), 1, cv2.LINE_AA)
        cv2.circle(img, (30, 30), 4, (20, 20, 240), -1, cv2.LINE_AA)
        cv2.putText(img, badge_text, (42, 34), cv2.FONT_HERSHEY_DUPLEX, 0.4, (240, 240, 255), 1, cv2.LINE_AA)
    
        # Filter Counter Badge (kanan atas)
        counter_text = f"{filter_name}  [{current_filter + 1}/{len(filters)}]"
        ct_size = cv2.getTextSize(counter_text, cv2.FONT_HERSHEY_DUPLEX, 0.38, 1)[0]
        ct_x = w - ct_size[0] - 25
        cv2.rectangle(img, (ct_x - 10, 15), (w - 10, 45), (10, 8, 18), -1)
        cv2.rectangle(img, (ct_x - 10, 15), (w - 10, 45), (35, 30, 60), 1, cv2.LINE_AA)
        cv2.putText(img, counter_text, (ct_x, 34), cv2.FONT_HERSHEY_DUPLEX, 0.38, (220, 220, 240), 1, cv2.LINE_AA)
    
        # Render Notification Snapshot jika ada
        if time.time() < snap_notification_time:
            cv2.rectangle(img, (15, 52), (360, 78), (10, 25, 18), -1)
            cv2.rectangle(img, (15, 52), (360, 78), (40, 220, 100), 1, cv2.LINE_AA)
            cv2.putText(img, snap_notification, (24, 69), cv2.FONT_HERSHEY_DUPLEX, 0.38, (0, 255, 150), 1, cv2.LINE_AA)
    
        # Flash Vignette Putih saat Air-Snap
        if snap_flash_alpha > 0.01:
            flash_layer = np.full_like(img, 255)
            img = cv2.addWeighted(flash_layer, snap_flash_alpha, img, 1.0 - snap_flash_alpha, 0)
            snap_flash_alpha *= 0.55
    
        # Render Video Clip Recording Overlay jika sedang merekam ('r' / ✊)
        if is_recording:
            rec_elapsed = time.time() - rec_start_time
            rec_remaining = max(0.0, rec_max_duration - rec_elapsed)
            
            if rec_writer is not None:
                rec_writer.write(img)
                
            pulse = int(127 + 128 * math.sin(time.time() * 12))
            cv2.rectangle(img, (w - 150, 15), (w - 15, 45), (10, 8, 18), -1)
            cv2.rectangle(img, (w - 150, 15), (w - 15, 45), (20, 20, 220), 1, cv2.LINE_AA)
            cv2.circle(img, (w - 132, 30), 5, (0, 0, pulse), -1, cv2.LINE_AA)
            cv2.putText(img, f"REC {rec_remaining:.1f}s", (w - 118, 34),
                        cv2.FONT_HERSHEY_DUPLEX, 0.4, (240, 240, 255), 1, cv2.LINE_AA)
            
            if rec_elapsed >= rec_max_duration:
                stop_recording()
                snap_notification = f"CLIP SAVED -> {os.path.basename(rec_filename)}"
                snap_notification_time = time.time() + 3.0
    
        # Render Telemetry HUD jika diaktifkan ('d')
        if show_telemetry_hud:
            hands_cnt = len(results.hand_landmarks) if results and results.hand_landmarks else 0
            img = draw_telemetry_hud(img, w, h, hands_cnt, fps_ema)
    
        # Render Glassmorphism Help Overlay jika diaktifkan ('h')
        if show_help_overlay:
            img = draw_help_overlay(img, w, h)
    
        # Output ke Virtual Camera
        if vcam is not None:
            frame_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            vcam.send(frame_rgb)
    
        cv2.imshow('RETROLENS Pake Python', img)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == 27: # ESC key
            if show_help_overlay:
                show_help_overlay = False
            else:
                break
        # Check if Air-Snap was requested from Control Panel GUI
        if gui_snap_requested:
            gui_snap_requested = False
            snap_filename = f"captures/Venom_Capture_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(snap_filename, img)
            play_sound_snap()
            snap_notification = f"PHOTO SAVED -> {os.path.basename(snap_filename)}"
            snap_notification_time = time.time() + 2.5
            snap_flash_alpha = 0.65

        elif key == ord('s'):
            snap_filename = f"captures/Venom_Capture_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(snap_filename, img)
            play_sound_snap()
            snap_notification = f"PHOTO SAVED -> {os.path.basename(snap_filename)}"
            snap_notification_time = time.time() + 2.5
            snap_flash_alpha = 0.65
        elif key == ord('c'):
            if gui and gui.root:
                try:
                    gui.root.deiconify()
                    gui.root.lift()
                except Exception:
                    pass
            play_sound_touch()
        elif key == ord('r'):
            if is_recording:
                stop_recording()
                snap_notification = f"CLIP SAVED -> {os.path.basename(rec_filename)}"
                snap_notification_time = time.time() + 3.0
            else:
                start_recording(img.shape)
        elif key == ord('d'):
            show_telemetry_hud = not show_telemetry_hud
            play_sound_touch()
        elif key == ord('m'):
            show_hud_buttons = not show_hud_buttons
            play_sound_touch()
        elif key == ord('h'):
            show_help_overlay = not show_help_overlay
            play_sound_touch()
        elif key == ord('1'):
            current_filter = filters.index("VENOM-VISION")
            snap_notification = "THEME ACTIVATED -> VENOM-VISION 🕷️"
            snap_notification_time = time.time() + 1.8
            play_sound_roar()
        elif key == ord('2'):
            current_filter = filters.index("SPIDER-MAN")
            snap_notification = "THEME ACTIVATED -> SPIDER-MAN 🕸️"
            snap_notification_time = time.time() + 1.8
            play_sound_thwip()
        elif key == ord('3'):
            current_filter = filters.index("SYMBIOTE-VORTEX")
            snap_notification = "THEME ACTIVATED -> SYMBIOTE-VORTEX ⚡"
            snap_notification_time = time.time() + 1.8
            play_sound_touch()
        elif key == ord('4'):
            current_filter = filters.index("MULTIVERSE-GLITCH")
            snap_notification = "THEME ACTIVATED -> MULTIVERSE-GLITCH 💥"
            snap_notification_time = time.time() + 1.8
            play_sound_glitch()
        elif key == ord('5'):
            current_filter = filters.index("ANTI-VENOM")
            snap_notification = "THEME ACTIVATED -> ANTI-VENOM 🤍"
            snap_notification_time = time.time() + 1.8
            play_sound_touch()
        elif key == ord('6'):
            current_filter = filters.index("SPIDER-VENOM-UNITE")
            snap_notification = "THEME ACTIVATED -> SPIDER-VENOM-UNITE 🕸️🕷️"
            snap_notification_time = time.time() + 1.8
            play_sound_thwip()
        elif key == ord('7'):
            current_filter = filters.index("CYBER-VORTEX")
            snap_notification = "THEME ACTIVATED -> CYBER-VORTEX ⚡🌀"
            snap_notification_time = time.time() + 1.8
            play_sound_glitch()
        elif key == ord('8'):
            current_filter = filters.index("CARNAGE-VORTEX")
            snap_notification = "THEME ACTIVATED -> CARNAGE-VORTEX 🩸🌀"
            snap_notification_time = time.time() + 1.8
            play_sound_roar()
        elif key == ord('9'):
            current_filter = filters.index("NIGHT-VISION")
            snap_notification = "THEME ACTIVATED -> NIGHT-VISION 🔭"
            snap_notification_time = time.time() + 1.8
            play_sound_touch()
        elif key == ord('0'):
            current_filter = filters.index("HOLOGRAM")
            snap_notification = "THEME ACTIVATED -> HOLOGRAM 🔣"
            snap_notification_time = time.time() + 1.8
            play_sound_glitch()

    camera.stop()
    if vcam is not None:
        vcam.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_app()