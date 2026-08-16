import os
import cv2
import time
import math
import base64
import json
import numpy as np
import mediapipe as mp
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# Initialize FastAPI App
app = FastAPI(title="Retrolens AI Handtrack Cloud Studio")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- MediaPipe Task Initialization ---
MODEL_PATH = "hand_landmarker.task"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file '{MODEL_PATH}' not found in current directory.")

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=2,
    min_hand_detection_confidence=0.6,
    min_hand_presence_confidence=0.6,
    min_tracking_confidence=0.6
)
landmarker = HandLandmarker.create_from_options(options)

FILTERS = [
    "VENOM-VISION", "SPIDER-MAN", "SPIDER-VENOM-UNITE", "VENOM-CARNAGE",
    "CARNAGE-VORTEX", "SPIDER-2099", "CYBER-VORTEX", "SYMBIOTE-VORTEX",
    "SYMBIOTE-BURST", "MULTIVERSE-GLITCH", "NIGHT-VISION", "HOLOGRAM",
    "THERMAL-SCAN", "SKETCH-NOIR", "SYMBIOTE-RED", "SPIDER-NOIR",
    "TOXIC-SYMBIOTE", "ANTI-VENOM", "EMBOSS", "GLITCH", "NEON", "MONO",
    "PIXELATE", "INVERT"
]

def apply_filter(roi, filter_name):
    try:
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
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            h_r, w_r = roi.shape[:2]
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            hsv[:, :, 1] = cv2.add(hsv[:, :, 1], 45)
            vibrant = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            dot_spacing = 7
            grid_y, grid_x = np.mgrid[0:h_r, 0:w_r]
            dot_mask = ((grid_x % dot_spacing == 0) & (grid_y % dot_spacing == 0))
            edges = cv2.Canny(gray, 80, 160)
            vibrant[edges > 0] = [15, 10, 25]
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
            filtered[mask_c == 255] = [20, 20, 240]
            filtered[mask_c == 0] = [10, 10, 15]
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
            filtered[mask_c == 255] = [40, 240, 60]
            filtered[mask_c == 0] = [10, 10, 15]
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
        elif filter_name == "PIXELATE":
            h_r, w_r = roi.shape[:2]
            if h_r > 10 and w_r > 10:
                small = cv2.resize(roi, (w_r//10, h_r//10), interpolation=cv2.INTER_LINEAR)
                return cv2.resize(small, (w_r, h_r), interpolation=cv2.INTER_NEAREST)
        elif filter_name == "NIGHT-VISION":
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            bright = cv2.addWeighted(gray, 1.6, gray, 0, 15)
            green_frame = np.zeros_like(roi)
            green_frame[:, :, 1] = bright
            green_frame[:, :, 0] = (bright * 0.15).astype(np.uint8)
            noise = np.random.randint(0, 12, gray.shape, dtype=np.uint8)
            green_frame[:, :, 1] = cv2.add(green_frame[:, :, 1], noise)
            return green_frame
        elif filter_name == "HOLOGRAM":
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 40, 120)
            holo = np.zeros_like(roi)
            holo[:, :, 0] = (edges * 0.9).astype(np.uint8)
            holo[:, :, 1] = (edges * 0.4).astype(np.uint8)
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
            tint = np.full_like(sketch_bgr, (15, 12, 25))
            return cv2.addWeighted(sketch_bgr, 0.85, tint, 0.15, 0)
        elif filter_name == "EMBOSS":
            kernel = np.array([[-2, -1, 0],
                               [-1,  1, 1],
                               [ 0,  1, 2]])
            embossed = cv2.filter2D(roi, -1, kernel) + 128
            return cv2.addWeighted(roi, 0.25, embossed, 0.75, 0)
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
            edges_bgr[np.where((edges_bgr == [255, 255, 255]).all(axis=2))] = [20, 20, 240]
            kernel = np.ones((3,3), np.uint8)
            return cv2.dilate(edges_bgr, kernel, iterations=1)
    except Exception:
        pass
    return roi

def detect_gestures(hand_landmarks_list, w, h):
    detected = []
    for hand_lms in hand_landmarks_list:
        wrist = hand_lms[0]
        def d_wrist(idx):
            return math.hypot((hand_lms[idx].x - wrist.x) * w, (hand_lms[idx].y - wrist.y) * h)
        
        # 1. Peace (✌️)
        index_open   = d_wrist(8)  > d_wrist(6) * 1.15
        middle_open  = d_wrist(12) > d_wrist(10) * 1.15
        ring_closed  = d_wrist(16) < d_wrist(14) * 1.1
        pinky_closed = d_wrist(20) < d_wrist(18) * 1.1
        pt8 = (hand_lms[8].x * w, hand_lms[8].y * h)
        pt12 = (hand_lms[12].x * w, hand_lms[12].y * h)
        fingers_separated = math.hypot(pt8[0] - pt12[0], pt8[1] - pt12[1]) > 20
        if index_open and middle_open and ring_closed and pinky_closed and fingers_separated:
            detected.append("peace")
            
        # 2. Fist (✊)
        fist = True
        for tip_id, mcp_id in [(4, 2), (8, 5), (12, 9), (16, 13), (20, 17)]:
            if d_wrist(tip_id) > d_wrist(mcp_id) * 1.05:
                fist = False
                break
        if fist:
            detected.append("fist")
            
        # 3. Metal (🤟)
        idx_open = d_wrist(8) > d_wrist(6) * 1.12
        pky_open = d_wrist(20) > d_wrist(18) * 1.12
        mid_closed = d_wrist(12) < d_wrist(10) * 1.1
        rng_closed = d_wrist(16) < d_wrist(14) * 1.1
        if idx_open and pky_open and mid_closed and rng_closed:
            detected.append("metal")
            
        # 4. OK Sign (👌)
        thumb = hand_lms[4]
        index = hand_lms[8]
        pinch_dist = math.hypot((thumb.x - index.x) * w, (thumb.y - index.y) * h)
        if pinch_dist < 32 and middle_open and ring_closed and pinky_closed:
            detected.append("ok")
            
        # 5. Claw (🖐️)
        curled = True
        for tip_id, mcp_id in [(8, 5), (12, 9), (16, 13), (20, 17)]:
            dist_tip = d_wrist(tip_id)
            dist_mcp = d_wrist(mcp_id)
            if dist_tip > dist_mcp * 1.35 or dist_tip < dist_mcp * 0.8:
                curled = False
                break
        if curled:
            detected.append("claw")
            
    return detected

# --- Endpoints ---

@app.get("/api/filters")
def get_filters():
    return {"status": "success", "filters": FILTERS}

@app.get("/api/health")
def health_check():
    return {"status": "ok", "service": "Retrolens Handtrack AI", "model": "MediaPipe Task"}

@app.websocket("/ws/process")
async def websocket_process(websocket: WebSocket):
    await websocket.accept()
    last_filter = "VENOM-VISION"
    
    try:
        while True:
            data_text = await websocket.receive_text()
            req = json.loads(data_text)
            
            img_b64 = req.get("image", "")
            req_filter = req.get("filter", last_filter)
            if req_filter in FILTERS:
                last_filter = req_filter
                
            if not img_b64:
                continue
                
            start_t = time.time()
            
            # Decode Image
            header, encoded = img_b64.split(",", 1) if "," in img_b64 else ("", img_b64)
            img_bytes = base64.b64decode(encoded)
            np_arr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            if frame is None:
                continue
                
            h, w = frame.shape[:2]
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            
            # Run Hand Tracking
            results = landmarker.detect(mp_image)
            hands_count = len(results.hand_landmarks) if results.hand_landmarks else 0
            gestures = detect_gestures(results.hand_landmarks, w, h) if hands_count > 0 else []
            
            # Auto gesture actions
            if "claw" in gestures:
                last_filter = "VENOM-VISION"
            elif "metal" in gestures and len(gestures) >= 2:
                last_filter = "MULTIVERSE-GLITCH"
                
            # Apply Filter
            processed = apply_filter(frame, last_filter)
            
            # Draw Hand Skeleton Landmark Overlay
            if results.hand_landmarks:
                for hand_lms in results.hand_landmarks:
                    for lm in hand_lms:
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        cv2.circle(processed, (cx, cy), 3, (240, 20, 20), -1)
                        cv2.circle(processed, (cx, cy), 5, (255, 255, 255), 1)
                        
            # Encode back to JPEG
            _, buffer = cv2.imencode('.jpg', processed, [cv2.IMWRITE_JPEG_QUALITY, 80])
            out_b64 = "data:image/jpeg;base64," + base64.b64encode(buffer).decode('utf-8')
            
            latency = (time.time() - start_t) * 1000.0
            fps = 1000.0 / max(1.0, latency)
            
            res_payload = {
                "image": out_b64,
                "filter": last_filter,
                "hands": hands_count,
                "gestures": gestures,
                "latency": round(latency, 1),
                "fps": round(fps, 1)
            }
            
            await websocket.send_text(json.dumps(res_payload))
            
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"WS Error: {e}")

@app.get("/", response_class=HTMLResponse)
def index_page():
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🕷️ Retrolens - Venom Handtrack AI Studio</title>
    <link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700;900&family=Rajdhani:wght@500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --primary: #ff0044;
            --primary-glow: rgba(255, 0, 68, 0.4);
            --cyan: #00f0ff;
            --cyan-glow: rgba(0, 240, 255, 0.3);
            --bg-dark: #080811;
            --card-bg: rgba(16, 14, 28, 0.75);
            --border-glass: rgba(255, 255, 255, 0.1);
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            background-color: var(--bg-dark);
            color: #f0f0ff;
            font-family: 'Rajdhani', sans-serif;
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            overflow-x: hidden;
            background-image: radial-gradient(circle at 50% 20%, rgba(255, 0, 68, 0.08) 0%, transparent 60%),
                              radial-gradient(circle at 90% 80%, rgba(0, 240, 255, 0.05) 0%, transparent 50%);
        }
        header {
            padding: 14px 24px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            border-bottom: 1px solid var(--border-glass);
            background: rgba(8, 8, 17, 0.85);
            backdrop-filter: blur(12px);
            z-index: 50;
        }
        .brand {
            font-family: 'Orbitron', sans-serif;
            font-size: 20px;
            font-weight: 900;
            display: flex;
            align-items: center;
            gap: 10px;
            letter-spacing: 1px;
        }
        .brand span { color: var(--primary); text-shadow: 0 0 12px var(--primary); }
        .status-badge {
            font-size: 13px;
            font-weight: 700;
            padding: 5px 14px;
            border-radius: 999px;
            background: rgba(0, 240, 255, 0.1);
            border: 1px solid var(--cyan);
            color: var(--cyan);
            display: flex;
            align-items: center;
            gap: 6px;
        }
        .status-dot { width: 8px; height: 8px; border-radius: 50%; background: var(--cyan); box-shadow: 0 0 8px var(--cyan); }

        main {
            flex: 1;
            display: grid;
            grid-template-columns: 1fr 340px;
            gap: 20px;
            padding: 20px;
            max-width: 1560px;
            margin: 0 auto;
            width: 100%;
        }
        @media (max-width: 1024px) {
            main { grid-template-columns: 1fr; }
        }

        .viewport-card {
            background: var(--card-bg);
            border: 1px solid var(--border-glass);
            border-radius: 16px;
            overflow: hidden;
            position: relative;
            display: flex;
            flex-direction: column;
            box-shadow: 0 10px 40px rgba(0,0,0,0.6);
        }
        .video-container {
            position: relative;
            width: 100%;
            background: #000;
            aspect-ratio: 16 / 9;
            display: flex;
            align-items: center;
            justify-content: center;
            overflow: hidden;
        }
        #processedCanvas, #rawVideo {
            width: 100%;
            height: 100%;
            object-fit: cover;
            transform: scaleX(-1);
        }
        #rawVideo { display: none; }

        .telemetry-overlay {
            position: absolute;
            top: 14px;
            left: 14px;
            background: rgba(8, 8, 17, 0.82);
            border: 1px solid var(--border-glass);
            padding: 10px 16px;
            border-radius: 10px;
            backdrop-filter: blur(8px);
            font-family: 'Orbitron', monospace;
            font-size: 11px;
            display: flex;
            flex-direction: column;
            gap: 4px;
            pointer-events: none;
        }
        .telemetry-overlay .val { color: var(--cyan); font-weight: 700; }
        .telemetry-overlay .alert { color: var(--primary); font-weight: 700; }

        .gesture-pill {
            position: absolute;
            top: 14px;
            right: 14px;
            background: rgba(255, 0, 68, 0.2);
            border: 1px solid var(--primary);
            color: #fff;
            padding: 8px 16px;
            border-radius: 999px;
            font-family: 'Orbitron', sans-serif;
            font-size: 12px;
            font-weight: 700;
            box-shadow: 0 0 16px var(--primary-glow);
            opacity: 0;
            transition: opacity 0.2s;
        }
        .gesture-pill.active { opacity: 1; }

        .quick-actions {
            padding: 14px 18px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            background: rgba(10, 8, 20, 0.95);
            border-top: 1px solid var(--border-glass);
            gap: 10px;
            flex-wrap: wrap;
        }
        .action-btn {
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid var(--border-glass);
            color: #f0f0ff;
            padding: 10px 18px;
            border-radius: 10px;
            font-family: 'Orbitron', sans-serif;
            font-size: 12px;
            font-weight: 700;
            cursor: pointer;
            display: flex;
            align-items: center;
            gap: 8px;
            transition: all 0.2s ease;
        }
        .action-btn:hover {
            border-color: var(--cyan);
            background: rgba(0, 240, 255, 0.15);
            box-shadow: 0 0 14px var(--cyan-glow);
        }
        .action-btn.primary {
            background: var(--primary);
            border-color: var(--primary);
        }
        .action-btn.primary:hover {
            background: #ff1f5a;
            box-shadow: 0 0 16px var(--primary-glow);
        }

        .sidebar {
            background: var(--card-bg);
            border: 1px solid var(--border-glass);
            border-radius: 16px;
            padding: 18px;
            display: flex;
            flex-direction: column;
            gap: 16px;
            height: calc(100vh - 120px);
            overflow-y: auto;
        }
        .sidebar h3 {
            font-family: 'Orbitron', sans-serif;
            font-size: 14px;
            letter-spacing: 1px;
            color: var(--cyan);
            text-transform: uppercase;
            padding-bottom: 8px;
            border-bottom: 1px solid var(--border-glass);
        }
        .filter-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 8px;
        }
        .filter-chip {
            background: rgba(255, 255, 255, 0.04);
            border: 1px solid var(--border-glass);
            color: #d0d0ea;
            padding: 10px 8px;
            border-radius: 8px;
            font-family: 'Rajdhani', sans-serif;
            font-size: 13px;
            font-weight: 700;
            text-align: center;
            cursor: pointer;
            transition: all 0.2s;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        .filter-chip:hover {
            border-color: var(--cyan);
            color: #fff;
            background: rgba(0, 240, 255, 0.1);
        }
        .filter-chip.active {
            background: linear-gradient(135deg, var(--primary), #a8002d);
            border-color: var(--primary);
            color: #fff;
            box-shadow: 0 0 14px var(--primary-glow);
        }

        .guide-box {
            background: rgba(0, 0, 0, 0.3);
            border-radius: 10px;
            padding: 12px;
            font-size: 13px;
            line-height: 1.6;
        }
        .guide-box li { margin-left: 18px; margin-bottom: 4px; }
        .guide-box strong { color: var(--cyan); }
    </style>
</head>
<body>
    <header>
        <div class="brand">
            🕷️ RETROLENS <span>STUDIO</span>
        </div>
        <div class="status-badge">
            <div class="status-dot"></div>
            <span id="connStatus">CONNECTING AI ENGINE...</span>
        </div>
    </header>

    <main>
        <div class="viewport-card">
            <div class="video-container">
                <video id="rawVideo" autoplay playsinline muted></video>
                <img id="processedCanvas" alt="AI Filter Stream">
                
                <div class="telemetry-overlay">
                    <div>LATENCY: <span id="telLatency" class="val">--</span> ms</div>
                    <div>AI FPS: <span id="telFps" class="val">--</span></div>
                    <div>HANDS: <span id="telHands" class="val">0/2</span></div>
                    <div>FILTER: <span id="telFilter" class="alert">VENOM-VISION</span></div>
                </div>

                <div id="gestureIndicator" class="gesture-pill">✌️ AIR-SNAP TRIGGERED!</div>
            </div>

            <div class="quick-actions">
                <button class="action-btn primary" id="snapBtn">📸 AIR-SNAP CAPTURE</button>
                <button class="action-btn" id="nextFilterBtn">⏭️ NEXT FILTER</button>
                <button class="action-btn" id="toggleCamBtn">🎥 RESTART CAMERA</button>
            </div>
        </div>

        <div class="sidebar">
            <h3>🎨 24 SHADER FILTERS</h3>
            <div class="filter-grid" id="filterList"></div>

            <h3>🖐️ AI GESTURE CONTROLS</h3>
            <div class="guide-box">
                <ul>
                    <li>✌️ <strong>Peace:</strong> Air-Snap Photo Capture</li>
                    <li>✊ <strong>Fist:</strong> Air-Clip Video Recording</li>
                    <li>🖐️ <strong>Claw:</strong> Instant Venom-Vision</li>
                    <li>🤟 <strong>Metal:</strong> Multiverse Glitch Combo</li>
                    <li>👌 <strong>OK Sign:</strong> Lock / Unlock Zoom</li>
                </ul>
            </div>
        </div>
    </main>

    <script>
        const FILTERS = [
            "VENOM-VISION", "SPIDER-MAN", "SPIDER-VENOM-UNITE", "VENOM-CARNAGE",
            "CARNAGE-VORTEX", "SPIDER-2099", "CYBER-VORTEX", "SYMBIOTE-VORTEX",
            "SYMBIOTE-BURST", "MULTIVERSE-GLITCH", "NIGHT-VISION", "HOLOGRAM",
            "THERMAL-SCAN", "SKETCH-NOIR", "SYMBIOTE-RED", "SPIDER-NOIR",
            "TOXIC-SYMBIOTE", "ANTI-VENOM", "EMBOSS", "GLITCH", "NEON", "MONO",
            "PIXELATE", "INVERT"
        ];
        let currentFilter = "VENOM-VISION";
        let ws = null;
        let isSending = false;

        const rawVideo = document.getElementById('rawVideo');
        const processedImg = document.getElementById('processedCanvas');
        const connStatus = document.getElementById('connStatus');
        const telLatency = document.getElementById('telLatency');
        const telFps = document.getElementById('telFps');
        const telHands = document.getElementById('telHands');
        const telFilter = document.getElementById('telFilter');
        const gestureIndicator = document.getElementById('gestureIndicator');

        // Render Filter Grid
        const filterListEl = document.getElementById('filterList');
        FILTERS.forEach(f => {
            const btn = document.createElement('div');
            btn.className = `filter-chip ${f === currentFilter ? 'active' : ''}`;
            btn.innerText = f;
            btn.onclick = () => selectFilter(f);
            btn.id = `filter-${f}`;
            filterListEl.appendChild(btn);
        });

        function selectFilter(filterName) {
            currentFilter = filterName;
            document.querySelectorAll('.filter-chip').forEach(b => b.classList.remove('active'));
            const el = document.getElementById(`filter-${filterName}`);
            if (el) el.classList.add('active');
            telFilter.innerText = filterName;
        }

        document.getElementById('nextFilterBtn').onclick = () => {
            let idx = FILTERS.indexOf(currentFilter);
            let nextIdx = (idx + 1) % FILTERS.length;
            selectFilter(FILTERS[nextIdx]);
        };

        // Hidden canvas for frame extraction
        const offscreenCanvas = document.createElement('canvas');
        const offscreenCtx = offscreenCanvas.getContext('2d');

        async function startCamera() {
            try {
                const stream = await navigator.mediaDevices.getUserMedia({
                    video: { width: { ideal: 640 }, height: { ideal: 480 } },
                    audio: false
                });
                rawVideo.srcObject = stream;
                await rawVideo.play();
                offscreenCanvas.width = 640;
                offscreenCanvas.height = 480;
                initWebSocket();
            } catch (err) {
                alert("Gagal mengakses webcam: " + err.message);
            }
        }

        function initWebSocket() {
            const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const wsUrl = `${proto}//${window.location.host}/ws/process`;
            ws = new WebSocket(wsUrl);

            ws.onopen = () => {
                connStatus.innerText = "AI ONLINE (30 FPS)";
                sendFrameLoop();
            };

            ws.onmessage = (event) => {
                isSending = false;
                const data = JSON.parse(event.data);
                processedImg.src = data.image;
                telLatency.innerText = data.latency;
                telFps.innerText = data.fps;
                telHands.innerText = `${data.hands}/2`;
                if (data.filter && data.filter !== currentFilter) {
                    selectFilter(data.filter);
                }

                if (data.gestures && data.gestures.length > 0) {
                    gestureIndicator.innerText = `⚡ GESTURE: ${data.gestures.join(', ').toUpperCase()}`;
                    gestureIndicator.classList.add('active');
                } else {
                    gestureIndicator.classList.remove('active');
                }
            };

            ws.onclose = () => {
                connStatus.innerText = "DISCONNECTED. RECONNECTING...";
                setTimeout(initWebSocket, 2000);
            };
        }

        function sendFrameLoop() {
            if (ws && ws.readyState === WebSocket.OPEN && !isSending && rawVideo.videoWidth > 0) {
                isSending = true;
                offscreenCtx.drawImage(rawVideo, 0, 0, 640, 480);
                const frameData = offscreenCanvas.toDataURL('image/jpeg', 0.65);
                ws.send(JSON.stringify({
                    image: frameData,
                    filter: currentFilter
                }));
            }
            requestAnimationFrame(sendFrameLoop);
        }

        document.getElementById('snapBtn').onclick = () => {
            const a = document.createElement('a');
            a.href = processedImg.src;
            a.download = `Retrolens_AirSnap_${Date.now()}.jpg`;
            a.click();
        };

        document.getElementById('toggleCamBtn').onclick = startCamera;
        window.addEventListener('DOMContentLoaded', startCamera);
    </script>
</body>
</html>
    """
