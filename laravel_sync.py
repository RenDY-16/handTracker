"""
Retrolens Python to Laravel Bridge & Auto-Sync Client
Syncs local captures, recordings, and telemetry events with the Laravel Handtrack backend.
"""
import os
import requests
import json
import time

LARAVEL_BASE_URL = os.environ.get("LARAVEL_URL", "http://127.0.0.1:8000")

def sync_photo(filepath, filter_name="VENOM-VISION", gesture="peace"):
    """Uploads a local photo to Laravel backend API."""
    url = f"{LARAVEL_BASE_URL}/api/captures/photo"
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return False

    try:
        with open(filepath, 'rb') as f:
            files = {'file': (os.path.basename(filepath), f, 'image/jpeg')}
            data = {
                'filter_name': filter_name,
                'gesture_trigger': gesture,
                'resolution': '1920x1080',
            }
            res = requests.post(url, files=files, data=data, timeout=5)
            if res.status_code == 201:
                print(f"✅ Photo synced to Laravel: {res.json()['data']['filename']}")
                return True
    except Exception as e:
        print(f"⚠️ Laravel sync error: {e}")
    return False

def sync_video(filepath, filter_name="VENOM-VISION", duration=5.0, gesture="fist"):
    """Uploads a 5-second video clip to Laravel backend API."""
    url = f"{LARAVEL_BASE_URL}/api/captures/video"
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return False

    try:
        with open(filepath, 'rb') as f:
            files = {'video': (os.path.basename(filepath), f, 'video/mp4')}
            data = {
                'filter_name': filter_name,
                'duration': duration,
                'gesture_trigger': gesture,
                'resolution': '1920x1080',
            }
            res = requests.post(url, files=files, data=data, timeout=10)
            if res.status_code == 201:
                print(f"✅ Video clip synced to Laravel: {res.json()['data']['filename']}")
                return True
    except Exception as e:
        print(f"⚠️ Laravel sync error: {e}")
    return False

def send_telemetry_event(gesture, filter_name, fps=30.0, latency=16.0, hands_detected=1):
    """Sends telemetry data from Python to Laravel."""
    url = f"{LARAVEL_BASE_URL}/api/telemetry/log"
    payload = {
        'gesture': gesture,
        'filter_name': filter_name,
        'fps': fps,
        'latency': latency,
        'hands_detected': hands_detected,
        'source': 'python_client',
    }
    try:
        requests.post(url, json=payload, timeout=2)
    except Exception:
        pass

if __name__ == "__main__":
    print(f"🕷️ Testing connection to Laravel at {LARAVEL_BASE_URL}...")
    try:
        res = requests.get(f"{LARAVEL_BASE_URL}/api/telemetry/summary", timeout=3)
        if res.status_code == 200:
            print("🎉 Laravel Handtrack Backend is connected and online!")
            print("Summary:", res.json())
        else:
            print("⚠️ Backend returned:", res.status_code)
    except Exception as e:
        print(f"❌ Connection failed: {e}. Pastikan `php artisan serve` aktif.")
