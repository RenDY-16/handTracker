"""Test kamera index 2 (NVIDIA Broadcast) dengan OpenCV 4.10"""
import cv2
print(f"OpenCV version: {cv2.__version__}")

# Test index 2 dengan berbagai backend
for backend_name, backend in [("DEFAULT", None), ("DSHOW", cv2.CAP_DSHOW), ("MSMF", cv2.CAP_MSMF)]:
    if backend is None:
        cap = cv2.VideoCapture(2)
    else:
        cap = cv2.VideoCapture(2, backend)
    
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        for _ in range(10):
            cap.read()
        ret, frame = cap.read()
        if ret:
            h, w = frame.shape[:2]
            fname = f"cv4_test_{backend_name}.jpg"
            cv2.imwrite(fname, frame)
            print(f"  {backend_name}: {w}x{h} -> {fname}")
        else:
            print(f"  {backend_name}: Gagal baca frame")
        cap.release()
    else:
        print(f"  {backend_name}: Gagal buka")
