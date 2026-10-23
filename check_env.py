"""Kiểm tra môi trường: Python, GPU/CUDA, thư viện, 4 mô hình, đọc video, dashboard, Telegram."""
import json
import socket
import sys
from pathlib import Path

ok = True
print(f"Python      : {sys.version.split()[0]}")
try:
    import torch
    print(f"torch       : {torch.__version__} | CUDA: {torch.cuda.is_available()}"
          + (f" | GPU: {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else ""))
    if not torch.cuda.is_available():
        print("  [CANH BAO] Khong thay GPU -> se chay rat cham. Cap nhat driver NVIDIA (>= 570 cho CUDA 12.8).")
    import cv2
    import numpy as np
    import ultralytics
    from ultralytics import YOLO
    print(f"ultralytics : {ultralytics.__version__} | opencv: {cv2.__version__}")
except Exception as e:
    print("[LOI] Thieu thu vien:", e)
    sys.exit(1)

dummy = np.zeros((720, 1280, 3), np.uint8)
for name, w in [("Pose (nguoi nga)", "weights/yolo26n-pose.pt"),
                ("Danh nhau small", "third_party/Fight-Violence-detection-yolov8/yolo_small_weights.pt"),
                ("Danh nhau nano", "third_party/Fight-Violence-detection-yolov8/Yolo_nano_weights.pt"),
                ("Chay/khoi D-Fire", "weights/best_dfire.pt")]:
    try:
        YOLO(w).predict(dummy, imgsz=1280, verbose=False)
        print(f"  [OK] {name:18s} {w}")
    except Exception as e:
        ok = False
        print(f"  [LOI] {name:18s} {w}: {e}")

v = Path("videos/2.mp4")
if v.exists():
    cap = cv2.VideoCapture(str(v))
    good, _ = cap.read()
    print(f"  [{'OK' if good else 'LOI'}] Doc video H.265 ({v})")
    ok &= good

# ---- dashboard
try:
    import fastapi
    import uvicorn
    print(f"  [OK] Web dashboard      fastapi {fastapi.__version__}, uvicorn {uvicorn.__version__}")
except Exception as e:
    ok = False
    print("  [LOI] Web dashboard: thieu fastapi/uvicorn:", e)
s = socket.socket()
try:
    s.bind(("0.0.0.0", 8000))
    print("  [OK] Cong 8000 trong (dashboard)")
except OSError:
    print("  [CANH BAO] Cong 8000 dang bi chuong trinh khac dung -> dashboard da chay? hoac doi PORT trong start.bat")
finally:
    s.close()

# ---- telegram (khong bat buoc)
cfg = Path("config/telegram.json")
if not cfg.exists():
    print("  [--] Telegram: chua cau hinh (tuy chon) - xem HUONG_DAN.md muc Telegram")
else:
    try:
        c = json.loads(cfg.read_text(encoding="utf-8"))
        import requests
        r = requests.get(f"https://api.telegram.org/bot{c['token']}/getMe", timeout=10).json()
        if r.get("ok"):
            print(f"  [OK] Telegram bot @{r['result']['username']} | {len(c.get('chat_ids', []))} chat nhan canh bao")
        else:
            print(f"  [CANH BAO] Telegram: token sai ({r.get('description')})")
    except Exception as e:
        print(f"  [CANH BAO] Telegram: khong kiem tra duoc (mat mang?) - {e}")

sys.exit(0 if ok else 1)
