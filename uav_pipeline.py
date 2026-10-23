"""
================================================================================
UAV SCHOOL-SAFETY AI PIPELINE  (chạy trên MÁY CHỦ)
Nhận luồng video từ UAV -> phát hiện CHÁY/KHÓI + NGƯỜI NGÃ + ĐÁNH NHAU
-> xác nhận theo thời gian -> cảnh báo (lưu ảnh + ghi log + tùy chọn Telegram).
UAV chỉ bay và truyền hình ảnh về; toàn bộ AI chạy ở đây.

Mọi mô hình chạy trên CÙNG MỘT khung hình gốc (sạch); kết quả chỉ vẽ lên bản sao
để hiển thị/ghi video, nên mô hình này không "nhìn thấy" nét vẽ của mô hình khác.
--------------------------------------------------------------------------------
NGUỒN VIDEO (--source):
  * Aviateur (OpenIPC) chuyển tiếp RTP:   rtp://@:5600   (tự nhận H.264 / H.265)
  * Drone số / bộ phát streaming:   rtsp://192.168.1.10:554/live   |  udp://@:11111
  * FPV analog (VRX -> card thu USB EasyCap):   0   hoặc   1   (chỉ số thiết bị)
  * File video để thử nghiệm:  sample.mp4
  Nguồn trực tiếp được đọc ở luồng riêng, luôn lấy khung MỚI NHẤT (không bị trễ dồn).
--------------------------------------------------------------------------------
MÔ HÌNH:
  * Người ngã  : YOLO26n-pose (skeleton) + luật nằm (--pose-weights)
  * Đánh nhau  : Fight-YOLOv8 small (--fight-weights, "" = tắt)
                 + tuỳ chọn bản nano xác nhận chéo (--fight-confirm-weights)
  * Cháy/khói  : YOLO train trên D-Fire (--fire-weights, bỏ trống = tắt)
--------------------------------------------------------------------------------
VÍ DỤ:
  python uav_pipeline.py --source 2.mp4 --save-video out_2.mp4
  python uav_pipeline.py --source rtsp://192.168.1.10:554/live --view \\
         --fire-weights runs/dfire_full/weights/best.pt
================================================================================
"""
import argparse
import csv
import os
import signal
import socket
import sys
import tempfile
import threading
import time
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

import cv2
import numpy as np
from ultralytics import YOLO

# Chỉ số 17 điểm khớp COCO (cho mô hình pose)
L_SHO, R_SHO, L_HIP, R_HIP = 5, 6, 11, 12
KP_CONF_MIN = 0.4
# skeleton COCO: (khớp a, khớp b, nhóm) ; nhóm 0 = thân/đầu (vàng), 1 = trái (xanh), 2 = phải (đỏ)
SKELETON = [(0, 1, 0), (0, 2, 0), (1, 3, 0), (2, 4, 0), (5, 6, 0), (11, 12, 0), (5, 11, 1), (6, 12, 2),
            (5, 7, 1), (7, 9, 1), (6, 8, 2), (8, 10, 2), (11, 13, 1), (13, 15, 1), (12, 14, 2), (14, 16, 2)]
SKEL_COLOR = {0: (0, 220, 255), 1: (255, 170, 0), 2: (60, 60, 255)}   # BGR
LEFT_KP = {1, 3, 5, 7, 9, 11, 13, 15}
RIGHT_KP = {2, 4, 6, 8, 10, 12, 14, 16}


def draw_skeleton(canvas, kp, thick=2):
    """kp: mảng (17, 3) = x, y, độ tin cậy."""
    for a, b, g in SKELETON:
        if kp[a, 2] > KP_CONF_MIN and kp[b, 2] > KP_CONF_MIN:
            cv2.line(canvas, (int(kp[a, 0]), int(kp[a, 1])), (int(kp[b, 0]), int(kp[b, 1])),
                     SKEL_COLOR[g], thick, cv2.LINE_AA)
    for i, (x, y, c) in enumerate(kp):
        if c > KP_CONF_MIN:
            g = 1 if i in LEFT_KP else 2 if i in RIGHT_KP else 0
            cv2.circle(canvas, (int(x), int(y)), thick + 1, SKEL_COLOR[g], -1, cv2.LINE_AA)
FIGHT_DIR = "third_party/Fight-Violence-detection-yolov8"


def now_str():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class TimeWindow:
    """Cửa sổ trượt theo THỜI GIAN (giây), không theo số khung:
    vẫn đúng khi bị mất khung hoặc nguồn báo sai fps."""

    def __init__(self, seconds):
        self.seconds = seconds
        self.items = deque()          # (t, value)

    def add(self, t, value):
        self.items.append((t, value))
        while self.items and self.items[0][0] < t - self.seconds:
            self.items.popleft()

    def full(self, t):
        return bool(self.items) and self.items[0][0] <= t - 0.9 * self.seconds

    def ratio(self):
        return sum(bool(v) for _, v in self.items) / max(len(self.items), 1)

    def max(self):
        return max((v for _, v in self.items), default=0.0)


class AlertManager:
    """Gộp cảnh báo (chống spam), lưu ảnh minh chứng và ghi log CSV.
    `t` = thời điểm của khung hình (giây video khi chạy file, đồng hồ thật khi chạy trực tiếp),
    nên thời gian chờ giữa 2 cảnh báo giống nhau dù file được xử lý nhanh hơn thời gian thực."""

    def __init__(self, out_dir="outputs/alerts", cooldown_s=15.0, telegram=None):
        self.dir = Path(out_dir)
        (self.dir / "snapshots").mkdir(parents=True, exist_ok=True)
        self.cooldown = cooldown_s
        self.telegram = telegram
        self._last = defaultdict(lambda: -1e9)
        self.log_path = self.dir / "alert_log.csv"
        if not self.log_path.exists():
            with open(self.log_path, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(["thoi_gian", "giay_video", "loai", "chi_tiet", "anh"])

    def fire(self, frame, t, conf):
        self._raise("CHAY_KHOI", f"do_tin_cay={conf:.2f}", frame, t, key="fire")

    def fall(self, frame, t, seconds):
        self._raise("NGUOI_NGA", f"thoi_gian_nam={seconds:.1f}s", frame, t, key="fall")

    def violence(self, frame, t, conf):
        self._raise("DANH_NHAU", f"do_tin_cay={conf:.2f}", frame, t, key="violence")

    def _raise(self, kind, detail, frame, t, key):
        if t - self._last[key] < self.cooldown:
            return
        self._last[key] = t
        ts = now_str()
        fname = self.dir / "snapshots" / f"{kind}_{ts.replace(':', '-')}_{t:.1f}.jpg"
        cv2.imwrite(str(fname), frame)
        with open(self.log_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([ts, f"{t:.1f}", kind, detail, fname.name])
        print(f"[CANH BAO] {ts} | t={t:.1f}s | {kind} | {detail} | -> {fname.name}")
        if self.telegram:   # gui o luong rieng: mang cham khong lam dung xu ly video
            threading.Thread(target=self._send_telegram, args=(kind, detail, fname), daemon=True).start()

    def _send_telegram(self, kind, detail, fname):
        try:
            import requests
            url = f"https://api.telegram.org/bot{self.telegram['token']}/sendPhoto"
            with open(fname, "rb") as img:
                r = requests.post(url, data={"chat_id": self.telegram["chat_id"],
                                             "caption": f"{kind}: {detail}"},
                                  files={"photo": img}, timeout=10)
            if not r.ok:   # token / chat id sai -> Telegram tra loi HTTP, khong nem exception
                print(f"  (Telegram tu choi: HTTP {r.status_code} {r.text[:120]})")
        except Exception as e:
            print("  (Không gửi được Telegram:", e, ")")


class FireSmokeDetector:
    """Mô hình detection đã huấn luyện trên D-Fire (lớp fire/smoke)."""

    def __init__(self, weights, conf=0.45, confirm_frames=5):
        self.model = YOLO(weights)
        self.conf = conf
        self.confirm = confirm_frames
        self.streak = 0
        self.boxes = []

    def update(self, frame):
        """-> (có cảnh báo không, độ tin cậy cao nhất)."""
        res = self.model.predict(frame, conf=self.conf, verbose=False)[0]
        self.boxes = [(tuple(map(int, b.xyxy[0])), self.model.names[int(b.cls[0])], float(b.conf[0]))
                      for b in res.boxes]
        best = max((c for _, _, c in self.boxes), default=0.0)
        self.streak = self.streak + 1 if best >= self.conf else 0
        return self.streak >= self.confirm, best

    def draw(self, canvas):
        for (x1, y1, x2, y2), cls, c in self.boxes:
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(canvas, f"{cls} {c:.2f}", (x1, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)


class FallDetector:
    """YOLO-Pose + luật hình học + cửa sổ thời gian để phát hiện người ngã.
    Không cần định danh (ID) từng người: mỗi khung hình chỉ hỏi "có ai đang nằm không",
    rồi báo khi trong `lying_seconds` gần nhất có >= `lying_ratio` số khung có người nằm.
    Vì vậy không bị ảnh hưởng khi tracker đổi ID hay UAV di chuyển."""

    def __init__(self, weights="weights/yolo26n-pose.pt", lying_seconds=4.0, lying_ratio=0.5, imgsz=1280,
                 show_skeleton=True):
        self.model = YOLO(weights)
        self.imgsz = imgsz
        self.window = TimeWindow(lying_seconds)
        self.ratio = lying_ratio
        self.show_skeleton = show_skeleton
        # người trong khung hiện tại: (box xywh, có nằm không, keypoint (17, 3) = x, y, conf)
        # -> dùng lại được cho các mô hình phân tích hành động khác mà không phải chạy pose lần nữa
        self.people = []

    @staticmethod
    def _is_lying(box_xywh, kpts):
        x, y, w, h = box_xywh
        wide = w > 1.25 * h                      # khung người bè ngang
        torso_horizontal = False
        try:
            xy, cf = kpts.xy[0].cpu().numpy(), kpts.conf[0].cpu().numpy()
            if min(cf[[L_SHO, R_SHO, L_HIP, R_HIP]]) > KP_CONF_MIN:
                sho = (xy[L_SHO] + xy[R_SHO]) / 2
                hip = (xy[L_HIP] + xy[R_HIP]) / 2
                dx, dy = hip[0] - sho[0], hip[1] - sho[1]
                angle = np.degrees(np.arctan2(abs(dx), abs(dy) + 1e-6))
                torso_horizontal = angle > 55    # >55° so với phương dọc ~ nằm ngang
        except Exception:
            pass
        return wide or torso_horizontal

    def update(self, frame, t):
        """t = giây của khung hình -> (có cảnh báo không, số giây có người nằm trong cửa sổ)."""
        res = self.model.predict(frame, verbose=False, imgsz=self.imgsz)[0]
        self.people = []
        for i, xywh in enumerate(res.boxes.xywh.cpu().numpy()):
            kpts = res.keypoints[i] if res.keypoints is not None else None
            kp = (np.concatenate([kpts.xy[0].cpu().numpy(), kpts.conf[0].cpu().numpy()[:, None]], 1)
                  if kpts is not None and kpts.conf is not None else None)
            self.people.append((xywh, self._is_lying(xywh, kpts), kp))
        self.window.add(t, any(l for _, l, _ in self.people))
        r = self.window.ratio()
        return self.window.full(t) and r >= self.ratio, r * self.window.seconds

    def draw(self, canvas):
        for (x, y, w, h), lying, kp in self.people:
            color = (0, 165, 255) if lying else (0, 200, 0)
            x1, y1, x2, y2 = int(x - w / 2), int(y - h / 2), int(x + w / 2), int(y + h / 2)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2 if lying else 1)
            if self.show_skeleton and kp is not None:
                draw_skeleton(canvas, kp, thick=2 if h > 100 else 1)
            if lying:
                cv2.putText(canvas, "NAM", (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


class ViolenceDetector:
    """Fight-YOLOv8 (lớp 'violence') + bỏ phiếu theo thời gian.
    Báo khi trong `window_s` gần nhất có >= `ratio` số khung với conf >= `conf`.
    Tuỳ chọn: bản nano phải xác nhận (conf >= `confirm_conf` trong cửa sổ) để giảm báo nhầm."""

    def __init__(self, weights, imgsz=1280, conf=0.5, window_s=1.0, ratio=0.15,
                 confirm_weights=None, confirm_conf=0.7):
        self.model = YOLO(weights)
        self.vid = [k for k, v in self.model.names.items() if "no" not in v.lower()]
        self.confirm = YOLO(confirm_weights) if confirm_weights else None
        self.imgsz, self.conf, self.ratio, self.confirm_conf = imgsz, conf, ratio, confirm_conf
        self.hits, self.confs, self.cconf = TimeWindow(window_s), TimeWindow(window_s), TimeWindow(window_s)
        self.boxes = []

    def _violence(self, model, frame, conf):
        b = model.predict(frame, imgsz=self.imgsz, conf=conf, verbose=False)[0].boxes
        return [(tuple(map(int, b.xyxy[i])), float(b.conf[i])) for i in range(len(b)) if int(b.cls[i]) in self.vid]

    def update(self, frame, t):
        """t = giây của khung hình -> (có cảnh báo không, độ tin cậy cao nhất trong cửa sổ)."""
        self.boxes = self._violence(self.model, frame, 0.25)
        c = max((c for _, c in self.boxes), default=0.0)
        self.hits.add(t, c >= self.conf)
        self.confs.add(t, c)
        if self.confirm is not None:
            self.cconf.add(t, max((c for _, c in self._violence(self.confirm, frame, 0.25)), default=0.0))
        fired = self.hits.full(t) and self.hits.ratio() >= self.ratio
        if fired and self.confirm is not None:
            fired = self.cconf.max() >= self.confirm_conf
        return fired, self.confs.max()

    def draw(self, canvas):
        for (x1, y1, x2, y2), c in self.boxes:
            if c >= self.conf:
                cv2.rectangle(canvas, (x1, y1), (x2, y2), (255, 0, 255), 3)
                cv2.putText(canvas, f"danh nhau {c:.2f}", (x1, max(y1 - 6, 14)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)


class LatestFrameReader:
    """Đọc camera/luồng trực tiếp ở luồng riêng, chỉ giữ khung MỚI NHẤT.
    Nếu AI chậm hơn camera thì bỏ bớt khung cũ thay vì trễ dồn."""

    def __init__(self, src):
        self.src = src
        while True:   # nguon chua san sang (vd chua bat stream server trong fpv4win) -> cho
            try:
                self.cap = open_source(src)
                break
            except RuntimeError as e:
                print(f"{e} -> cho nguon video, thu lai sau 2s... (Ctrl+C de thoat)")
                time.sleep(2)
        self.frame, self.idx, self.ok = None, 0, True
        self.lock = threading.Lock()
        self.stop = False
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        while not self.stop:
            ok, f = self.cap.read()
            if not ok:
                print("Mất khung hình, thử kết nối lại sau 2s...")
                time.sleep(2)
                try:
                    self.cap.release()
                    self.cap = open_source(self.src)
                except RuntimeError as e:
                    print(" ", e)
                continue
            with self.lock:
                self.frame, self.idx = f, self.idx + 1

    def read(self, last_idx, timeout=None):
        """Chờ khung mới hơn `last_idx`; -> (idx, frame).
        Hết `timeout` giây mà chưa có khung mới (mất tín hiệu) -> (last_idx, None) nhưng `stop` vẫn False."""
        deadline = None if timeout is None else time.time() + timeout
        while not self.stop:
            with self.lock:
                if self.frame is not None and self.idx != last_idx:
                    return self.idx, self.frame
            if deadline is not None and time.time() >= deadline:
                break
            time.sleep(0.002)
        return last_idx, None

    def release(self, wait_s=2.0):
        """Dừng luồng đọc. -> True nếu đã dừng hẳn; False nếu vẫn kẹt trong cap.read()
        (mất tín hiệu) - khi đó không được giải phóng cap, gọi os._exit để thoát an toàn."""
        self.stop = True
        self.thread.join(wait_s)
        if self.thread.is_alive():
            return False
        self.cap.release()
        return True


def sane_fps(v, default):
    """Luong http/udp doi khi bao fps = 0 hoac rat lon -> dung gia tri mac dinh."""
    return v if v and 1 <= v <= 60 else default


RTP_CODECS = {96: "H264", 97: "H265"}   # payload type RTP theo quy uoc OpenIPC / Aviateur
# luong mang: mo toi da 10 s, doc ket qua sau 5 s khong co khung -> bao mat tin hieu va ket noi lai
# (mac dinh OpenCV doi ~30 s; vd UAV khoi dong lai -> phien RTP moi, FFmpeg bo qua goi cu)
NET_TIMEOUTS = [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 10000, cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000]


def add_capture_options(extra):
    """Them tuy chon FFmpeg cho OpenCV (OPENCV_FFMPEG_CAPTURE_OPTIONS).
    Tren Windows plugin FFmpeg cua OpenCV dung C runtime rieng (msvcrt) va chi chep bien moi truong
    luc khoi dong -> sua os.environ thoi la khong du, phai _putenv ca ban sao cua msvcrt."""
    cur = os.environ.get("OPENCV_FFMPEG_CAPTURE_OPTIONS", "")
    if extra in cur:
        return
    val = f"{cur}|{extra}" if cur else extra
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = val
    if sys.platform == "win32":
        import ctypes
        ctypes.cdll.msvcrt._putenv(f"OPENCV_FFMPEG_CAPTURE_OPTIONS={val}".encode())


def rtp_sdp(src, wait_s=3.0):
    """rtp://@:5600 -> file .sdp de FFmpeg doc luong RTP tho (Aviateur chuyen tiep video ra cong UDP).
    RTP tho khong tu mo ta codec -> nghe thu 1 goi de lay payload type (96 = H.264, 97 = H.265)."""
    port = urlparse(src).port or 5600
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.bind(("0.0.0.0", port))
        s.settimeout(wait_s)
        pkt = s.recv(65535)
    except socket.timeout:
        raise RuntimeError(f"Chua nhan duoc goi RTP nao tren cong UDP {port} (Aviateur da bat chuyen tiep chua?)")
    except OSError as e:
        raise RuntimeError(f"Khong nghe duoc cong UDP {port}: {e} (chuong trinh khac dang dung cong nay?)")
    finally:
        s.close()
    if len(pkt) < 12 or pkt[0] >> 6 != 2:
        raise RuntimeError(f"Du lieu tren cong UDP {port} khong phai RTP")
    pt = pkt[1] & 0x7F
    if pt not in RTP_CODECS:
        raise RuntimeError(f"RTP payload type {pt} khong ro codec (ho tro 96 = H.264, 97 = H.265)")
    sdp = Path(tempfile.gettempdir()) / f"uav_rtp_{port}.sdp"
    sdp.write_text("v=0\no=- 0 0 IN IP4 127.0.0.1\ns=UAV\nc=IN IP4 0.0.0.0\nt=0 0\n"
                   f"m=video {port} RTP/AVP {pt}\na=rtpmap:{pt} {RTP_CODECS[pt]}/90000\n", encoding="ascii")
    print(f"[INFO] RTP cong {port}: {RTP_CODECS[pt]} (payload type {pt})")
    return str(sdp)


def open_source(src):
    src = int(src) if str(src).isdigit() else src
    if str(src).startswith("rtp://"):
        add_capture_options("protocol_whitelist;file,udp,rtp")   # file .sdp duoc phep mo udp/rtp
        cap = cv2.VideoCapture(rtp_sdp(src), cv2.CAP_FFMPEG, NET_TIMEOUTS)
    elif "://" in str(src):
        # bat buoc FFmpeg cho luong mang: neu server chua bat, mac dinh OpenCV tu chuyen sang MSMF (Windows)
        # -> MSMF dem ~3s (khung den tung dot) va treo khi mat ket noi, khong bao gio ket noi lai
        cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG, NET_TIMEOUTS)
    else:
        cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"Không mở được nguồn video: {src}")
    return cap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--fire-weights", default=None,
                    help="best.pt của mô hình cháy-khói (bỏ trống = tắt phát hiện cháy)")
    ap.add_argument("--pose-weights", default="weights/yolo26n-pose.pt")
    ap.add_argument("--fight-weights", default=f"{FIGHT_DIR}/yolo_small_weights.pt",
                    help='mô hình đánh nhau ("" = tắt)')
    ap.add_argument("--fight-confirm-weights", default=None,
                    help=f"tuỳ chọn: bản nano xác nhận chéo, vd {FIGHT_DIR}/Yolo_nano_weights.pt")
    ap.add_argument("--fps", type=int, default=25, help="fps dùng khi nguồn không báo fps")
    ap.add_argument("--imgsz", type=int, default=1280,
                    help="kich thuoc anh dua vao mo hinh pose va danh nhau (1280 = giu do phan giai 720p; "
                         "640 lam nguoi nho lai, bo sot nguoi nam - xem report.md)")
    ap.add_argument("--every", type=int, default=1, help="chạy AI mỗi N khung hình (chỉ với file video)")
    ap.add_argument("--lying-seconds", type=float, default=4.0,
                    help="do dai cua so thoi gian xet nguoi nam (giay)")
    ap.add_argument("--lying-ratio", type=float, default=0.5,
                    help="bao nga khi ti le khung co nguoi nam trong cua so >= gia tri nay")
    ap.add_argument("--cooldown", type=float, default=15.0, help="giây tối thiểu giữa 2 cảnh báo cùng loại")
    ap.add_argument("--view", action="store_true")
    ap.add_argument("--no-skeleton", action="store_true", help="khong ve skeleton (van chay pose de phat hien nga)")
    ap.add_argument("--save-video", default=None,
                    help="ghi video da ve khung/canh bao, vd out_pipeline.mp4 "
                         "(ten khong co thu muc se luu vao outputs/videos/)")
    ap.add_argument("--telegram-token", default=None)
    ap.add_argument("--telegram-chat", default=None)
    args = ap.parse_args()

    # ten file tran -> dat vao outputs/videos/
    if args.save_video and not Path(args.save_video).parent.name:
        Path("outputs/videos").mkdir(parents=True, exist_ok=True)
        args.save_video = str(Path("outputs/videos") / args.save_video)

    # Nguon la FILE video thi doc tuan tu (khong bo khung) va het file thi DUNG;
    # nguon live (rtsp/udp/webcam) doc o luong rieng, lay khung moi nhat, tu ket noi lai.
    is_file = Path(str(args.source)).is_file()
    if is_file:
        cap = open_source(args.source)
        src_fps = sane_fps(cap.get(cv2.CAP_PROP_FPS), args.fps)
        every = max(args.every, 1)
    else:
        try:
            reader = LatestFrameReader(args.source)
        except KeyboardInterrupt:
            print("Dung theo yeu cau.")
            return
        src_fps = sane_fps(reader.cap.get(cv2.CAP_PROP_FPS), args.fps)
        every = 1

    tele = ({"token": args.telegram_token, "chat_id": args.telegram_chat}
            if args.telegram_token and args.telegram_chat else None)
    alerts = AlertManager(cooldown_s=args.cooldown, telegram=tele)
    fire = FireSmokeDetector(args.fire_weights) if args.fire_weights else None
    fall = FallDetector(args.pose_weights, lying_seconds=args.lying_seconds,
                        lying_ratio=args.lying_ratio, imgsz=args.imgsz, show_skeleton=not args.no_skeleton)
    fight = (ViolenceDetector(args.fight_weights, imgsz=args.imgsz,
                              confirm_weights=args.fight_confirm_weights)
             if args.fight_weights else None)
    print(f"[INFO] Mo hinh: nga=bat, danh nhau={'bat' if fight else 'tat'}, "
          f"chay/khoi={'bat' if fire else 'tat'} | nguon {src_fps:.1f} fps")

    # dung bang kill/SIGTERM (Linux) hoac dong cua so console/Ctrl+Break (Windows: SIGBREAK)
    # cung dong file video dung cach nhu Ctrl+C
    def _stop(*_):
        raise KeyboardInterrupt
    for sig in ("SIGTERM", "SIGBREAK"):
        if hasattr(signal, sig):
            signal.signal(getattr(signal, sig), _stop)
    writer = None   # tao o khung hinh dau tien (can biet kich thuoc that)
    print(f"[{now_str()}] Bắt đầu giám sát nguồn: {args.source}"
          f" ({'file video' if is_file else 'luong truc tiep'})")
    n, t0, proc_fps = 0, time.time(), 0.0
    proc_n, proc_s = 0, 0.0          # tong so khung + tong thoi gian AI -> fps trung binh dung
    canvas, t_frame = None, t0   # khung da ve gan nhat + luc nhan khung (de hien "mat tin hieu")
    try:
        while True:
            if is_file:
                ok, frame = cap.read()
                if not ok:
                    print("Het file video.")
                    break
                n += 1
                if n % every != 0:
                    continue
                t = n / src_fps                    # giay trong video
            else:
                n, frame = reader.read(n, timeout=0.1)
                if frame is None:
                    if reader.stop:
                        break
                    # mat tin hieu: van cap nhat cua so (khong bi "Not Responding", van nhan q)
                    if args.view:
                        lost = time.time() - t_frame
                        if canvas is not None and lost >= 1.0:
                            view = canvas.copy()
                            cv2.putText(view, f"MAT TIN HIEU {lost:.0f}s - dang cho ket noi lai...", (10, 60),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                            cv2.imshow("UAV School-Safety AI", view)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            print("Nhan q - dung.")
                            break
                    continue
                t_frame = time.time()
                t = t_frame - t0                   # giay ke tu luc bat dau giam sat

            tick = time.time()
            # --- cac mo hinh deu chay tren khung GOC `frame` ---
            fire_hit = fire.update(frame) if fire else (False, 0.0)
            fall_hit = fall.update(frame, t)
            fight_hit = fight.update(frame, t) if fight else (False, 0.0)
            dt = time.time() - tick
            proc_n, proc_s = proc_n + 1, proc_s + dt
            proc_fps = 0.9 * proc_fps + 0.1 / max(dt, 1e-3) if proc_fps else 1 / max(dt, 1e-3)   # hien thi (muot)

            # --- ve ket qua len BAN SAO ---
            canvas = frame.copy()
            for det in (fire, fall, fight):
                if det is not None:
                    det.draw(canvas)
            cv2.putText(canvas, f"AI {proc_fps:.0f} fps", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            if fire_hit[0]:
                alerts.fire(canvas, t, fire_hit[1])
            if fall_hit[0]:
                alerts.fall(canvas, t, fall_hit[1])
            if fight_hit[0]:
                alerts.violence(canvas, t, fight_hit[1])

            if args.save_video:
                if writer is None:
                    h, w = canvas.shape[:2]
                    out_fps = src_fps / every
                    writer = cv2.VideoWriter(args.save_video, cv2.VideoWriter_fourcc(*"mp4v"), out_fps, (w, h))
                    print(f"[INFO] Ghi video ket qua -> {args.save_video} ({w}x{h} @ {out_fps:.1f} fps)")
                writer.write(canvas)
            if args.view:
                cv2.imshow("UAV School-Safety AI", canvas)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("Nhan q - dung.")
                    break
    except KeyboardInterrupt:
        print("Dung theo yeu cau.")
    finally:
        # dong file video TRUOC: luong doc camera co the dang bi chan cho du lieu
        if writer is not None:
            writer.release()
            print(f"[OK] Video ket qua: {args.save_video}")
        cv2.destroyAllWindows()
        cap.release() if is_file else reader.release()
        print(f"[{now_str()}] Kết thúc. Toc do AI trung binh ~{proc_n / max(proc_s, 1e-6):.0f} fps "
              f"({time.time() - t0:.0f}s)", flush=True)
        if not is_file:
            # luong doc (native FFmpeg) co the con chay: thoat thang, tranh crash khi Python don dep
            sys.stdout.flush()
            os._exit(0)


if __name__ == "__main__":
    main()
