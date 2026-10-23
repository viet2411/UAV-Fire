"""
================================================================================
UAV SCHOOL-SAFETY AI — WEB DASHBOARD (màn hình giám sát)
Cùng bộ AI với uav_pipeline.py (cháy/khói + người ngã + đánh nhau), thêm giao diện web:
  * video trực tiếp có khung + skeleton, trạng thái kết nối UAV, tốc độ AI, độ trễ
  * 3 thẻ module (bật/tắt, đổi màu khi cảnh báo), âm thanh cảnh báo
  * nhật ký cảnh báo có ảnh + clip 5 s trước / 5 s sau, nút "Đúng / Báo nhầm"
    (ghi outputs/alerts/feedback.csv -> đo precision ngoài thực địa, gom dữ liệu có nhãn)
  * ghi hình, chụp ảnh, bật/tắt khung - skeleton, chỉnh ngưỡng khi đang chạy
  * tự ghi VIDEO HÀNH TRÌNH ngay khi có hình (--auto-record ai|goc|ca_hai), chia file 10 phút
  * đếm số người + theo dõi TỤ TẬP ĐÔNG NGƯỜI: là TRẠNG THÁI (banner, đếm thời gian, dải trên dòng thời gian),
    không phải sự cố -> không ghi nhật ký / ảnh / Telegram
Mở trình duyệt:  http://localhost:8000   (máy khác trong LAN: http://<IP-máy-chủ>:8000)
--------------------------------------------------------------------------------
VÍ DỤ:
  python uav_dashboard.py --source rtp://@:5600 --fire-weights weights/best_dfire.pt      (Aviateur)
  python uav_dashboard.py --source 2.mp4 --loop                                            (demo bằng file)
================================================================================
"""
import argparse
import csv
import os
import shutil
import signal
import sys
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from notify_telegram import TelegramNotifier, load_config
from uav_pipeline import (FIGHT_DIR, AlertManager, FallDetector, FireSmokeDetector, LatestFrameReader,
                          TimeWindow, ViolenceDetector, draw_skeleton, now_str, open_source, sane_fps)

WEB_DIR = Path(__file__).parent / "web"
OUT = Path("outputs")
KIND_KEY = {"CHAY_KHOI": "fire", "NGUOI_NGA": "fall", "DANH_NHAU": "fight"}
INCIDENTS = ("fire", "fall", "fight")          # sự cố: ghi nhật ký, ảnh, clip, Telegram
MODULES = INCIDENTS + ("crowd",)               # crowd: chỉ là trạng thái theo dõi
CLIP_PRE_S, CLIP_POST_S, CLIP_FPS, CLIP_W = 5.0, 5.0, 10, 640
WINDOW = "UAV School-Safety AI"


# ----------------------------------------------------------------------------- crowd
class CrowdDetector:
    """Đếm người + phát hiện tụ tập, dùng lại kết quả pose của FallDetector (không chạy thêm mô hình).
    Hai người cùng nhóm khi khoảng cách 2 bàn chân < `link` x chiều cao người (chuẩn hoá theo chiều cao
    nên đúng cho cả người gần/xa UAV); ghép bắc cầu (A gần B, B gần C -> 1 nhóm).
    TRẠNG THÁI "đang tụ tập" (không phải sự cố): bật khi nhóm lớn nhất >= `min_group` người trong >= `on_ratio`
    thời gian của `window_s` giây gần nhất; chỉ tắt khi tỉ lệ đó < `off_ratio` (trễ - tránh nhấp nháy).
    Mức "rat_dong" khi nhóm >= 2 x min_group."""

    def __init__(self, min_group=5, link=1.5, window_s=5.0, on_ratio=0.6, off_ratio=0.3):
        self.min_group, self.link, self.on_ratio, self.off_ratio = min_group, link, on_ratio, off_ratio
        self.window = TimeWindow(window_s)
        self.groups, self.total, self.largest = [], 0, 0
        self.active, self.since, self.peak = False, None, 0

    def level(self):
        if not self.active:
            return "binh_thuong"
        return "rat_dong" if self.peak >= 2 * self.min_group else "dong"

    def update(self, people, t):
        """people = FallDetector.people -> (có cảnh báo không, số người nhóm lớn nhất)."""
        boxes = [p[0] for p in people]
        n = len(boxes)
        parent = list(range(n))

        def root(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i
        for i in range(n):
            for j in range(i + 1, n):
                (xi, yi, wi, hi), (xj, yj, wj, hj) = boxes[i], boxes[j]
                size = (max(wi, hi) + max(wj, hj)) / 2           # người nằm: dùng cạnh dài
                if np.hypot(xi - xj, (yi + hi / 2) - (yj + hj / 2)) < self.link * size:
                    parent[root(i)] = root(j)
        members = {}
        for i in range(n):
            members.setdefault(root(i), []).append(boxes[i])
        self.groups = sorted(members.values(), key=len, reverse=True)
        self.total, self.largest = n, len(self.groups[0]) if self.groups else 0
        self.window.add(t, self.largest >= self.min_group)
        r = self.window.ratio()
        if not self.active and self.window.full(t) and r >= self.on_ratio:
            self.active, self.since, self.peak = True, t, self.largest
        elif self.active and r < self.off_ratio:
            self.active, self.since, self.peak = False, None, 0
        if self.active:
            self.peak = max(self.peak, self.largest)
        return self.active, self.largest

    def draw(self, canvas, alarm):
        for g in self.groups:
            if len(g) < 2:
                continue
            x1 = int(min(x - w / 2 for x, y, w, h in g)) - 8
            y1 = int(min(y - h / 2 for x, y, w, h in g)) - 8
            x2 = int(max(x + w / 2 for x, y, w, h in g)) + 8
            y2 = int(max(y + h / 2 for x, y, w, h in g)) + 8
            big = len(g) >= self.min_group
            color = (0, 170, 255) if big and alarm else (255, 200, 0) if big else (200, 160, 80)   # cam khi đang tụ tập
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 3 if big else 1)
            label = f"{len(g)} nguoi"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(canvas, (x1, y2), (x1 + tw + 10, y2 + th + 10), color, -1)
            cv2.putText(canvas, label, (x1 + 5, y2 + th + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)


# ----------------------------------------------------------------------------- clips
class ClipRecorder:
    """Giữ ~5 s hình gần nhất (thu nhỏ, 10 fps); khi có cảnh báo ghi thêm 5 s rồi xuất clip
    WebM (VP8, trình duyệt phát được) ở luồng riêng. Không ghi được WebM -> .avi (MJPG)."""

    def __init__(self, out_dir, on_done=None):
        self.on_done = on_done        # gọi khi clip ghi xong: on_done(alert, path)
        self.dir = Path(out_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.buf = deque(maxlen=int(CLIP_PRE_S * CLIP_FPS))
        self.jobs = []                # [alert, frames, t_end]
        self.last_t = -1e9

    def add(self, t, canvas):
        if t - self.last_t < 1.0 / CLIP_FPS:
            return
        self.last_t = t
        h, w = canvas.shape[:2]
        small = cv2.resize(canvas, (CLIP_W, int(h * CLIP_W / w)))
        self.buf.append(small)
        for job in self.jobs:
            job[1].append(small)
        done = [j for j in self.jobs if t >= j[2]]
        self.jobs = [j for j in self.jobs if t < j[2]]
        for alert, frames, _ in done:
            threading.Thread(target=self._write, args=(alert, frames), daemon=True).start()

    def start(self, alert, t):
        self.jobs.append([alert, list(self.buf), t + CLIP_POST_S])

    def flush(self):
        for alert, frames, _ in self.jobs:
            self._write(alert, frames)
        self.jobs = []

    def _write(self, alert, frames):
        if not frames:
            return
        h, w = frames[0].shape[:2]
        for fourcc, ext in (("VP80", "webm"), ("MJPG", "avi")):
            path = self.dir / f"{alert['id']:04d}_{alert['kind']}.{ext}"
            wr = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*fourcc), CLIP_FPS, (w, h))
            if wr.isOpened():
                for f in frames:
                    wr.write(f)
                wr.release()
                alert["clip"] = f"/clips/{path.name}"
                if self.on_done:
                    self.on_done(alert, path)
                return


# ----------------------------------------------------------------------------- journey video
class JourneyRecorder:
    """Tự ghi video hành trình bay.
    * mode: "ai" (có khung AI), "goc" (hình gốc - dùng làm dữ liệu train lại), "ca_hai".
    * Ghi theo ĐỒNG HỒ THẬT: AI chậm hơn nguồn thì lặp khung, nhanh hơn thì bỏ bớt -> thời lượng video = thời gian bay.
    * Chia file mỗi `segment_min` phút (máy treo / mất điện chỉ mất đoạn cuối).
    * Còn < `min_free_gb` GB trống thì dừng ghi."""

    def __init__(self, out_dir, mode="ai", fps=25.0, segment_min=10.0, min_free_gb=2.0):
        self.dir = Path(out_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.mode, self.fps = mode, min(max(fps, 1.0), 30.0)
        self.segment_s, self.min_free = segment_min * 60, min_free_gb * 1e9
        self.enabled = True
        self.session = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.writers, self.part, self.files = {}, 0, []
        self.seg_start = self.next_t = None
        self.total_s, self.stopped_reason = 0.0, None
        self.last_disk_check = 0.0

    def _open(self, size):
        self.close()
        self.part += 1
        kinds = {"ai": ["ai"], "goc": ["goc"], "ca_hai": ["ai", "goc"]}[self.mode]
        for k in kinds:
            name = f"hanh_trinh_{self.session}_{self.part:02d}" + ("_goc" if k == "goc" else "") + ".mp4"
            w = cv2.VideoWriter(str(self.dir / name), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, size)
            if w.isOpened():
                self.writers[k] = w
                self.files.append(name)
        self.seg_start = time.time()
        print(f"[HANH TRINH] Ghi -> {', '.join(self.files[-len(kinds):])}", flush=True)

    def write(self, frame, canvas):
        if not self.enabled:
            return
        now = time.time()
        if now - self.last_disk_check > 30:
            self.last_disk_check = now
            if shutil.disk_usage(self.dir).free < self.min_free:
                self.stopped_reason = "o dia sap day"
                self.enabled = False
                self.close()
                print("[HANH TRINH] DUNG GHI: o dia con < 2 GB trong", flush=True)
                return
        if not self.writers or now - self.seg_start >= self.segment_s:
            h, w = frame.shape[:2]
            self._open((w, h))
            self.next_t = now
        n = 0                                   # số khung cần ghi để bám theo đồng hồ thật
        while self.next_t <= now and n < 10:
            n += 1
            self.next_t += 1.0 / self.fps
        if self.next_t < now:                   # tụt quá xa (vd mất tín hiệu) -> bắt kịp, không lặp vô hạn
            self.next_t = now
        for _ in range(n):
            if "ai" in self.writers:
                self.writers["ai"].write(canvas)
            if "goc" in self.writers:
                self.writers["goc"].write(frame)
        self.total_s += n / self.fps

    def pause(self):
        """Mất tín hiệu: không ghi; khi có hình lại thì đi tiếp (không lặp khung cũ)."""
        if self.next_t is not None:
            self.next_t = time.time()

    def set_enabled(self, on):
        self.enabled = bool(on)
        if not on:
            self.close()
        else:
            self.stopped_reason = None

    def close(self):
        for w in self.writers.values():
            w.release()
        if self.writers:
            print(f"[HANH TRINH] Da luu phan {self.part:02d}", flush=True)
        self.writers = {}

    def status(self):
        return {"on": self.enabled and bool(self.writers), "enabled": self.enabled, "mode": self.mode,
                "seconds": round(self.total_s), "part": self.part, "files": self.files[-4:],
                "stopped_reason": self.stopped_reason}


# ----------------------------------------------------------------------------- shared state
class Dashboard:
    """Trạng thái dùng chung giữa vòng xử lý AI (luồng chính) và web server (luồng phụ)."""

    def __init__(self, args, modules_available):
        self.lock = threading.Lock()
        self.args = args
        self.cfg = {
            "modules": {k: v for k, v in modules_available.items()},
            "overlays": {"boxes": True, "skeleton": not args.no_skeleton},
            "thresholds": {"fight_conf": 0.5, "lying_ratio": args.lying_ratio, "fire_conf": 0.45,
                           "crowd_min": args.crowd_min},
            "cooldown": args.cooldown,
        }
        self.available = dict(modules_available)
        self.status = {"source": str(args.source), "mode": "file" if Path(str(args.source)).is_file() else "live",
                       "connected": False, "lost_s": 0.0, "ai_fps": 0.0, "latency_ms": 0.0, "src_fps": 0.0,
                       "frames": 0, "started": now_str(), "uptime_s": 0, "recording": False, "gpu": gpu_name(),
                       "ended": False, "people": 0, "largest_group": 0,
                       "crowd": {"active": False, "level": "binh_thuong", "largest": 0, "peak": 0,
                                 "threshold": args.crowd_min, "since_s": 0}}
        self.modules = {k: {"state": "ok", "value": 0.0, "last_alert": None} for k in MODULES}
        self.people_hist = deque(maxlen=3600)   # (giây, số người) mỗi ~1 s, cho đồ thị
        self.alerts = []
        self.canvas, self.seq = None, 0
        self._jpeg, self._jpeg_seq = None, -1
        self.record_req = None        # True / False khi web bấm nút ghi hình
        self.journey_req = None       # True / False khi web bật/tắt ghi hành trình
        self.snapshot_req = []        # danh sách Event chờ ảnh chụp
        self.feedback_path = OUT / "alerts" / "feedback.csv"
        self.notifier = None          # TelegramNotifier (nếu có config/telegram.json)

    def jpeg(self):
        with self.lock:
            if self.canvas is None:
                return None, -1
            if self._jpeg_seq != self.seq:
                ok, buf = cv2.imencode(".jpg", self.canvas, [cv2.IMWRITE_JPEG_QUALITY, 80])
                self._jpeg, self._jpeg_seq = buf.tobytes(), self.seq
            return self._jpeg, self._jpeg_seq

    def snapshot(self):
        with self.lock:
            return {"status": dict(self.status), "cfg": self.cfg, "available": self.available,
                    "modules": self.modules, "alerts": self.alerts[-100:], "stats": self.stats(),
                    "people_hist": list(self.people_hist)}

    def stats(self):
        s = {}
        for k in INCIDENTS:
            a = [x for x in self.alerts if x["key"] == k]
            ok = sum(x["verdict"] == "dung" for x in a)
            bad = sum(x["verdict"] == "nham" for x in a)
            s[k] = {"total": len(a), "dung": ok, "nham": bad,
                    "precision": round(ok / (ok + bad), 2) if ok + bad else None}
        return s

    def set_verdict(self, alert_id, verdict):
        with self.lock:
            a = next((x for x in self.alerts if x["id"] == alert_id), None)
            if a is None:
                return None
            a["verdict"] = verdict
        new = not self.feedback_path.exists()
        with open(self.feedback_path, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["thoi_gian_xac_nhan", "id", "loai", "thoi_gian_canh_bao", "giay", "chi_tiet",
                            "danh_gia", "anh"])
            w.writerow([now_str(), a["id"], a["kind"], a["time"], a["t"], a["detail"], verdict or "bo_chon",
                        a["snapshot"]])
        if self.notifier:
            self.notifier.update_verdict(a)
        return a


def gpu_name():
    try:
        import torch
        return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    except Exception:
        return "?"


class WebAlertManager(AlertManager):
    """AlertManager của uav_pipeline (ảnh + log CSV + Telegram) + báo sang dashboard."""

    def __init__(self, dash, clips, **kw):
        super().__init__(**kw)
        self.dash, self.clips = dash, clips


    def _raise(self, kind, detail, frame, t, key):
        self.cooldown = self.dash.cfg["cooldown"]
        before = self._last[key]
        super()._raise(kind, detail, frame, t, key)
        if self._last[key] == before:            # bị chặn bởi thời gian chờ
            return
        snap = sorted((self.dir / "snapshots").glob(f"{kind}_*_{t:.1f}.jpg"))
        with self.dash.lock:
            a = {"id": len(self.dash.alerts) + 1, "kind": kind, "key": KIND_KEY[kind], "time": now_str(),
                 "t": round(t, 1), "detail": detail, "snapshot": f"/snapshots/{snap[-1].name}" if snap else None,
                 "clip": None, "verdict": None}
            self.dash.alerts.append(a)
            self.dash.modules[KIND_KEY[kind]]["last_alert"] = a["time"]
        self.clips.start(a, t)
        if self.dash.notifier and snap:
            self.dash.notifier.send_alert(a, snap[-1])


# ----------------------------------------------------------------------------- web server
def build_app(dash):
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import FileResponse, StreamingResponse
    from fastapi.staticfiles import StaticFiles

    app = FastAPI(title="UAV School-Safety AI")
    for d in ("alerts/snapshots", "alerts/clips", "snapshots", "videos"):
        (OUT / d).mkdir(parents=True, exist_ok=True)
    app.mount("/snapshots", StaticFiles(directory=OUT / "alerts" / "snapshots"), name="snapshots")
    app.mount("/clips", StaticFiles(directory=OUT / "alerts" / "clips"), name="clips")
    app.mount("/manual", StaticFiles(directory=OUT / "snapshots"), name="manual")
    app.mount("/videos", StaticFiles(directory=OUT / "videos"), name="videos")

    @app.get("/")
    def index():
        return FileResponse(WEB_DIR / "dashboard.html")

    @app.get("/api/state")
    def state():
        return dash.snapshot()

    @app.get("/video.mjpg")
    def video():
        def gen():
            last = -1
            while True:
                jpg, seq = dash.jpeg()
                if jpg is None or seq == last:
                    time.sleep(0.01)
                    continue
                last = seq
                yield b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(jpg)).encode() + \
                      b"\r\n\r\n" + jpg + b"\r\n"
        return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")

    @app.post("/api/config")
    def config(body: dict):
        with dash.lock:
            for grp in ("modules", "overlays", "thresholds"):
                for k, v in (body.get(grp) or {}).items():
                    if k in dash.cfg[grp]:
                        if grp == "modules" and not dash.available.get(k):
                            continue
                        dash.cfg[grp][k] = type(dash.cfg[grp][k])(v)
            if "cooldown" in body:
                dash.cfg["cooldown"] = float(body["cooldown"])
        return dash.cfg

    @app.post("/api/alerts/{alert_id}/verdict")
    def verdict(alert_id: int, body: dict):
        v = body.get("verdict")
        if v not in ("dung", "nham", None):
            raise HTTPException(400, "verdict = dung | nham | null")
        a = dash.set_verdict(alert_id, v)
        if a is None:
            raise HTTPException(404, "khong co canh bao nay")
        return a

    @app.post("/api/record")
    def record(body: dict):
        dash.record_req = bool(body.get("on"))
        return {"ok": True}

    @app.post("/api/journey")
    def journey(body: dict):
        dash.journey_req = bool(body.get("on"))
        return {"ok": True}

    @app.post("/api/snapshot")
    def snapshot():
        ev = threading.Event()
        holder = {}
        dash.snapshot_req.append((ev, holder))
        ev.wait(3)
        return holder or {"error": "chua co hinh"}

    return app


def start_web(dash, host, port):
    import uvicorn
    server = uvicorn.Server(uvicorn.Config(build_app(dash), host=host, port=port, log_level="warning"))
    threading.Thread(target=server.run, daemon=True).start()
    return server


def lan_ips():
    import socket
    ips = set()
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ips.add(s.getsockname()[0])
        s.close()
    except OSError:
        pass
    return sorted(ips)


# ----------------------------------------------------------------------------- drawing
def draw(canvas, cfg, fire, fall, fight, crowd, crowd_alarm):
    ov = cfg["overlays"]
    if cfg["modules"]["crowd"] and (ov["boxes"] or crowd_alarm):
        crowd.draw(canvas, crowd_alarm)
    if fall is not None and (cfg["modules"]["fall"] or cfg["modules"]["crowd"]):
        for (x, y, w, h), lying, kp in fall.people:
            if ov["boxes"] or lying:
                color = (0, 165, 255) if lying else (0, 200, 0)
                x1, y1, x2, y2 = int(x - w / 2), int(y - h / 2), int(x + w / 2), int(y + h / 2)
                cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 3 if lying else 1)
                if lying:
                    cv2.putText(canvas, "NAM", (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            if ov["skeleton"] and kp is not None:
                draw_skeleton(canvas, kp, thick=2 if h > 100 else 1)
    if ov["boxes"]:
        if fire is not None and cfg["modules"]["fire"]:
            fire.draw(canvas)
        if fight is not None and cfg["modules"]["fight"]:
            fight.draw(canvas)


# ----------------------------------------------------------------------------- main loop
def main():
    ap = argparse.ArgumentParser(description="UAV School-Safety AI - web dashboard")
    ap.add_argument("--source", required=True)
    ap.add_argument("--fire-weights", default=None)
    ap.add_argument("--pose-weights", default="weights/yolo26n-pose.pt")
    ap.add_argument("--fight-weights", default=f"{FIGHT_DIR}/yolo_small_weights.pt")
    ap.add_argument("--fight-confirm-weights", default=None)
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--lying-seconds", type=float, default=4.0)
    ap.add_argument("--lying-ratio", type=float, default=0.5)
    ap.add_argument("--cooldown", type=float, default=15.0)
    ap.add_argument("--crowd-min", type=int, default=5, help="trang thai tu tap khi nhom >= N nguoi")
    ap.add_argument("--no-skeleton", action="store_true")
    ap.add_argument("--view", action="store_true", help="mo them cua so video tren may chu (nhu run_live.bat); q de dung")
    ap.add_argument("--auto-record", choices=["ai", "goc", "ca_hai"], default=None,
                    help="tu ghi video hanh trinh tu luc co hinh: ai = co khung AI, goc = hinh goc, ca_hai")
    ap.add_argument("--segment-min", type=float, default=10.0, help="do dai moi file hanh trinh (phut)")
    ap.add_argument("--loop", action="store_true", help="file video: phat lai lien tuc (demo)")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--telegram-config", default="config/telegram.json",
                    help="file cau hinh Telegram (token, chat_ids, notify_crowd)")
    ap.add_argument("--telegram-token", default=None, help="(cu) dung --telegram-config thay the")
    ap.add_argument("--telegram-chat", default=None)
    args = ap.parse_args()

    is_file = Path(str(args.source)).is_file()
    fire = FireSmokeDetector(args.fire_weights) if args.fire_weights else None
    fall = FallDetector(args.pose_weights, lying_seconds=args.lying_seconds, lying_ratio=args.lying_ratio,
                        imgsz=args.imgsz)
    fight = (ViolenceDetector(args.fight_weights, imgsz=args.imgsz, confirm_weights=args.fight_confirm_weights)
             if args.fight_weights else None)
    crowd = CrowdDetector(min_group=args.crowd_min)
    dash = Dashboard(args, {"fire": fire is not None, "fall": True, "fight": fight is not None, "crowd": True})
    tg_cfg = load_config(args.telegram_config)
    if tg_cfg:
        src = str(args.source)
        label = ("RTP · Aviateur" if src.startswith("rtp://") else "HTTP" if src.startswith("http") else
                 f"File {Path(src).name}" if is_file else src)
        dash.notifier = TelegramNotifier(tg_cfg, dash, source_label=label)
        print(f"[TELEGRAM] Bat - gui toi {len(dash.notifier.chats)} chat", flush=True)
    clips = ClipRecorder(OUT / "alerts" / "clips",
                         on_done=(lambda a, p: dash.notifier.send_clip(a, p)) if dash.notifier else None)
    tele = ({"token": args.telegram_token, "chat_id": args.telegram_chat}
            if args.telegram_token and args.telegram_chat and not tg_cfg else None)
    alerts = WebAlertManager(dash, clips, cooldown_s=args.cooldown, telegram=tele)
    start_web(dash, args.host, args.port)
    print(f"[WEB] Mo trinh duyet: http://localhost:{args.port}" +
          "".join(f"   |   may khac trong LAN: http://{ip}:{args.port}" for ip in lan_ips()), flush=True)

    def _stop(*_):
        raise KeyboardInterrupt
    for sig in ("SIGTERM", "SIGBREAK"):
        if hasattr(signal, sig):
            signal.signal(getattr(signal, sig), _stop)

    if is_file:
        cap = open_source(args.source)
        src_fps = sane_fps(cap.get(cv2.CAP_PROP_FPS), args.fps)
    else:
        try:
            reader = LatestFrameReader(args.source)
        except KeyboardInterrupt:
            os._exit(0)
        src_fps = sane_fps(reader.cap.get(cv2.CAP_PROP_FPS), args.fps)
    dash.status["src_fps"] = round(src_fps, 1)
    journey = (JourneyRecorder(OUT / "videos", args.auto_record, fps=src_fps, segment_min=args.segment_min)
               if args.auto_record else None)
    if journey:
        print(f"[HANH TRINH] Tu ghi video hanh trinh ({args.auto_record}), moi file {args.segment_min:g} phut "
              f"-> {OUT / 'videos'}", flush=True)
    print(f"[{now_str()}] Bat dau giam sat: {args.source} ({'file' if is_file else 'truc tiep'}, {src_fps:.1f} fps)",
          flush=True)

    t0 = time.time()
    n, t_frame, t_file0 = 0, t0, t0
    proc = deque(maxlen=60)          # thời gian xử lý các khung gần nhất (đo fps AI thật)
    writer, rec_path = None, None
    try:
        while True:
            # ---------------- lấy khung
            if is_file:
                ok, frame = cap.read()
                if not ok:
                    if args.loop:
                        cap.release()
                        cap = open_source(args.source)
                        t_file0 = time.time()
                        continue
                    print("Het file video - dashboard van mo de xem lai. Ctrl+C de thoat.", flush=True)
                    dash.status.update(ended=True, connected=False)
                    clips.flush()
                    while True:
                        time.sleep(1)
                n += 1
                # chạy theo thời gian thực như luồng thật (file đọc nhanh hơn thì chờ)
                lag = (n / src_fps) - (time.time() - t0)
                if lag > 0:
                    time.sleep(lag)
            else:
                n, frame = reader.read(n, timeout=0.2)
                if frame is None:
                    if reader.stop:
                        break
                    dash.status.update(connected=False, lost_s=round(time.time() - t_frame, 1),
                                       uptime_s=int(time.time() - t0))
                    if journey:
                        journey.pause()
                        if dash.journey_req is not None:          # bấm bật/tắt lúc đang mất tín hiệu
                            journey.set_enabled(dash.journey_req)
                            print(f"[HANH TRINH] {'Bat' if dash.journey_req else 'Tat'} ghi hanh trinh", flush=True)
                            dash.journey_req = None
                        dash.status["journey"] = journey.status()
                    if args.view:
                        if dash.canvas is not None and time.time() - t_frame >= 1.0:
                            v = dash.canvas.copy()
                            cv2.putText(v, f"MAT TIN HIEU {time.time() - t_frame:.0f}s - dang cho ket noi lai...",
                                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                            cv2.imshow(WINDOW, v)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            print("Nhan q - dung.", flush=True)
                            break
                    continue
            t_frame = time.time()
            t = t_frame - t0

            # ---------------- AI (khung gốc), áp cấu hình từ web
            cfg = dash.cfg
            if fight is not None:
                fight.conf = cfg["thresholds"]["fight_conf"]
            fall.ratio = cfg["thresholds"]["lying_ratio"]
            if fire is not None:
                fire.conf = cfg["thresholds"]["fire_conf"]
            crowd.min_group = int(cfg["thresholds"]["crowd_min"])
            tick = time.time()
            fire_hit = fire.update(frame) if fire is not None and cfg["modules"]["fire"] else (False, 0.0)
            # pose chạy nếu cần cho NGÃ hoặc ĐÁM ĐÔNG
            pose_hit = fall.update(frame, t) if cfg["modules"]["fall"] or cfg["modules"]["crowd"] else (False, 0.0)
            fall_hit = pose_hit if cfg["modules"]["fall"] else (False, 0.0)
            fight_hit = fight.update(frame, t) if fight is not None and cfg["modules"]["fight"] else (False, 0.0)
            was_crowd = crowd.active
            crowd_hit = crowd.update(fall.people, t) if cfg["modules"]["crowd"] else (False, 0)
            if crowd.active and not was_crowd and dash.notifier:
                dash.notifier.crowd_started(crowd.largest, crowd.total)
            proc.append(time.time() - tick)

            # ---------------- vẽ + cảnh báo
            canvas = frame.copy()
            draw(canvas, cfg, fire, fall, fight, crowd, crowd_hit[0])
            if fire_hit[0]:
                alerts.fire(canvas, t, fire_hit[1])
            if fall_hit[0]:
                alerts.fall(canvas, t, fall_hit[1])
            if fight_hit[0]:
                alerts.violence(canvas, t, fight_hit[1])
            clips.add(t, canvas)

            # ---------------- ghi hình / chụp ảnh theo yêu cầu từ web
            if dash.record_req is not None:
                want, dash.record_req = dash.record_req, None
                if want and writer is None:
                    rec_path = OUT / "videos" / f"rec_{datetime.now():%Y%m%d_%H%M%S}.mp4"
                    h, w = canvas.shape[:2]
                    writer = cv2.VideoWriter(str(rec_path), cv2.VideoWriter_fourcc(*"mp4v"), src_fps, (w, h))
                    print(f"[REC] Bat dau ghi -> {rec_path}", flush=True)
                elif not want and writer is not None:
                    writer.release()
                    writer = None
                    print(f"[REC] Da luu {rec_path}", flush=True)
            if writer is not None:
                writer.write(canvas)
            if dash.journey_req is not None:
                if journey is None:     # chưa bật lúc khởi động -> tạo khi bấm nút
                    journey = JourneyRecorder(OUT / "videos", "ai", fps=src_fps, segment_min=args.segment_min)
                journey.set_enabled(dash.journey_req)
                print(f"[HANH TRINH] {'Bat' if dash.journey_req else 'Tat'} ghi hanh trinh", flush=True)
                dash.journey_req = None
            if journey:
                journey.write(frame, canvas)
            while dash.snapshot_req:
                ev, holder = dash.snapshot_req.pop(0)
                p = OUT / "snapshots" / f"chup_{datetime.now():%Y%m%d_%H%M%S_%f}.jpg"
                cv2.imwrite(str(p), canvas)
                holder["url"], holder["path"] = f"/manual/{p.name}", str(p)
                ev.set()

            # ---------------- cập nhật trạng thái cho web
            ai_fps = len(proc) / max(sum(proc), 1e-6)
            mods = {"fire": fire_hit, "fall": fall_hit, "fight": fight_hit, "crowd": crowd_hit}
            with dash.lock:
                dash.canvas, dash.seq = canvas, dash.seq + 1
                dash.status.update(connected=True, lost_s=0.0, ai_fps=round(ai_fps, 1),
                                   latency_ms=round(1000 * proc[-1], 0), frames=dash.status["frames"] + 1,
                                   uptime_s=int(time.time() - t0), recording=writer is not None,
                                   journey=journey.status() if journey else None,
                                   video_t=round(time.time() - t_file0, 1) if is_file else None,
                                   people=crowd.total if cfg["modules"]["crowd"] else len(fall.people),
                                   largest_group=crowd.largest,
                                   crowd={"active": crowd.active and cfg["modules"]["crowd"], "level": crowd.level(),
                                          "largest": crowd.largest, "peak": crowd.peak, "threshold": crowd.min_group,
                                          "since_s": round(t - crowd.since, 1) if crowd.active else 0})
                if not dash.people_hist or t - dash.people_hist[-1][0] >= 1.0:
                    dash.people_hist.append((round(t, 1), dash.status["people"], int(dash.status["crowd"]["active"])))
                for k, (hit, val) in mods.items():
                    m = dash.modules[k]
                    m["value"] = round(float(val), 2)
                    m["state"] = ("off" if not cfg["modules"][k] else
                                  "alarm" if hit else "ok")
            if args.view:
                cv2.imshow(WINDOW, canvas)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("Nhan q - dung.", flush=True)
                    break
    except KeyboardInterrupt:
        print("Dung theo yeu cau.", flush=True)
    finally:
        if writer is not None:
            writer.release()
            print(f"[REC] Da luu {rec_path}", flush=True)
        if journey:
            journey.close()
        clips.flush()
        cv2.destroyAllWindows()
        print(f"[{now_str()}] Ket thuc.", flush=True)
        sys.stdout.flush()
        os._exit(0)   # luồng web + luồng đọc camera là daemon: thoát thẳng


if __name__ == "__main__":
    main()
