"""
Cảnh báo qua Telegram cho uav_dashboard.py.

  * Sự cố (đánh nhau / ngã / cháy): ảnh + chú thích tiếng Việt + nút "✔ Đúng / ✖ Báo nhầm";
    5 s sau gửi tiếp clip 10 s (trả lời vào tin cảnh báo).
  * Bấm nút trong Telegram = bấm nút trên dashboard (cùng ghi outputs/alerts/feedback.csv), và ngược lại.
  * Tụ tập đông người: chỉ là trạng thái -> mặc định KHÔNG gửi; bật "notify_crowd" để gửi tin im lặng.
  * Lệnh trong chat: /trangthai, /anh (chụp ngay), /start (xem chat id).
  * Long polling (getUpdates): không cần IP công khai hay mở cổng.

Cấu hình: config/telegram.json  (KHÔNG đưa lên git - chứa token)
  {"token": "...", "chat_ids": [-100123..., 456...], "notify_crowd": false}
Chỉ các chat trong "chat_ids" mới nhận cảnh báo và được bấm nút / ra lệnh (ảnh có học sinh -> riêng tư).
"""
import json
import queue
import threading
import time
from pathlib import Path

import requests

API = "https://api.telegram.org/bot{token}/{method}"
KIND_VI = {"DANH_NHAU": ("🥊", "ĐÁNH NHAU"), "NGUOI_NGA": ("🧍", "NGƯỜI NGÃ"), "CHAY_KHOI": ("🔥", "CHÁY / KHÓI")}
VERDICT_VI = {"dung": "✔ Đúng", "nham": "✖ Báo nhầm", None: "chưa đánh giá"}


def load_config(path):
    p = Path(path)
    if not p.exists():
        return None
    cfg = json.loads(p.read_text(encoding="utf-8"))
    return cfg if cfg.get("token") else None


class TelegramNotifier:
    def __init__(self, cfg, dash=None, source_label=""):
        self.token = cfg["token"]
        self.chats = [int(c) for c in cfg.get("chat_ids", [])]
        self.notify_crowd = bool(cfg.get("notify_crowd", False))
        self.dash, self.source_label = dash, source_label
        self.msgs = {}                 # alert id -> [(chat_id, message_id)]
        self.q = queue.Queue()
        self.offset = None
        threading.Thread(target=self._sender, daemon=True).start()
        threading.Thread(target=self._poller, daemon=True).start()
        if not self.chats:
            print("[TELEGRAM] Chua co chat_ids trong config/telegram.json -> nhan /start cho bot de lay chat id.",
                  flush=True)

    # ------------------------------------------------------------------ low level
    def _call(self, method, files=None, timeout=20, **params):
        try:
            r = requests.post(API.format(token=self.token, method=method), data=params, files=files, timeout=timeout)
            d = r.json()
            if not d.get("ok"):
                print(f"  (Telegram {method} loi: {d.get('description')})", flush=True)
            return d.get("result")
        except Exception as e:
            print(f"  (Khong goi duoc Telegram {method}: {e})", flush=True)
            return None

    def _sender(self):
        """Một luồng gửi tuần tự: giữ thứ tự (ảnh trước, clip sau) và không làm chậm xử lý video."""
        while True:
            fn, args = self.q.get()
            try:
                fn(*args)
            except Exception as e:
                print("  (Telegram loi:", e, ")", flush=True)

    # ------------------------------------------------------------------ outgoing
    def _caption(self, a):
        icon, name = KIND_VI.get(a["kind"], ("⚠", a["kind"]))
        detail = (a["detail"].replace("do_tin_cay=", "độ tin cậy ").replace("thoi_gian_nam=", "nằm "))
        v = a.get("verdict")
        return (f"🚨 <b>{icon} {name}</b>  #{a['id']}\n"
                f"🕒 {a['time']}\n📊 {detail}\n📡 {self.source_label}\n"
                f"Đánh giá: <b>{VERDICT_VI[v]}</b>")

    def _keyboard(self, a):
        v = a.get("verdict")
        return json.dumps({"inline_keyboard": [[
            {"text": ("● " if v == "dung" else "") + "✔ Đúng", "callback_data": f"v:{a['id']}:dung"},
            {"text": ("● " if v == "nham" else "") + "✖ Báo nhầm", "callback_data": f"v:{a['id']}:nham"}]]})

    def send_alert(self, a, image_path):
        self.q.put((self._send_alert, (dict(a), image_path)))

    def _send_alert(self, a, image_path):
        for chat in self.chats:
            with open(image_path, "rb") as f:
                m = self._call("sendPhoto", files={"photo": f}, chat_id=chat, caption=self._caption(a),
                               parse_mode="HTML", reply_markup=self._keyboard(a))
            if m:
                self.msgs.setdefault(a["id"], []).append((chat, m["message_id"]))

    def send_clip(self, a, clip_path):
        self.q.put((self._send_clip, (dict(a), clip_path)))

    def _send_clip(self, a, clip_path):
        icon, name = KIND_VI.get(a["kind"], ("⚠", a["kind"]))
        for chat, mid in self.msgs.get(a["id"], []):
            with open(clip_path, "rb") as f:
                self._call("sendDocument", files={"document": f}, timeout=60, chat_id=chat,
                           caption=f"🎞 Clip {icon} {name} #{a['id']} (5 s trước → 5 s sau)",
                           reply_to_message_id=mid, disable_notification=True)

    def update_verdict(self, a):
        """Đánh giá đổi (từ dashboard hoặc Telegram) -> cập nhật chú thích + nút trong mọi chat."""
        self.q.put((self._update_verdict, (dict(a),)))

    def _update_verdict(self, a):
        for chat, mid in self.msgs.get(a["id"], []):
            self._call("editMessageCaption", chat_id=chat, message_id=mid, caption=self._caption(a),
                       parse_mode="HTML", reply_markup=self._keyboard(a))

    def send_text(self, text, silent=False, chats=None):
        for chat in (chats or self.chats):
            self.q.put((lambda c=chat: self._call("sendMessage", chat_id=c, text=text, parse_mode="HTML",
                                                  disable_notification=silent), ()))

    def crowd_started(self, group, total):
        if self.notify_crowd:
            self.send_text(f"👥 <b>Đang tụ tập đông người</b>\nNhóm lớn nhất {group} người · tổng {total} người",
                           silent=True)

    # ------------------------------------------------------------------ incoming (long polling)
    def _poller(self):
        while True:
            res = self._get_updates()
            for u in res or []:
                self.offset = u["update_id"] + 1
                try:
                    if "callback_query" in u:
                        self._on_callback(u["callback_query"])
                    elif "message" in u:
                        self._on_message(u["message"])
                except Exception as e:
                    print("  (Telegram xu ly update loi:", e, ")", flush=True)
            if res is None:
                time.sleep(5)

    def _get_updates(self):
        try:
            r = requests.get(API.format(token=self.token, method="getUpdates"),
                             params={"timeout": 30, "offset": self.offset,
                                     "allowed_updates": json.dumps(["message", "callback_query"])}, timeout=40)
            d = r.json()
            return d.get("result", []) if d.get("ok") else None
        except Exception:
            return None

    def _on_callback(self, cq):
        chat = cq["message"]["chat"]["id"]
        if chat not in self.chats:
            return self._call("answerCallbackQuery", callback_query_id=cq["id"], text="Chat này không có quyền.")
        _, aid, v = cq["data"].split(":")
        cur = next((x for x in self.dash.alerts if x["id"] == int(aid)), None) if self.dash else None
        new = None if cur and cur.get("verdict") == v else v          # bấm lại = bỏ chọn
        a = self.dash.set_verdict(int(aid), new) if self.dash else None
        who = cq["from"].get("first_name", "")
        self._call("answerCallbackQuery", callback_query_id=cq["id"],
                   text=f"Đã ghi: {VERDICT_VI[new]}" if a else "Không tìm thấy cảnh báo")
        if a:
            print(f"[TELEGRAM] {who} danh gia canh bao #{aid}: {new}", flush=True)

    def _on_message(self, m):
        chat, text = m["chat"]["id"], (m.get("text") or "").strip()
        cmd = text.split("@")[0].split()[0].lower() if text.startswith("/") else ""
        if cmd == "/start" or chat not in self.chats:
            if cmd == "/start":
                ok = "✅ Chat này ĐÃ được nhận cảnh báo." if chat in self.chats else \
                     "⚠ Chat này CHƯA được nhận cảnh báo. Thêm chat id vào config/telegram.json rồi chạy lại."
                self._call("sendMessage", chat_id=chat, text=f"UAV School-Safety AI\nChat id: <code>{chat}</code>\n{ok}",
                           parse_mode="HTML")
            if chat not in self.chats:
                print(f"[TELEGRAM] Tin nhan tu chat chua dang ky: id={chat} ({m['chat'].get('title') or m['chat'].get('first_name')})",
                      flush=True)
            return
        if cmd == "/trangthai" and self.dash:
            s = self.dash.snapshot()
            st, mods = s["status"], s["cfg"]["modules"]
            on = ", ".join(k for k, v in mods.items() if v)
            last = s["alerts"][-3:]
            lines = [f"📡 {'ĐÃ KẾT NỐI' if st['connected'] else 'MẤT TÍN HIỆU'} · AI {st['ai_fps']:.0f} fps",
                     f"👥 {st['people']} người · nhóm lớn nhất {st['largest_group']}" +
                     (" · <b>đang tụ tập</b>" if st["crowd"]["active"] else ""),
                     f"⚙ Module bật: {on}", f"🚨 Tổng cảnh báo: {len(s['alerts'])}"]
            lines += [f"  #{a['id']} {KIND_VI[a['kind']][1]} {a['time'][11:]}" for a in last]
            self._call("sendMessage", chat_id=chat, text="\n".join(lines), parse_mode="HTML")
        elif cmd == "/anh" and self.dash:
            ev, holder = threading.Event(), {}
            self.dash.snapshot_req.append((ev, holder))
            ev.wait(5)
            if holder.get("path"):
                with open(holder["path"], "rb") as f:
                    self._call("sendPhoto", files={"photo": f}, chat_id=chat, caption="📷 Ảnh hiện tại")
            else:
                self._call("sendMessage", chat_id=chat, text="Chưa có hình từ UAV.")
        elif cmd in ("/help", "/trogiup"):
            self._call("sendMessage", chat_id=chat, text="/trangthai – tình trạng hệ thống\n/anh – chụp ảnh hiện tại")
