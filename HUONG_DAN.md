# UAV School-Safety AI — Hướng dẫn chạy trên Windows

Giám sát an toàn trường học bằng UAV: phát hiện **đánh nhau**, **người ngã**, **cháy / khói**,
đếm người và theo dõi **tụ tập đông người**. Màn hình giám sát chạy trên trình duyệt, cảnh báo về **Telegram**.

```
UAV (OpenIPC) ─WFB─> card RTL8812AU ─> Aviateur ─RTP─> uav_dashboard.py (GPU) ─> trình duyệt + Telegram
```

## 1. Cài đặt (làm 1 lần)
Yêu cầu: Windows 10/11, GPU NVIDIA, **driver NVIDIA ≥ 570** (PyTorch bản CUDA 12.8), Internet (hoặc gói offline, mục 1b).

Nhấp đúp **`setup_windows.bat`**: cài Python 3.12 (nếu chưa có), tạo `venv\`, cài PyTorch 2.11 + CUDA 12.8 (~2.8 GB)
và thư viện trong `requirements-windows.txt`, rồi chạy `check_env.py`.
Kết quả đúng: `CUDA: True`, các dòng `[OK]` cho 4 mô hình, video, dashboard.

### 1b. Cài ở nơi KHÔNG có Internet (nơi thi)
1. Ở nhà (có mạng): chạy `setup_windows.bat`, rồi **`tai_goi_offline.bat`** → tải mọi thư viện vào `wheels\` (~3 GB).
2. Chép **cả thư mục** `UAV-Fire` sang máy thi. Nếu máy thi chưa có Python 3.12: tải sẵn bộ cài
   *Windows installer (64-bit)* tại python.org, cài trước.
3. Trên máy thi chạy `setup_windows.bat` → tự cài từ `wheels\`, không cần mạng.
   (Telegram vẫn cần Internet; màn hình giám sát trong mạng LAN thì không.)

## 2. Chạy thử (không cần UAV)
Nhấp đúp **`start_demo.bat`** → cửa sổ video + trình duyệt tự mở **http://localhost:8000**, phát `videos\2.mp4` lặp lại (cảnh đánh nhau).
`start_demo.bat videos\1.mp4` → cảnh người ngã.

## 3. Chạy với UAV
Nhấp đúp **`start.bat`** (mặc định nguồn Aviateur `rtp://@:5600`) → mở **cửa sổ video** trên máy này (nhấn **q** để dừng)
và **màn hình giám sát web** (trình duyệt tự mở). Nguồn khác: truyền làm tham số, ví dụ `start.bat 0` (card thu USB).
**Video hành trình** tự ghi ngay khi có hình (`set JOURNEY=--auto-record ai` trong `start.bat`; `goc` = hình gốc để làm dữ liệu
train lại AI, `ca_hai` = cả hai, để trống = tắt). Bật/tắt khi đang chạy bằng nút **🎥 Hành trình**. Còn < 2 GB trống thì tự dừng ghi. Cài đặt (ngưỡng đám đông, cổng, tắt cửa sổ video `set VIEW=`, tắt cháy/khói) sửa ở đầu file bằng Notepad.
Chưa chạy `setup_windows.bat` thì `start.bat` thử dùng Python có sẵn của máy (cần đủ thư viện).

| Phần mềm thu | Nguồn (`SRC` trong start.bat / tham số) | Ghi chú |
|---|---|---|
| **Aviateur** (OpenIPC, ≥ v0.3.3) | `rtp://@:5600` (mặc định) | Chọn card → đánh dấu **Forward RTP** → **target port** `5600` → **Start (F5)**. Khi chuyển tiếp, Aviateur **không tự hiện hình** (chỉ chuyển tiếp, không giải mã). Card cần driver **libusb** (`tools\zadig-2.8.exe`) + Microsoft Visual C++ Redistributable (`tools\vcredist_x64.exe`). Tự nhận H.264 / H.265. **Cổng 5601 cũng phải trống** (FFmpeg dùng cho RTCP) |
| Card thu USB / webcam | `0` | |

- Nên đặt **GOP (khoảng keyframe) của camera OpenIPC ~1 s**: GOP dài → chờ lâu mới có hình và mỗi lần nối lại chậm.
- Mất sóng: màn hình hiện **MẤT TÍN HIỆU**, tự kết nối lại (tối đa ~5 s sau khi có sóng trở lại).
- Máy khác / điện thoại cùng WiFi: mở `http://<IP máy chủ>:8000` (lần đầu Windows Firewall hỏi → **Cho phép** cho Python, mạng Private).

## 4. Màn hình giám sát
| Khu vực | Chức năng |
|---|---|
| Thanh trên | Trạng thái UAV, số người · nhóm lớn nhất, tốc độ AI, độ trễ, GPU, **🔇/🔊 âm thanh** (bấm 1 lần để bật) |
| Video | Khung người, **skeleton**, khung nhóm người; viền đỏ khi có sự cố; banner vàng khi **đang tụ tập** |
| 4 thẻ module | Bật/tắt từng module; **đỏ = sự cố**, **vàng cam = đang tụ tập**; tỉ lệ cảnh báo đúng |
| Nhật ký cảnh báo | Ảnh (bấm để xem ảnh lớn + **clip 5 s trước → 5 s sau**), nút **✔ Đúng / ✖ Báo nhầm** |
| Thanh điều khiển | Ghi hình, chụp ảnh, bật/tắt khung - skeleton, ngưỡng đánh nhau / ngã / cháy / đám đông, thời gian chờ |
| Dòng thời gian | Vạch sự cố, dải tụ tập, đồ thị số người |

**Sự cố** (đánh nhau, ngã, cháy) → ghi nhật ký + ảnh + clip + Telegram.
**Tụ tập đông người** là *trạng thái theo dõi* → chỉ hiện banner / thẻ / dải thời gian, **không** ghi nhật ký.

Nút **Đúng / Báo nhầm** ghi vào `outputs\alerts\feedback.csv` → tự tính tỉ lệ đúng ngoài thực tế và gom dữ liệu có nhãn.

## 5. Cảnh báo qua Telegram
1. Telegram → @BotFather → `/newbot` → lấy **token**.
2. Chép `config\telegram.example.json` thành `config\telegram.json`, dán token.
3. Chạy `start.bat`, nhắn **`/start`** cho bot (hoặc thêm bot vào nhóm rồi gửi `/start` trong nhóm) → bot trả lời **chat id**.
4. Thêm chat id vào `"chat_ids"` trong `config\telegram.json` (nhóm: số âm `-100…`), chạy lại.

Tin cảnh báo: ảnh + mô tả + nút **✔ Đúng / ✖ Báo nhầm** (đồng bộ với màn hình giám sát) + clip gửi sau ~5 s.
Lệnh: `/trangthai`, `/anh`. Tụ tập đông người: mặc định không gửi (`"notify_crowd": true` → gửi tin im lặng).
Chỉ các chat có trong `chat_ids` mới nhận cảnh báo / bấm nút được.

⚠ `config\telegram.json` chứa token: **không gửi / không đưa lên mạng** thư mục có file này. Lộ token → @BotFather `/revoke`.

## 6. Kết quả lưu ở đâu
| Thư mục | Nội dung |
|---|---|
| `outputs\alerts\alert_log.csv` | Nhật ký sự cố (thời gian, loại, độ tin cậy, ảnh) |
| `outputs\alerts\snapshots\` · `clips\` | Ảnh và clip từng sự cố |
| `outputs\alerts\feedback.csv` | Đánh giá Đúng / Báo nhầm |
| `outputs\videos\hanh_trinh_*.mp4` | **Video hành trình** tự ghi từ lúc có hình (chia file 10 phút; `_goc` = hình gốc không có khung AI) |
| `outputs\videos\rec_*.mp4` · `outputs\snapshots\` | Video ghi và ảnh chụp bằng nút trên màn hình |

## 7. Cách chạy khác
- **`run_live.bat` / `run_file.bat`**: phiên bản cửa sổ OpenCV (không có web) — dự phòng khi trình duyệt có vấn đề.
- Tuỳ chọn nâng cao (`uav_dashboard.py`): `--fight-confirm-weights third_party\Fight-Violence-detection-yolov8\Yolo_nano_weights.pt`
  (bản nano xác nhận chéo: ít báo nhầm hơn, có thể sót), `--imgsz 960` (máy yếu), `--no-skeleton`, `--cooldown 30`.

## 8. Giới hạn đã biết
- **Đánh nhau**: trên 2 video thử, 5/9 cảnh báo đúng (56 %); hay báo nhầm khi 2 người đứng sát nhau; mô hình nhạy với nén video.
- **Người ngã**: bắt tốt người nằm duỗi; bỏ sót người nằm co.
- **Đếm người**: người bị che / đứng chồng lên nhau dễ bị đếm thiếu.
- **Cháy / khói**: mô hình học từ ảnh chụp mặt đất (D-Fire), chưa có ảnh cháy nhìn từ UAV.

## 9. Lỗi thường gặp
| Hiện tượng | Cách xử lý |
|---|---|
| `CUDA: False` | Cập nhật driver NVIDIA (≥ 570); chạy lại `setup_windows.bat` |
| Trình duyệt mở nhưng chưa có hình | Chờ nạp mô hình (~10 s); xem cửa sổ đen có báo "cho nguon video" không |
| `Chua nhan duoc goi RTP ... cong 5600` lặp lại | Aviateur chưa bật chuyển tiếp, sai cổng, hoặc chương trình khác chiếm cổng 5600/5601 |
| Điện thoại không mở được trang | Cùng mạng WiFi? Cho phép Python qua Windows Firewall; dùng IP in ra ở dòng `[WEB]` |
| Không nhận Telegram | `check_env.py` → dòng Telegram; máy có Internet?; chat id đã thêm vào `chat_ids` chưa |
| Cửa sổ `.bat` đóng ngay | Mở `cmd`, `cd C:\UAV-Fire`, chạy file `.bat` để đọc lỗi |
