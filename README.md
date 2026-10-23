# UAV School-Safety AI

Hệ thống AI giám sát an toàn trường học bằng UAV (drone): nhận hình trực tiếp từ UAV, **phát hiện sự cố**
(đánh nhau, người ngã, cháy / khói), **đếm người và theo dõi tụ tập đông người**, hiển thị trên **màn hình
giám sát web** và **cảnh báo qua Telegram**. Dự án nghiên cứu khoa học kỹ thuật của học sinh THPT.

```
UAV (OpenIPC) ──WFB──> card RTL8812AU ──> Aviateur ──RTP──> uav_dashboard.py (GPU)
                                                                     ├─> màn hình giám sát (trình duyệt, điện thoại)
                                                                     └─> Telegram (ảnh + clip + nút Đúng / Báo nhầm)
```

## Tính năng
| | Mô hình / cách làm | Kiểu cảnh báo |
|---|---|---|
| **Đánh nhau** | Fight-YOLOv8 small (imgsz 1280) + bỏ phiếu theo thời gian | Sự cố |
| **Người ngã** | YOLO26n-pose (17 khớp) + luật "khung bè ngang / thân nghiêng > 55°" trong 4 s | Sự cố |
| **Cháy / khói** | YOLO26s nhóm tự train trên D-Fire (mAP50 0.80) | Sự cố |
| **Đếm người, tụ tập** | Dùng lại kết quả pose; gom nhóm theo khoảng cách chuẩn hoá chiều cao người | Trạng thái theo dõi (không ghi nhật ký) |

- **Màn hình giám sát web**: video có khung + skeleton, trạng thái kết nối UAV, 4 thẻ module, nhật ký cảnh báo có ảnh
  và clip 5 s trước → 5 s sau, nút **Đúng / Báo nhầm** (đo độ chính xác ngoài thực tế), ghi hình, chụp ảnh, chỉnh ngưỡng.
- **Video hành trình**: tự ghi suốt buổi bay từ lúc có hình (có khung AI và/hoặc hình gốc), chia file 10 phút, dừng khi ổ gần đầy.
- **Telegram**: ảnh + mô tả tiếng Việt + nút Đúng / Báo nhầm (đồng bộ với màn hình), clip, lệnh `/trangthai`, `/anh`.
- **Nguồn video**: RTP từ Aviateur (`rtp://@:5600`, tự nhận H.264 / H.265), card thu USB / webcam, file video. Tự kết nối lại khi mất sóng.
- Tốc độ (RTX 4060 Ti, 720p): pose + đánh nhau + cháy ~50 fps.

## Chạy trên Windows
Yêu cầu: Windows 10/11, GPU NVIDIA (driver ≥ 570 cho PyTorch CUDA 12.8).

1. Tải repo: `git clone https://github.com/viet2411/UAV-Fire.git` (hoặc **Code → Download ZIP**), giải nén vào ví dụ `C:\UAV-Fire`.
2. Nhấp đúp **`setup_windows.bat`**: cài Python 3.12, PyTorch + CUDA, thư viện, rồi tự kiểm tra (`check_env.py`).
3. **Chạy thử không cần UAV**: đặt một video `.mp4` vào `videos\` (repo không kèm video), nhấp đúp **`start_demo.bat`**
   (hoặc `start_demo.bat videos\ten_video.mp4`). Trình duyệt mở **http://localhost:8000**.
4. **Chạy với UAV**: Aviateur → chọn card thu → đánh dấu **Forward RTP**, *target port* `5600` → **Start (F5)** → nhấp đúp
   **`start.bat`**.
   Card thu RTL8812AU cần driver **libusb**: chạy `tools\zadig-2.8.exe` → *Options → List All Devices* → chọn card → libusb → Install
   (Aviateur cũng cần `tools\vcredist_x64.exe`).
5. (Tuỳ chọn) **Telegram**: chép `config\telegram.example.json` thành `config\telegram.json`, điền token bot
   (@BotFather) và chat id — xem `HUONG_DAN.md`.

Hướng dẫn từng bước cho người bay: **`HUONG_DAN_NHANH.html`** (in được A4). Hướng dẫn đầy đủ: **`HUONG_DAN.md`**.
Cài ở nơi không có Internet: chạy `tai_goi_offline.bat` trước ở nơi có mạng.

## Chạy trên Linux / WSL
```bash
python3.12 -m venv venv && source venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
python uav_dashboard.py --source video.mp4 --loop --fire-weights weights/best_dfire.pt --view   # demo
python uav_dashboard.py --source rtp://@:5600 --fire-weights weights/best_dfire.pt             # UAV qua Aviateur
```
Phiên bản không có web (chỉ cửa sổ OpenCV): `python uav_pipeline.py --source ... --view`.
Mã nghiên cứu (train / đánh giá mô hình, đo skeleton theo độ cao) không nằm trong repo này.

## Cấu trúc
```
UAV-Fire/
├── start.bat              chạy với UAV qua Aviateur (cửa sổ video + màn hình giám sát web)
├── start_demo.bat         chạy thử bằng video trong videos\
├── run_live.bat / run_file.bat   bản dự phòng: chỉ cửa sổ OpenCV, không có web
├── setup_windows.bat      cài đặt (1 lần) · tai_goi_offline.bat: tải sẵn thư viện để cài không cần mạng
├── check_env.py           kiểm tra GPU, mô hình, thư viện, Telegram
├── uav_dashboard.py       chương trình chính: AI + web server + đếm người + Telegram
├── uav_pipeline.py        các bộ phát hiện, đọc luồng RTP / HTTP / USB, cảnh báo
├── notify_telegram.py     cảnh báo Telegram
├── web/dashboard.html     giao diện màn hình giám sát
├── config/telegram.example.json   mẫu cấu hình Telegram (chép thành telegram.json)
├── weights/               best_dfire.pt (cháy/khói, nhóm tự train) · yolo26n-pose.pt (người + skeleton)
├── third_party/           Fight-Violence-detection-yolov8: phát hiện đánh nhau
├── videos/                đặt video thử vào đây (repo không kèm video)
├── tools/                 zadig-2.8.exe (driver card thu) · vcredist_x64.exe (Aviateur cần)
├── HUONG_DAN_NHANH.html   hướng dẫn từng bước cho người bay (in được A4) · HUONG_DAN.md: hướng dẫn đầy đủ
└── requirements-windows.txt / requirements.txt
```
Khi chạy, kết quả (nhật ký, ảnh, clip, video ghi) lưu trong `outputs\`.

## Kết quả & giới hạn
- **Cháy / khói** (D-Fire, tập val): mAP50 **0.80**, mAP50-95 0.47, precision 0.79, recall 0.75. Học từ ảnh chụp mặt đất,
  chưa có ảnh cháy nhìn từ UAV.
- **Đánh nhau** (2 video UAV thử, nhãn người dùng xác nhận): **5/9** cảnh báo đúng (precision 56 %); hay nhầm khi 2 người
  đứng sát nhau; nhạy với nén video.
- **Người ngã**: bắt tốt người nằm duỗi; bỏ sót người nằm co.
- **Skeleton ở 15 m**: trung bình 12.5/17 khớp mỗi người → nên bay sao cho người cao ≳ 130 px.
- Đếm người: người bị che / đứng chồng lên nhau dễ bị đếm thiếu.

## Quyền riêng tư
Repo **không** chứa video, ảnh trích từ video hay dataset có hình người thật. Khi triển khai: xử lý trên máy chủ của trường,
chỉ chat Telegram được cấp quyền mới nhận ảnh, không dùng nhận diện khuôn mặt. Token Telegram để trong
`config/telegram.json` (đã bị `.gitignore` loại trừ).

## Giấy phép & nguồn
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) (YOLO26, pose) — AGPL-3.0.
- [Fight-Violence-detection-yolov8](https://github.com/Musawer1214/Fight-Violence-detection-yolov8) — MIT.
- [mmaction2](https://github.com/open-mmlab/mmaction2) ST-GCN++ (thử nghiệm) — Apache-2.0.
- Dataset: [D-Fire](https://github.com/gaia-solutions-on-demand/DFireDataset), VisDrone, Okutama-Action (không kèm trong repo).
- Phần mềm đi kèm trong `tools/`: [Zadig](https://zadig.akeo.ie/) 2.8, Microsoft Visual C++ Redistributable — theo giấy phép của từng tác giả. Nên dùng bản mới nhất từ trang chính thức;
  [Aviateur](https://github.com/OpenIPC/aviateur/releases) tải riêng.

Mã nguồn được phát triển với sự hỗ trợ của công cụ AI; nhóm tác giả chịu trách nhiệm kiểm chứng và điều chỉnh.
