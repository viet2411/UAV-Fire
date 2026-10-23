@echo off
REM ======================================================================
REM  UAV School-Safety AI - chay giam sat voi luong truc tiep tu UAV
REM  Mo cua so video tren may nay (nhan q de dung) + man hinh giam sat web
REM  (trinh duyet tu mo http://localhost:8000; dien thoai cung mang: http://<IP may nay>:8000)
REM
REM   1. Mo Aviateur: chon card thu, danh dau "Forward RTP", target port = 5600, bam Start (F5)
REM      (khi Forward RTP bat, Aviateur khong tu hien hinh - hinh hien o phan mem nay)
REM   2. Nhap dup file nay.
REM Nguon khac:  start.bat 0  (card thu USB)
REM ======================================================================
cd /d "%~dp0"
set PYTHONUTF8=1
REM giam bo dem cua FFmpeg de do tre thap
set OPENCV_FFMPEG_CAPTURE_OPTIONS=fflags;nobuffer^|flags;low_delay

set SRC=rtp://@:5600
if not "%~1"=="" set SRC=%~1

REM --- cai dat ---
REM Canh bao tu tap khi nhom >= N nguoi (san truong: 8-10)
set CROWD_MIN=8
set PORT=8000
REM Bo trong dong duoi neu khong dung phat hien chay/khoi
set FIRE=--fire-weights weights\best_dfire.pt
REM Mo trinh duyet tu dong (1 = co, 0 = khong)
set OPEN_BROWSER=1
REM Telegram: dien token + chat id vao config\telegram.json (xem HUONG_DAN.md)

REM --- tuy chon: tat cua so video tren may nay (chi dung web) -> doi thanh: set VIEW=
set VIEW=--view

REM --- tu dong ghi VIDEO HANH TRINH tu luc co hinh (outputs\videos\hanh_trinh_*.mp4, chia file 10 phut)
REM     ai = co khung AI  |  goc = hinh goc (du lieu de train lai AI)  |  ca_hai  |  tat: doi thanh  set JOURNEY=
set JOURNEY=--auto-record ai

REM --- Python: venv (setup_windows.bat) neu co, khong thi dung Python cua may ---
set PY=venv\Scripts\python.exe
if not exist "%PY%" (
    set PY=python
    py -3.12 --version >nul 2>&1 && set PY=py -3.12
)

%PY% -c "import cv2, ultralytics, fastapi" >nul 2>&1 || (echo [LOI] Python "%PY%" chua co thu vien can thiet - chay setup_windows.bat truoc. & pause & exit /b 1)
if "%OPEN_BROWSER%"=="1" start "" /b cmd /c "timeout /t 8 /nobreak >nul & start http://localhost:%PORT%"
%PY% uav_dashboard.py --source "%SRC%" --crowd-min %CROWD_MIN% --port %PORT% %FIRE% %VIEW% %JOURNEY%
pause
