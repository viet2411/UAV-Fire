@echo off
REM Chay voi luong truc tiep tu Aviateur (OpenIPC): video RTP chuyen tiep ra cong UDP 5600.
REM   1. Mo Aviateur: chon card thu, danh dau "Forward RTP", target port = 5600, bam Start (F5)
REM      (khi Forward RTP bat, Aviateur khong tu hien hinh - hinh hien o phan mem nay)
REM   2. Chay file nay.   Nhan q trong cua so video hoac Ctrl+C de dung.
REM Nguon khac:  run_live.bat http://127.0.0.1:8080/stream.ts  (fpv4win)  |  run_live.bat 0  (card thu USB)
cd /d "%~dp0"
set PYTHONUTF8=1
REM giam bo dem cua FFmpeg de do tre thap
set OPENCV_FFMPEG_CAPTURE_OPTIONS=fflags;nobuffer^|flags;low_delay

set SRC=rtp://@:5600
if not "%~1"=="" set SRC=%~1

REM --- tuy chon: gui canh bao qua Telegram (dien token va chat id) ---
set TELEGRAM_TOKEN=
set TELEGRAM_CHAT=
set TG=
if not "%TELEGRAM_TOKEN%"=="" set TG=--telegram-token %TELEGRAM_TOKEN% --telegram-chat %TELEGRAM_CHAT%

REM --- tuy chon: ghi video ket qua (bo REM o dong duoi) ---
set SAVE=
REM set SAVE=--save-video live_%date:~-4%%date:~3,2%%date:~0,2%.mp4

venv\Scripts\python uav_pipeline.py --source "%SRC%" --view --fire-weights weights\best_dfire.pt %TG% %SAVE%
pause
