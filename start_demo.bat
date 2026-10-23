@echo off
REM Demo bang video mau (phat lai lien tuc, toc do that). Khong can UAV.
REM   start_demo.bat               -> videos\2.mp4 (danh nhau)
REM   start_demo.bat videos\1.mp4  -> nguoi nga
REM Cua so video tren may nay (nhan q de dung) + man hinh giam sat web http://localhost:8000
cd /d "%~dp0"
set PYTHONUTF8=1
set SRC=%~1
if "%SRC%"=="" set SRC=videos\2.mp4
if not exist "%SRC%" (echo [LOI] Khong thay video "%SRC%". Dat mot video .mp4 vao thu muc videos\ roi chay: start_demo.bat videos\ten_video.mp4 & pause & exit /b 1)
set PY=venv\Scripts\python.exe
if not exist "%PY%" (
    set PY=python
    py -3.12 --version >nul 2>&1 && set PY=py -3.12
)
%PY% -c "import cv2, ultralytics, fastapi" >nul 2>&1 || (echo [LOI] Python "%PY%" chua co thu vien can thiet - chay setup_windows.bat truoc. & pause & exit /b 1)
start "" /b cmd /c "timeout /t 8 /nobreak >nul & start http://localhost:8000"
%PY% uav_dashboard.py --source "%SRC%" --loop --crowd-min 4 --fire-weights weights\best_dfire.pt --view
pause
