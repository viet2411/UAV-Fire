@echo off
setlocal
cd /d "%~dp0"
set PYTHONUTF8=1
echo ============================================================
echo  UAV School-Safety AI - cai dat moi truong (Windows)
echo ============================================================

REM Co thu muc wheels\ (tao bang tai_goi_offline.bat) -> cai KHONG can Internet
set PIPSRC=
if exist wheels\ (
    set PIPSRC=--no-index --find-links wheels
    echo [i] Tim thay thu muc wheels\ - cai dat OFFLINE
)

echo [1/4] Kiem tra Python 3.12 ...
py -3.12 --version >nul 2>&1
if errorlevel 1 (
    echo   Chua co Python 3.12 - dang cai bang "py install 3.12" ...
    py install 3.12
    py -3.12 --version >nul 2>&1
    if errorlevel 1 (
        echo [LOI] Khong cai duoc Python 3.12. Tai ban 3.12 tai https://www.python.org/downloads/
        goto :fail
    )
)
py -3.12 --version

echo [2/4] Tao moi truong venv ...
if not exist venv\Scripts\python.exe (
    py -3.12 -m venv venv || goto :fail
)
venv\Scripts\python -m pip install --upgrade pip %PIPSRC% || goto :fail

echo [3/4] Cai PyTorch + CUDA 12.8 (khoang 2.8 GB, co the mat vai phut) ...
REM cu128 bat buoc cho GPU RTX 50xx (Blackwell); van chay tren RTX 20/30/40
if defined PIPSRC (
    venv\Scripts\python -m pip install torch==2.11.0 torchvision==0.26.0 %PIPSRC% || goto :fail
) else (
    venv\Scripts\python -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128 || goto :fail
)

echo [4/4] Cai cac thu vien con lai ...
venv\Scripts\python -m pip install -r requirements-windows.txt %PIPSRC% || goto :fail

echo.
echo Kiem tra moi truong ...
venv\Scripts\python check_env.py || goto :fail
echo.
echo [OK] Cai dat xong. Chay thu: start_demo.bat   ^|   Chay voi UAV: start.bat
pause
exit /b 0

:fail
echo.
echo [LOI] Cai dat that bai - xem thong bao phia tren.
pause
exit /b 1
