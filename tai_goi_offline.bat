@echo off
REM Chay 1 lan tren may CO INTERNET: tai san moi thu vien vao thu muc wheels\ (~3 GB)
REM -> sau do setup_windows.bat cai duoc ma KHONG can mang (vd tai noi thi).
REM Python 3.12: tai them file cai dat tai https://www.python.org/downloads/ (Windows installer 64-bit)
REM              va de vao thu muc nay neu may thi chua co Python 3.12.
setlocal
cd /d "%~dp0"
set PYTHONUTF8=1
py -3.12 --version >nul 2>&1 || (echo [LOI] Can Python 3.12 de tai dung goi. Chay setup_windows.bat truoc. & pause & exit /b 1)
if not exist wheels mkdir wheels
echo [1/3] Tai pip ...
py -3.12 -m pip download pip -d wheels || goto :fail
echo [2/3] Tai PyTorch 2.11.0 + CUDA 12.8 (~2.8 GB) ...
py -3.12 -m pip download torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128 -d wheels || goto :fail
echo [3/3] Tai cac thu vien con lai ...
py -3.12 -m pip download -r requirements-windows.txt --extra-index-url https://download.pytorch.org/whl/cu128 -d wheels || goto :fail
echo.
echo [OK] Da tai xong vao thu muc wheels\  - chep ca thu muc UAV-Fire sang may thi.
pause
exit /b 0
:fail
echo [LOI] Tai that bai - kiem tra mang.
pause
exit /b 1
