@echo off
REM Chay thu bang file video.  Cach dung:  run_file.bat            (mac dinh videos\2.mp4)
REM                                        run_file.bat videos\1.mp4
cd /d "%~dp0"
set PYTHONUTF8=1
set SRC=%~1
if "%SRC%"=="" set SRC=videos\2.mp4
venv\Scripts\python uav_pipeline.py --source "%SRC%" --view --fire-weights weights\best_dfire.pt --save-video out_test.mp4
echo.
echo Canh bao + anh: outputs\alerts\    Video ket qua: outputs\videos\out_test.mp4
pause
