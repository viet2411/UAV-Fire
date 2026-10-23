@echo off
REM Chay giam sat voi fpv4win (ban co Stream Server).
REM Truoc do: mo fpv4win, cho co hinh, bam Start Stream, TAT VLC.
cd /d "%~dp0"
call start.bat http://127.0.0.1:8080/stream.ts
