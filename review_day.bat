@echo off
REM ============================================================
REM  ThreatWatch — FINAL REVIEW one-click launcher
REM  Double-click this on demo day. It starts the dashboard
REM  (Windows Python so the webcam works) and opens the browser.
REM  Board connects from the website itself: Connect button.
REM ============================================================
title ThreatWatch review launcher

REM Kill any zombie server left over inside WSL (it has no webcam/BLE/serial
REM and would silently answer on port 8000 instead of this launch).
wsl -- bash -lc "pkill -f threatwatch/run.py 2>/dev/null; pkill -f 'run.py' 2>/dev/null; true" >nul 2>&1

REM Is the dashboard already up AND healthy?
curl -s -o nul -w "%%{http_code}" --max-time 3 http://localhost:8000/api/status | findstr "200" >nul
if %errorlevel%==0 goto already_up

echo Starting ThreatWatch dashboard (log window opening)...
start "ThreatWatch Server Logs" cmd /k ""C:\Users\DELL\AppData\Local\Programs\Python\Python310\python.exe" -u "C:\Users\DELL\threatwatch\run.py""

REM Give the model + camera time to load
set /a tries=0
:waitloop
timeout /t 2 /nobreak >nul
curl -s -o nul -w "%%{http_code}" --max-time 2 http://localhost:8000/ | findstr "200" >nul
if %errorlevel%==0 goto up
set /a tries+=1
if %tries% lss 30 goto waitloop
echo Server did not come up in 60s - check the server window for errors.
pause
exit /b 1

:already_up
echo Dashboard already running.

:up
echo Opening dashboard...
start "" http://localhost:8000/
echo.
echo ============================================================
echo   DEMO SEQUENCE
echo   1. Wear the rig, finger on the MAX30102, power bank on.
echo   2. On the website: pick the COM port (or BLE), click Connect.
echo   3. Show live vitals, then press MULTIMODAL THREAT.
echo   4. Reset alert when done (Alert Reset button).
echo
echo   The "ThreatWatch Server Logs" window shows the live
echo   pipeline log (model, fusion, alerts). Keep it open.
echo ============================================================
echo This window can be closed; keep the server window running.
pause
