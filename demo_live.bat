@echo off
REM ThreatWatch live demo launcher (double-click friendly).
REM Runs the WSL pipeline; the launcher auto-spawns the Windows BLE/serial
REM bridges itself. Pass --wired for USB mode:  demo_live.bat --wired
title ThreatWatch live demo
wsl -- bash -lc "cd /mnt/c/Users/DELL/threatwatch && exec /home/pratik/.venvs/threatwatch/bin/python scripts/demo_live.py %*"
echo.
echo Demo stopped. Press any key to close.
pause >nul
