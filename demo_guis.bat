@echo off
REM ThreatWatch review-GUI launcher (double-click friendly).
REM Opens the three presentation views:
REM   1. Netron     - interactive layer graph of the headline model
REM   2. XnViewMP   - browser for every result figure in results\
REM   3. TensorBoard - training curves at http://localhost:6006 (runs in WSL)
REM All three are installed already (scoop install netron xnviewmp; venv has tensorboard).
title ThreatWatch review GUIs
set "ROOT=C:\Users\DELL\threatwatch"

REM 1. Netron - the CNN layer graph (export with: python scripts/export_onnx.py --all)
if exist "%ROOT%\models\cnn_wnorm.onnx" (
  start "" "%USERPROFILE%\scoop\shims\netron.exe" "%ROOT%\models\cnn_wnorm.onnx"
) else (
  echo [gui] models\cnn_wnorm.onnx not found - run: python scripts\export_onnx.py --all
)

REM 2. XnViewMP - opens results\ as a folder, arrow-key through the confusion matrices
start "" "%USERPROFILE%\scoop\shims\xnviewmp.exe" "%ROOT%\results"

REM 3. TensorBoard - skip if something already serves :6006
%SystemRoot%\System32\curl.exe -s -o nul http://localhost:6006
if errorlevel 1 (
  echo [gui] starting TensorBoard in WSL, loading the browser takes a few seconds...
  start "" wsl -- bash -lc "cd /mnt/c/Users/DELL/threatwatch && exec /home/pratik/.venvs/threatwatch/bin/tensorboard --logdir results/tensorboard --port 6006"
) else (
  echo [gui] TensorBoard already running on :6006
)
start "" http://localhost:6006

echo.
echo Opened: Netron (model graph), XnViewMP (figures), TensorBoard (localhost:6006)
echo The TensorBoard console window owns the server - closing it stops TensorBoard.
echo.
pause >nul