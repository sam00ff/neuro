@echo off
:: NeuroLinked Tray Launcher — Windows
:: Double-click this to start the brain in system tray (no black window).
cd /d "%~dp0\.."
where pythonw >nul 2>&1
if %errorlevel% == 0 (
    start "" pythonw.exe "desktop\tray_launcher.py"
) else (
    start "" python.exe "desktop\tray_launcher.py"
)
exit /b 0
