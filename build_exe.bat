@echo off
title NeuroLinked - Build Executable
color 0B
echo.
echo  ============================================
echo   NEUROLINKED - Build Protected .exe
echo  ============================================
echo.
echo  This will bundle all Python code into a
echo  single opaque .exe so users cannot read
echo  the source code.
echo.
echo  Output: dist\NeuroLinked\NeuroLinked.exe
echo.

:: ---- Check PyInstaller ----
python -c "import PyInstaller" >nul 2>&1
if %errorlevel% neq 0 (
    echo [1/4] Installing PyInstaller...
    pip install --quiet pyinstaller
    if %errorlevel% neq 0 (
        echo  ERROR: PyInstaller install failed.
        pause
        exit /b 1
    )
) else (
    echo [1/4] PyInstaller already installed.
)

:: ---- Clean previous build ----
echo.
echo [2/4] Cleaning previous build...
if exist "build" rmdir /s /q build
if exist "dist" rmdir /s /q dist

:: ---- Run PyInstaller ----
echo.
echo [3/4] Building executable (this takes 2-5 minutes)...
pyinstaller NeuroLinked.spec --clean --noconfirm
if %errorlevel% neq 0 (
    echo  ERROR: Build failed.
    pause
    exit /b 1
)

:: ---- Copy dashboard and support files ----
echo.
echo [4/4] Copying dashboard and support files...
xcopy /E /I /Y dashboard "dist\NeuroLinked\dashboard" >nul
copy /Y README.md "dist\NeuroLinked\" >nul 2>&1
copy /Y UPDATE_GUIDE.md "dist\NeuroLinked\" >nul 2>&1
copy /Y CLAUDE.md "dist\NeuroLinked\" >nul 2>&1

:: Create empty brain_state so first run has a place to write
mkdir "dist\NeuroLinked\brain_state" 2>nul

:: Create start.bat for users
(
    echo @echo off
    echo title NeuroLinked Brain
    echo color 0B
    echo echo Starting NeuroLinked Brain...
    echo echo Dashboard: http://localhost:8000
    echo echo.
    echo NeuroLinked.exe
    echo pause
) > "dist\NeuroLinked\start.bat"

echo.
echo  ============================================
echo   BUILD COMPLETE!
echo  ============================================
echo.
echo   Distribution folder: dist\NeuroLinked\
echo   Launcher:            dist\NeuroLinked\start.bat
echo.
echo   All .py source files are hidden inside
echo   the .exe bundle. Users only see:
echo     - NeuroLinked.exe
echo     - dashboard\ (HTML/CSS/JS)
echo     - brain_state\ (their data)
echo.
echo  ============================================
pause
