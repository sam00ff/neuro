#!/bin/bash
# NeuroLinked Tray Launcher — macOS / Linux
cd "$(dirname "$0")/.."
if command -v python3 >/dev/null 2>&1; then
    python3 desktop/tray_launcher.py &
elif command -v python >/dev/null 2>&1; then
    python desktop/tray_launcher.py &
else
    echo "Python 3 required. Install with: brew install python3 (Mac) or apt install python3 (Linux)"
    exit 1
fi
