"""
NeuroLinked Desktop Tray Launcher — V1.3

Fixes the V1.2 complaint: "ships as .bat scripts that pop open a black
terminal window." This runs the brain in the background as a proper
tray app. Right-click the tray icon to:
  - Open dashboard in browser
  - Stop the brain
  - Show status / logs
  - Check for updates

Uses pystray + PIL for the tray icon (both are small). Falls back to
a console banner if pystray isn't installed.

Built as a wrapper — it runs `python run.py` as a subprocess in the
background and manages it via tray menu. No Electron, no Tauri,
no Node.js.
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
import urllib.request
import webbrowser
from typing import Optional


APP_NAME = "NeuroLinked"
DASHBOARD_URL = "http://localhost:8000"
PORT = 8000


def _app_root() -> str:
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class BrainProcess:
    """Manages the brain subprocess."""

    def __init__(self):
        self.process: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()

    def start(self) -> bool:
        with self._lock:
            if self.process and self.process.poll() is None:
                return True  # already running
            root = _app_root()
            # Prefer compiled .exe if present; else run run.py with Python
            exe = os.path.join(root, "NeuroLinked.exe")
            cmd = [exe] if os.path.exists(exe) else [sys.executable, os.path.join(root, "run.py")]
            # CREATE_NO_WINDOW on Windows so no terminal pops up
            creationflags = 0
            if sys.platform == "win32":
                creationflags = 0x08000000  # CREATE_NO_WINDOW
            try:
                self.process = subprocess.Popen(
                    cmd, cwd=root,
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    creationflags=creationflags,
                )
                return True
            except Exception as e:
                print(f"[TRAY] Failed to start brain: {e}")
                return False

    def stop(self) -> bool:
        with self._lock:
            if self.process and self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                self.process = None
                return True
        return False

    def is_running(self) -> bool:
        with self._lock:
            return self.process is not None and self.process.poll() is None

    def dashboard_reachable(self) -> bool:
        try:
            urllib.request.urlopen(DASHBOARD_URL, timeout=1)
            return True
        except Exception:
            return False


class UpdateChecker:
    """Checks a version URL for newer releases. Config in brain_state/updater.json."""

    def __init__(self, manifest_url: Optional[str] = None):
        self.manifest_url = manifest_url or os.environ.get(
            "NLKD_UPDATE_MANIFEST_URL",
            # Default: empty = disabled. Users set this to their update server.
            "",
        )
        self.current_version = "1.3.0"
        self._latest_info: Optional[dict] = None
        self._last_check_ts = 0.0

    def check(self) -> dict:
        if not self.manifest_url:
            return {"ok": False, "error": "update URL not configured",
                    "current": self.current_version}
        try:
            with urllib.request.urlopen(self.manifest_url, timeout=5) as r:
                import json
                data = json.loads(r.read().decode("utf-8"))
            self._latest_info = data
            self._last_check_ts = time.time()
            latest = data.get("version", "0.0.0")
            newer = _version_gt(latest, self.current_version)
            return {
                "ok": True,
                "current": self.current_version,
                "latest": latest,
                "update_available": newer,
                "download_url": data.get("download_url", ""),
                "notes": data.get("notes", ""),
            }
        except Exception as e:
            return {"ok": False, "error": f"{type(e).__name__}: {e}",
                    "current": self.current_version}


def _version_gt(a: str, b: str) -> bool:
    """Semver-ish compare: returns True if a > b."""
    try:
        pa = [int(x) for x in a.split(".")[:3]]
        pb = [int(x) for x in b.split(".")[:3]]
        while len(pa) < 3: pa.append(0)
        while len(pb) < 3: pb.append(0)
        return pa > pb
    except Exception:
        return False


def run_tray():
    """Main entry — create tray icon + menu."""
    brain = BrainProcess()
    updater = UpdateChecker()
    brain.start()

    try:
        import pystray  # type: ignore
        from PIL import Image, ImageDraw  # type: ignore
    except ImportError:
        print("=" * 50)
        print(" NeuroLinked — Tray mode not available")
        print(" Install pystray + Pillow for the tray icon:")
        print("   pip install pystray Pillow")
        print("=" * 50)
        print(" Brain is running anyway. Open:", DASHBOARD_URL)
        print(" Press Ctrl+C to quit.")
        try:
            while True:
                time.sleep(60)
        except KeyboardInterrupt:
            brain.stop()
        return

    # Build a simple 64x64 icon
    img = Image.new("RGB", (64, 64), color=(8, 2, 20))
    d = ImageDraw.Draw(img)
    # Neon-cyan brain dot matrix
    for x in range(8, 56, 6):
        for y in range(8, 56, 6):
            # Pseudo-random pattern
            if (x * 7 + y * 13) % 17 < 8:
                d.ellipse((x-2, y-2, x+2, y+2), fill=(0, 240, 255))

    def on_open(icon, item):
        webbrowser.open(DASHBOARD_URL)

    def on_stop_brain(icon, item):
        brain.stop()

    def on_start_brain(icon, item):
        brain.start()

    def on_status(icon, item):
        running = brain.is_running()
        reachable = brain.dashboard_reachable() if running else False
        msg = f"brain_running={running}  dashboard_reachable={reachable}"
        print(f"[TRAY] {msg}")

    def on_check_update(icon, item):
        info = updater.check()
        if info.get("update_available"):
            print(f"[TRAY] Update available: {info['current']} -> {info['latest']}")
        else:
            print(f"[TRAY] {info}")

    def on_quit(icon, item):
        brain.stop()
        icon.stop()

    menu = pystray.Menu(
        pystray.MenuItem("Open Dashboard", on_open, default=True),
        pystray.MenuItem("Start Brain", on_start_brain),
        pystray.MenuItem("Stop Brain", on_stop_brain),
        pystray.MenuItem("Status", on_status),
        pystray.MenuItem("Check for Updates", on_check_update),
        pystray.MenuItem("Quit", on_quit),
    )
    icon = pystray.Icon(APP_NAME, img, APP_NAME, menu)

    # Open dashboard shortly after startup
    def _open_after_ready():
        for _ in range(30):
            time.sleep(1)
            if brain.dashboard_reachable():
                webbrowser.open(DASHBOARD_URL)
                return
    threading.Thread(target=_open_after_ready, daemon=True).start()

    icon.run()


if __name__ == "__main__":
    run_tray()
