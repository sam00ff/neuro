"""
Demo Tenant Reaper — runs inside the demo container.

Scans tenant_manager's registry every 5 minutes and deletes ephemeral
demo tenants that haven't been active in the last 15 minutes. Prevents
unbounded growth on the public demo server.

Tenants are considered "ephemeral demo" if they have metadata
`{"demo": true}`. Real tenants (admin-created) are never reaped.
"""

from __future__ import annotations

import os
import threading
import time

from brain.tenant import tenant_manager


IDLE_TIMEOUT_SEC = int(os.environ.get("NLKD_DEMO_IDLE_TIMEOUT_SEC", "900"))
SCAN_INTERVAL_SEC = int(os.environ.get("NLKD_DEMO_SCAN_INTERVAL_SEC", "300"))


def reap_once() -> dict:
    """One reap pass. Returns {reaped_count, checked_count}."""
    now = time.time()
    reaped = 0
    checked = 0
    for t in list(tenant_manager.list_tenants()):
        checked += 1
        if not t.metadata.get("demo"):
            continue
        idle_for = now - t.last_seen
        if idle_for > IDLE_TIMEOUT_SEC:
            tenant_manager.delete_tenant(t.tenant_id, wipe_data=True)
            reaped += 1
            print(f"[DEMO] Reaped idle tenant {t.name!r} (idle {idle_for:.0f}s)")
    return {"checked": checked, "reaped": reaped, "ts": now}


def start_reaper_thread() -> threading.Thread:
    def _loop():
        while True:
            try:
                r = reap_once()
                if r["reaped"]:
                    print(f"[DEMO] Reap pass complete: {r}")
            except Exception as e:
                print(f"[DEMO] Reap error: {e}")
            time.sleep(SCAN_INTERVAL_SEC)
    t = threading.Thread(target=_loop, name="demo-reaper", daemon=True)
    t.start()
    return t


if __name__ == "__main__":
    print(f"[DEMO] Reaper starting — idle_timeout={IDLE_TIMEOUT_SEC}s, "
          f"scan_interval={SCAN_INTERVAL_SEC}s")
    start_reaper_thread()
    while True:
        time.sleep(3600)
