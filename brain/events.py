"""
Event Bus + Webhooks + SSE — V1.3

Business integrations need events. "Tell me when a new memory is
stored. Tell me when the agent completes a plan. Tell me when an
insight fires." Before V1.3, zero support — the WebSocket was 3D
visualization only.

Three transports for the same event stream:
  1. In-process pub/sub (fastest — used by the server itself)
  2. Webhooks — POST to any URL the user registered
  3. SSE — Server-Sent Events endpoint, keeps a browser/mobile open
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Optional


def _app_root() -> str:
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _webhooks_path() -> str:
    d = os.path.join(_app_root(), "brain_state")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, "webhooks.json")


@dataclass
class Event:
    id: str
    type: str
    ts: float
    payload: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Webhook:
    webhook_id: str
    url: str
    event_types: List[str]            # ["*"] = all; otherwise specific types
    secret: str = ""                   # used to sign payloads with HMAC
    active: bool = True
    created_at: float = field(default_factory=time.time)
    # Stats
    deliveries: int = 0
    failures: int = 0
    last_attempt_ts: float = 0.0
    last_status: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


class EventBus:
    """Central pub/sub. Every subsystem that emits events calls emit()."""

    def __init__(self, max_recent: int = 500):
        self._subs: List[Callable[[Event], None]] = []
        self._recent: deque = deque(maxlen=max_recent)
        self._lock = threading.Lock()

    def emit(self, event_type: str, payload: Optional[dict] = None) -> Event:
        ev = Event(
            id=str(uuid.uuid4()),
            type=event_type,
            ts=time.time(),
            payload=payload or {},
        )
        with self._lock:
            self._recent.append(ev)
            subs = list(self._subs)
        for fn in subs:
            try:
                fn(ev)
            except Exception as e:
                print(f"[EVENT] subscriber error on {event_type}: {e}")
        return ev

    def subscribe(self, fn: Callable[[Event], None]) -> Callable[[], None]:
        """Register a callback. Returns an unsubscribe function."""
        with self._lock:
            self._subs.append(fn)
        def _unsub():
            with self._lock:
                try:
                    self._subs.remove(fn)
                except ValueError:
                    pass
        return _unsub

    def recent(self, limit: int = 50, event_type: Optional[str] = None) -> List[dict]:
        with self._lock:
            items = list(self._recent)
        if event_type:
            items = [e for e in items if e.type == event_type]
        return [e.to_dict() for e in items[-limit:]]


class WebhookManager:
    """Persistent webhook registry + delivery with retry."""

    def __init__(self, bus: EventBus):
        self.bus = bus
        self._webhooks: Dict[str, Webhook] = {}
        self._lock = threading.Lock()
        self._load()
        # Subscribe to ALL events to dispatch webhooks
        self.bus.subscribe(self._dispatch)

    def _load(self):
        path = _webhooks_path()
        if not os.path.exists(path):
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for w_data in data.get("webhooks", []):
                known = {k: v for k, v in w_data.items() if k in Webhook.__dataclass_fields__}
                w = Webhook(**known)
                self._webhooks[w.webhook_id] = w
        except Exception as e:
            print(f"[WEBHOOK] Failed to load registry: {e}")

    def _save(self):
        path = _webhooks_path()
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"webhooks": [w.to_dict() for w in self._webhooks.values()]}, f, indent=2)
        if os.path.exists(path):
            os.replace(tmp, path)
        else:
            os.rename(tmp, path)

    def register(self, url: str, event_types: Optional[List[str]] = None,
                 secret: str = "") -> Webhook:
        import secrets as _secrets
        wh = Webhook(
            webhook_id=str(uuid.uuid4()),
            url=url,
            event_types=event_types or ["*"],
            secret=secret or _secrets.token_urlsafe(16),
        )
        with self._lock:
            self._webhooks[wh.webhook_id] = wh
            self._save()
        return wh

    def unregister(self, webhook_id: str) -> bool:
        with self._lock:
            if webhook_id in self._webhooks:
                del self._webhooks[webhook_id]
                self._save()
                return True
        return False

    def list_webhooks(self) -> List[Webhook]:
        with self._lock:
            return list(self._webhooks.values())

    def _dispatch(self, event: Event):
        """Fan out to every webhook that subscribed to this event's type."""
        matches = []
        with self._lock:
            for w in self._webhooks.values():
                if not w.active:
                    continue
                if "*" in w.event_types or event.type in w.event_types:
                    matches.append(w)
        for w in matches:
            threading.Thread(
                target=self._deliver, args=(w, event), daemon=True,
            ).start()

    def _deliver(self, webhook: Webhook, event: Event):
        """POST the event. Retries twice with backoff on failure."""
        import hashlib, hmac, urllib.request, urllib.error
        payload = json.dumps({
            "id": event.id, "type": event.type, "ts": event.ts,
            "payload": event.payload,
        }).encode("utf-8")
        # HMAC signature so the receiver can verify authenticity
        sig = hmac.new(
            webhook.secret.encode("utf-8"), payload, hashlib.sha256
        ).hexdigest()
        headers = {
            "Content-Type": "application/json",
            "X-NL-Event-Type": event.type,
            "X-NL-Event-Id": event.id,
            "X-NL-Signature": f"sha256={sig}",
            "User-Agent": "NeuroLinked/1.3 Webhooks",
        }
        backoff = [0.0, 0.5, 2.0]
        last_status = 0
        for attempt, delay in enumerate(backoff):
            if delay:
                time.sleep(delay)
            try:
                req = urllib.request.Request(webhook.url, data=payload,
                                             headers=headers, method="POST")
                with urllib.request.urlopen(req, timeout=10) as resp:
                    last_status = resp.status
                    if 200 <= resp.status < 300:
                        break
            except urllib.error.HTTPError as e:
                last_status = e.code
                if 400 <= e.code < 500:
                    break  # client error, won't get better with retry
            except Exception:
                last_status = -1
        with self._lock:
            webhook.deliveries += 1
            webhook.last_attempt_ts = time.time()
            webhook.last_status = last_status
            if not (200 <= last_status < 300):
                webhook.failures += 1
            self._save()


# Module singletons
event_bus = EventBus()
webhook_manager = WebhookManager(event_bus)
