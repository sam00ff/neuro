"""
Multi-tenant brain management — V1.3

Problem V1.2 had: one install = one brain = one user. No good for a
10-person marketing agency. They'd have to spin up 10 separate installs
and their team knowledge lived in 10 silos.

Solution in V1.3: one install runs N isolated brains, PLUS shared
"channels" that teams can subscribe to. Marketing team shares campaign
knowledge; sales team shares customer knowledge; everyone sees
"company-wide" knowledge if they're subscribed.

Architecture:
  brain_state/
    tenants/
      <tenant_id>/
        knowledge.db         - isolated semantic memory
        meta.json            - tenant metadata
        drafts/
        sheets/
        ...
    channels/
      <channel_id>.db        - shared memory bus
    registry.json            - tenant registry + keys + subscriptions

Each tenant gets a unique token. Pass it as `X-NL-Tenant-Token` header
or `?token=` param on API calls. The manager routes to the right
knowledge store and enforces roles (admin / user / readonly).
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import sys
import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

from brain.knowledge_store import KnowledgeStore


def _app_root() -> str:
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _tenants_dir() -> str:
    d = os.path.join(_app_root(), "brain_state", "tenants")
    os.makedirs(d, exist_ok=True)
    return d


def _channels_dir() -> str:
    d = os.path.join(_app_root(), "brain_state", "channels")
    os.makedirs(d, exist_ok=True)
    return d


def _registry_path() -> str:
    return os.path.join(_app_root(), "brain_state", "registry.json")


VALID_ROLES = {"admin", "user", "readonly"}


@dataclass
class Tenant:
    tenant_id: str
    name: str
    role: str = "user"
    token: str = ""
    subscribed_channels: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def public_dict(self) -> dict:
        """Safe-to-return dict (NO token)."""
        d = asdict(self)
        d.pop("token", None)
        return d

    def full_dict(self) -> dict:
        """Full dict including token — only returned on create or to admins."""
        return asdict(self)


@dataclass
class Channel:
    channel_id: str
    name: str
    description: str = ""
    created_at: float = field(default_factory=time.time)
    subscriber_count: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


class TenantManager:
    """Routes API calls to the right tenant's knowledge store."""

    def __init__(self):
        self._tenants: Dict[str, Tenant] = {}
        self._channels: Dict[str, Channel] = {}
        self._stores: Dict[str, KnowledgeStore] = {}         # tenant_id -> KnowledgeStore
        self._channel_stores: Dict[str, KnowledgeStore] = {} # channel_id -> KnowledgeStore
        self._token_index: Dict[str, str] = {}               # token -> tenant_id
        self._load_registry()

    # ---------- Registry persistence ----------

    def _load_registry(self):
        path = _registry_path()
        if not os.path.exists(path):
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for t_data in data.get("tenants", []):
                # Only known fields — ignores extras from future versions
                known = {k: v for k, v in t_data.items() if k in Tenant.__dataclass_fields__}
                t = Tenant(**known)
                self._tenants[t.tenant_id] = t
                if t.token:
                    self._token_index[t.token] = t.tenant_id
            for c_data in data.get("channels", []):
                known = {k: v for k, v in c_data.items() if k in Channel.__dataclass_fields__}
                c = Channel(**known)
                self._channels[c.channel_id] = c
        except Exception as e:
            print(f"[TENANT] Failed to load registry: {e}")

    def _save_registry(self):
        path = _registry_path()
        tmp = path + ".tmp"
        data = {
            "tenants": [t.full_dict() for t in self._tenants.values()],
            "channels": [c.to_dict() for c in self._channels.values()],
        }
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        # Atomic swap
        if os.path.exists(path):
            os.replace(tmp, path)
        else:
            os.rename(tmp, path)

    # ---------- Tenant CRUD ----------

    def create_tenant(self, name: str, role: str = "user", metadata: Optional[dict] = None) -> Tenant:
        if role not in VALID_ROLES:
            raise ValueError(f"role must be one of {VALID_ROLES}")
        tenant_id = str(uuid.uuid4())
        # Tenant API token: 32 chars, URL-safe
        token = "nlkd_" + secrets.token_urlsafe(24)
        t = Tenant(
            tenant_id=tenant_id,
            name=name,
            role=role,
            token=token,
            metadata=metadata or {},
        )
        self._tenants[tenant_id] = t
        self._token_index[token] = tenant_id
        # Ensure tenant folder + knowledge store
        self._ensure_tenant_store(tenant_id)
        self._save_registry()
        print(f"[TENANT] Created tenant {name!r} id={tenant_id[:8]}... role={role}")
        return t

    def delete_tenant(self, tenant_id: str, wipe_data: bool = False) -> bool:
        t = self._tenants.pop(tenant_id, None)
        if t is None:
            return False
        if t.token in self._token_index:
            self._token_index.pop(t.token, None)
        self._stores.pop(tenant_id, None)
        if wipe_data:
            import shutil
            path = os.path.join(_tenants_dir(), tenant_id)
            if os.path.isdir(path):
                shutil.rmtree(path, ignore_errors=True)
        self._save_registry()
        return True

    def list_tenants(self) -> List[Tenant]:
        return list(self._tenants.values())

    def get_tenant(self, tenant_id: str) -> Optional[Tenant]:
        return self._tenants.get(tenant_id)

    def rotate_token(self, tenant_id: str) -> Optional[str]:
        t = self._tenants.get(tenant_id)
        if t is None:
            return None
        if t.token in self._token_index:
            self._token_index.pop(t.token)
        t.token = "nlkd_" + secrets.token_urlsafe(24)
        self._token_index[t.token] = tenant_id
        self._save_registry()
        return t.token

    # ---------- Auth ----------

    def auth(self, token: Optional[str]) -> Optional[Tenant]:
        """Resolve token -> tenant (or None). Updates last_seen on success."""
        if not token:
            return None
        tid = self._token_index.get(token)
        if tid is None:
            return None
        t = self._tenants.get(tid)
        if t is not None:
            t.last_seen = time.time()
        return t

    def require_role(self, token: Optional[str], *allowed_roles: str) -> Tenant:
        """Raise if the token is invalid or the role is not allowed."""
        t = self.auth(token)
        if t is None:
            raise PermissionError("Invalid tenant token")
        if t.role not in allowed_roles:
            raise PermissionError(
                f"Tenant role {t.role!r} not in required roles {allowed_roles}"
            )
        return t

    # ---------- Knowledge routing ----------

    def _tenant_db_path(self, tenant_id: str) -> str:
        d = os.path.join(_tenants_dir(), tenant_id)
        os.makedirs(d, exist_ok=True)
        return os.path.join(d, "knowledge.db")

    def _channel_db_path(self, channel_id: str) -> str:
        return os.path.join(_channels_dir(), f"{channel_id}.db")

    def _ensure_tenant_store(self, tenant_id: str) -> KnowledgeStore:
        """Get or create the tenant's isolated KnowledgeStore."""
        if tenant_id in self._stores:
            return self._stores[tenant_id]
        store = KnowledgeStore(db_path=self._tenant_db_path(tenant_id))
        self._stores[tenant_id] = store
        return store

    def _ensure_channel_store(self, channel_id: str) -> KnowledgeStore:
        if channel_id in self._channel_stores:
            return self._channel_stores[channel_id]
        store = KnowledgeStore(db_path=self._channel_db_path(channel_id))
        self._channel_stores[channel_id] = store
        return store

    def get_store(self, tenant_id: str) -> Optional[KnowledgeStore]:
        if tenant_id not in self._tenants:
            return None
        return self._ensure_tenant_store(tenant_id)

    # ---------- Channels ----------

    def create_channel(self, name: str, description: str = "") -> Channel:
        channel_id = str(uuid.uuid4())
        ch = Channel(channel_id=channel_id, name=name, description=description)
        self._channels[channel_id] = ch
        self._ensure_channel_store(channel_id)
        self._save_registry()
        print(f"[TENANT] Created channel {name!r} id={channel_id[:8]}...")
        return ch

    def delete_channel(self, channel_id: str) -> bool:
        ch = self._channels.pop(channel_id, None)
        if ch is None:
            return False
        # Unsubscribe all tenants
        for t in self._tenants.values():
            if channel_id in t.subscribed_channels:
                t.subscribed_channels.remove(channel_id)
        self._channel_stores.pop(channel_id, None)
        self._save_registry()
        return True

    def list_channels(self) -> List[Channel]:
        # Update subscriber counts
        for ch in self._channels.values():
            ch.subscriber_count = sum(
                1 for t in self._tenants.values() if ch.channel_id in t.subscribed_channels
            )
        return list(self._channels.values())

    def subscribe(self, tenant_id: str, channel_id: str) -> bool:
        t = self._tenants.get(tenant_id)
        if t is None or channel_id not in self._channels:
            return False
        if channel_id not in t.subscribed_channels:
            t.subscribed_channels.append(channel_id)
            self._save_registry()
        return True

    def unsubscribe(self, tenant_id: str, channel_id: str) -> bool:
        t = self._tenants.get(tenant_id)
        if t is None:
            return False
        if channel_id in t.subscribed_channels:
            t.subscribed_channels.remove(channel_id)
            self._save_registry()
            return True
        return False

    def publish_to_channel(self, channel_id: str, text: str, source: str = "tenant",
                           tags: Optional[list] = None) -> dict:
        if channel_id not in self._channels:
            return {"ok": False, "error": "channel not found"}
        store = self._ensure_channel_store(channel_id)
        try:
            entry_id = store.store(text=text, source=source, tags=tags or [])
            return {"ok": True, "entry_id": entry_id, "channel_id": channel_id}
        except Exception as e:
            return {"ok": False, "error": f"{type(e).__name__}: {e}"}

    # ---------- Unified query (tenant + their subscribed channels) ----------

    def recall(self, tenant_id: str, query: str, limit: int = 20,
               include_channels: bool = True) -> List[dict]:
        """
        Semantic recall that merges tenant's own memory + anything in
        channels they've subscribed to. Perfect for the "every employee
        gets institutional memory on day one" pitch.
        """
        t = self._tenants.get(tenant_id)
        if t is None:
            return []
        results: List[dict] = []
        store = self._ensure_tenant_store(tenant_id)
        for r in store.semantic_search(query, limit=limit):
            r["_origin"] = "tenant"
            results.append(r)
        if include_channels:
            for ch_id in t.subscribed_channels:
                if ch_id in self._channels:
                    ch_store = self._ensure_channel_store(ch_id)
                    for r in ch_store.semantic_search(query, limit=limit):
                        r["_origin"] = f"channel:{self._channels[ch_id].name}"
                        results.append(r)
        # Re-sort combined results by similarity
        results.sort(key=lambda r: r.get("_similarity", 0), reverse=True)
        return results[:limit]

    def store(self, tenant_id: str, text: str, source: str = "tenant",
              tags: Optional[list] = None) -> dict:
        store = self.get_store(tenant_id)
        if store is None:
            return {"ok": False, "error": "unknown tenant"}
        try:
            entry_id = store.store(text=text, source=source, tags=tags or [])
            return {"ok": True, "entry_id": entry_id, "tenant_id": tenant_id}
        except Exception as e:
            return {"ok": False, "error": f"{type(e).__name__}: {e}"}

    # ---------- Stats ----------

    def stats(self) -> dict:
        return {
            "tenant_count": len(self._tenants),
            "channel_count": len(self._channels),
            "by_role": {
                role: sum(1 for t in self._tenants.values() if t.role == role)
                for role in VALID_ROLES
            },
            "total_subscriptions": sum(
                len(t.subscribed_channels) for t in self._tenants.values()
            ),
        }


# Module singleton. Import and use: from brain.tenant import tenant_manager
tenant_manager = TenantManager()
