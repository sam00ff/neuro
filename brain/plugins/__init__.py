"""
Plugin loader — V1.3

Drop-in folders in brain/plugins/ or plugins/ (user-installed) that
expose business integrations. Every plugin:
  - has a manifest.json with metadata
  - registers one or more tools via @register_tool
  - optionally defines webhooks / event subscriptions

The loader auto-discovers plugins on startup and registers their tools
into the agent framework. No code changes needed to server.py when a
new plugin is installed — just drop a folder in.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


@dataclass
class PluginInfo:
    name: str
    version: str
    description: str
    path: str
    manifest: Dict[str, Any] = field(default_factory=dict)
    loaded: bool = False
    error: Optional[str] = None
    tools_added: List[str] = field(default_factory=list)
    loaded_at: float = field(default_factory=time.time)


def _app_root() -> str:
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _plugin_search_dirs() -> List[str]:
    """Places we look for plugins."""
    return [
        os.path.dirname(os.path.abspath(__file__)),           # brain/plugins/
        os.path.join(_app_root(), "plugins"),                  # user-dropped
        os.path.join(_app_root(), "brain_state", "plugins"),   # tenant-installed
    ]


_REGISTERED: Dict[str, PluginInfo] = {}


def list_plugins() -> List[PluginInfo]:
    return list(_REGISTERED.values())


def discover_and_load() -> List[PluginInfo]:
    """Scan every search dir for valid plugin folders and import them.
    Plugins with the same name found in multiple dirs are loaded once
    (first found wins — user-installed dirs override builtins)."""
    found: List[PluginInfo] = []
    seen_names: set = set()
    # Reverse order so user-installed dirs (later in list) get precedence
    for base in reversed(_plugin_search_dirs()):
        if not os.path.isdir(base):
            continue
        for entry in sorted(os.listdir(base)):
            entry_path = os.path.join(base, entry)
            if not os.path.isdir(entry_path):
                continue
            manifest_path = os.path.join(entry_path, "manifest.json")
            if not os.path.exists(manifest_path):
                continue
            # Quick name check to dedupe
            try:
                with open(manifest_path, "r", encoding="utf-8") as f:
                    candidate_name = json.load(f).get("name") or entry
            except Exception:
                candidate_name = entry
            if candidate_name in seen_names:
                continue
            seen_names.add(candidate_name)
            info = _load_plugin(entry_path, manifest_path)
            if info is not None:
                found.append(info)
                _REGISTERED[info.name] = info
    return found


def _load_plugin(folder: str, manifest_path: str) -> Optional[PluginInfo]:
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        name = manifest.get("name") or os.path.basename(folder)
        version = manifest.get("version", "0.0.1")
        description = manifest.get("description", "")
        entry = manifest.get("entry", "plugin.py")

        info = PluginInfo(
            name=name, version=version, description=description,
            path=folder, manifest=manifest,
        )

        # Snapshot registered tools before import
        from brain.tools import list_tools
        before = {t.name for t in list_tools()}

        entry_path = os.path.join(folder, entry)
        if not os.path.exists(entry_path):
            info.error = f"entry file not found: {entry}"
            info.loaded = False
            return info

        spec = importlib.util.spec_from_file_location(
            f"nlkd_plugin_{name.replace('-', '_')}", entry_path,
        )
        if spec is None or spec.loader is None:
            info.error = "could not build import spec"
            return info
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Snapshot after — anything added is from this plugin
        after = {t.name for t in list_tools()}
        info.tools_added = sorted(after - before)
        info.loaded = True
        print(f"[PLUGIN] Loaded {name} v{version} — added tools: {info.tools_added}")
        return info
    except Exception as e:
        print(f"[PLUGIN] Failed to load {folder}: {e}")
        return PluginInfo(
            name=os.path.basename(folder), version="?",
            description="", path=folder, loaded=False,
            error=f"{type(e).__name__}: {e}",
        )
