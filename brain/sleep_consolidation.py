"""
Sleep Consolidation — V1.3

This is the thing Obsidian CANNOT do. While the user is idle, the brain
replays recent memories, finds cross-domain connections, and proactively
surfaces insights it thinks you'd want to know.

Runs on its own thread. Fires when:
  - No user activity for `idle_threshold_sec` (default 300s / 5min)
  - Hourly tick regardless (catches long-running idle)

Each pass produces:
  - Replayed memories (by recency + strength)
  - Cross-references (pairs of memories that share concept tokens)
  - Auto-generated insights (summaries of patterns)
  - Scheduled reminders (things the brain decides you care about)

Insights land in `brain_state/insights.json` (append-only log) so the
user can see them later: "oh look, it noticed that 3 of my clients
mentioned price in the same week."
"""

from __future__ import annotations

import json
import os
import re
import sys
import threading
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


def _app_root() -> str:
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _insights_path() -> str:
    d = os.path.join(_app_root(), "brain_state")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, "insights.jsonl")


# Simple stopword set for cross-reference extraction
_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "on", "in", "at", "of", "for",
    "to", "with", "from", "by", "as", "is", "it", "this", "that", "was", "were",
    "be", "been", "has", "have", "had", "will", "would", "can", "could", "our",
    "their", "they", "we", "i", "my", "your", "you", "me", "us", "not",
    "so", "too", "up", "out", "about", "over", "than",
}

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'-]{2,}")


def _keywords(text: str, min_len: int = 4) -> List[str]:
    """Extract non-stopword keywords longer than `min_len` chars."""
    if not text:
        return []
    words = _WORD_RE.findall(text.lower())
    return [w for w in words if len(w) >= min_len and w not in _STOPWORDS]


@dataclass
class Insight:
    ts: float
    kind: str                         # "cross-reference" | "pattern" | "reminder"
    title: str
    body: str
    supporting_entry_ids: List[int] = field(default_factory=list)
    score: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ConsolidationPass:
    started_at: float
    replayed_count: int = 0
    insights_generated: int = 0
    crossrefs_found: int = 0
    duration_sec: float = 0.0
    new_insights: List[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


class SleepConsolidator:
    """
    Runs background consolidation passes on the knowledge store.

    Typical use:
        sc = SleepConsolidator(knowledge_store)
        sc.start()   # spawns background thread
        ...
        sc.mark_activity()  # call this every time user does anything
    """

    def __init__(
        self,
        knowledge_store,
        idle_threshold_sec: float = 300.0,
        hourly_tick: bool = True,
        min_recent_entries: int = 5,
    ):
        self.ks = knowledge_store
        self.idle_threshold = idle_threshold_sec
        self.hourly_tick = hourly_tick
        self.min_recent_entries = min_recent_entries

        self._last_activity = time.time()
        self._last_pass = 0.0
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._history: List[ConsolidationPass] = []
        self._insights: List[Insight] = []

    # ---------- Lifecycle ----------

    def start(self):
        with self._lock:
            if self._running:
                return
            self._running = True
        self._thread = threading.Thread(
            target=self._loop, name="nlkd-sleep-consolidator", daemon=True,
        )
        self._thread.start()
        print("[SLEEP] Consolidator started")

    def stop(self):
        with self._lock:
            self._running = False
        if self._thread:
            self._thread.join(timeout=3)

    def mark_activity(self):
        """Call whenever the user interacts. Resets the idle timer."""
        self._last_activity = time.time()

    def _loop(self):
        while True:
            with self._lock:
                if not self._running:
                    break
            time.sleep(10)  # wake up every 10 sec to check conditions
            now = time.time()
            idle_for = now - self._last_activity
            since_last = now - self._last_pass
            should_run = (
                idle_for >= self.idle_threshold or
                (self.hourly_tick and since_last >= 3600)
            )
            if should_run:
                try:
                    self.consolidate()
                    self._last_pass = time.time()
                except Exception as e:
                    print(f"[SLEEP] consolidation error: {e}")

    # ---------- One pass ----------

    def consolidate(self) -> ConsolidationPass:
        """Run a single consolidation pass. Returns the pass summary."""
        t0 = time.time()
        pass_info = ConsolidationPass(started_at=t0)

        recent = self.ks.recent(limit=200)
        if len(recent) < self.min_recent_entries:
            pass_info.duration_sec = round(time.time() - t0, 3)
            self._history.append(pass_info)
            return pass_info

        pass_info.replayed_count = len(recent)

        # --- Cross-reference detection ---
        # For each pair of entries, count shared keywords. High overlap
        # + semantic distance = interesting cross-reference.
        entry_keywords: Dict[int, set] = {}
        for e in recent:
            eid = e.get("id") or e.get("entry_id")
            if eid is None:
                continue
            entry_keywords[eid] = set(_keywords(e.get("text", "")))

        crossrefs: List[Insight] = []
        checked = 0
        for i_idx, (id_a, kws_a) in enumerate(entry_keywords.items()):
            if len(kws_a) < 3:
                continue
            for id_b, kws_b in list(entry_keywords.items())[i_idx+1:]:
                if len(kws_b) < 3:
                    continue
                shared = kws_a & kws_b
                if len(shared) >= 3:
                    # Flag as interesting
                    ins = Insight(
                        ts=time.time(),
                        kind="cross-reference",
                        title=f"Connected: {', '.join(sorted(shared)[:5])}",
                        body=(
                            f"Entries #{id_a} and #{id_b} share concepts: "
                            f"{', '.join(sorted(shared))}"
                        ),
                        supporting_entry_ids=[id_a, id_b],
                        score=float(len(shared)),
                    )
                    crossrefs.append(ins)
                    checked += 1
                    if checked >= 20:
                        break
            if checked >= 20:
                break

        pass_info.crossrefs_found = len(crossrefs)

        # --- Pattern detection: most common concepts in recent memory ---
        all_kw: Counter = Counter()
        for kws in entry_keywords.values():
            all_kw.update(kws)
        top_patterns = [(w, c) for w, c in all_kw.most_common(5) if c >= 3]
        patterns: List[Insight] = []
        for word, count in top_patterns:
            patterns.append(Insight(
                ts=time.time(),
                kind="pattern",
                title=f"Recurring theme: {word!r}",
                body=f"'{word}' appeared across {count} recent memories. "
                     f"Worth a closer look?",
                score=float(count),
            ))

        all_new = crossrefs + patterns
        pass_info.insights_generated = len(all_new)
        pass_info.new_insights = [i.to_dict() for i in all_new]
        pass_info.duration_sec = round(time.time() - t0, 3)

        # Persist insights (append to JSONL)
        if all_new:
            path = _insights_path()
            with open(path, "a", encoding="utf-8") as f:
                for ins in all_new:
                    f.write(json.dumps(ins.to_dict()) + "\n")
            self._insights.extend(all_new)

        self._history.append(pass_info)
        if len(self._history) > 100:
            self._history = self._history[-100:]

        print(f"[SLEEP] Pass complete: {len(all_new)} new insights "
              f"({len(crossrefs)} cross-refs, {len(patterns)} patterns) "
              f"in {pass_info.duration_sec}s")
        return pass_info

    # ---------- Query ----------

    def recent_insights(self, limit: int = 20) -> List[dict]:
        """Return recent insights, preferring in-memory then disk."""
        if self._insights:
            return [i.to_dict() for i in self._insights[-limit:]]
        # Fall back to disk
        path = _insights_path()
        if not os.path.exists(path):
            return []
        with open(path, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]
        return [json.loads(l) for l in lines[-limit:]]

    def history(self, limit: int = 20) -> List[dict]:
        return [h.to_dict() for h in self._history[-limit:]]

    def status(self) -> dict:
        return {
            "running": self._running,
            "last_activity_ts": self._last_activity,
            "last_pass_ts": self._last_pass,
            "idle_threshold_sec": self.idle_threshold,
            "total_passes": len(self._history),
            "total_insights": len(self._insights),
        }
