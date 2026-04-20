"""
Calendar tool — create calendar events (.ics files + optional Google Calendar).

For simple cross-platform usage: generates .ics files that any calendar
app imports. For Google Calendar API integration, add googleapiclient creds
to brain_state/tool_config.json (future enhancement — .ics covers most cases).
"""

from __future__ import annotations

import os
import sys
import uuid
from datetime import datetime, timedelta
from typing import Optional

from brain.tools import register_tool


def _app_root() -> str:
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _ics_dir() -> str:
    d = os.path.join(_app_root(), "brain_state", "calendar")
    os.makedirs(d, exist_ok=True)
    return d


def _to_ics_dt(dt: datetime) -> str:
    # UTC format: 20260419T180000Z
    return dt.strftime("%Y%m%dT%H%M%SZ")


def _parse_iso(value: str) -> datetime:
    # Accept "2026-04-19T18:00:00" or with "Z" or offset
    v = value.replace("Z", "+00:00")
    return datetime.fromisoformat(v)


@register_tool(
    name="calendar.create_event",
    description="Create a calendar event. Writes an .ics file that any calendar app imports.",
    params={
        "title": "str — event title",
        "start": "str — ISO 8601 start datetime (e.g. 2026-04-19T18:00:00)",
        "end": "str — ISO 8601 end datetime (or duration_minutes instead)",
        "duration_minutes": "int — alternative to end",
        "description": "str — event description",
        "location": "str — event location",
        "attendees": "list[str] — email addresses",
    },
    required=["title", "start"],
    category="calendar",
)
def calendar_create_event(
    title: str,
    start: str,
    end: Optional[str] = None,
    duration_minutes: Optional[int] = None,
    description: str = "",
    location: str = "",
    attendees: Optional[list] = None,
) -> dict:
    try:
        start_dt = _parse_iso(start)
    except Exception as e:
        return {"ok": False, "error": f"Invalid start: {e}"}

    if end:
        try:
            end_dt = _parse_iso(end)
        except Exception as e:
            return {"ok": False, "error": f"Invalid end: {e}"}
    elif duration_minutes:
        end_dt = start_dt + timedelta(minutes=int(duration_minutes))
    else:
        end_dt = start_dt + timedelta(hours=1)

    uid = str(uuid.uuid4())
    now = _to_ics_dt(datetime.utcnow())
    lines = [
        "BEGIN:VCALENDAR",
        "VERSION:2.0",
        "PRODID:-//NeuroLinked//Agent//EN",
        "CALSCALE:GREGORIAN",
        "METHOD:PUBLISH",
        "BEGIN:VEVENT",
        f"UID:{uid}",
        f"DTSTAMP:{now}",
        f"DTSTART:{_to_ics_dt(start_dt)}",
        f"DTEND:{_to_ics_dt(end_dt)}",
        f"SUMMARY:{title}",
    ]
    if description:
        lines.append(f"DESCRIPTION:{description.replace(chr(10), chr(92) + 'n')}")
    if location:
        lines.append(f"LOCATION:{location}")
    for a in (attendees or []):
        lines.append(f"ATTENDEE;CN={a}:mailto:{a}")
    lines += ["END:VEVENT", "END:VCALENDAR"]

    fname = f"event_{int(datetime.utcnow().timestamp())}_{uid[:8]}.ics"
    path = os.path.join(_ics_dir(), fname)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\r\n".join(lines) + "\r\n")

    return {
        "ok": True,
        "event_id": uid,
        "ics_path": path,
        "title": title,
        "start": start_dt.isoformat() + "Z",
        "end": end_dt.isoformat() + "Z",
    }


@register_tool(
    name="calendar.list_events",
    description="List all calendar events created by the agent (from .ics files).",
    params={},
    required=[],
    category="calendar",
)
def calendar_list_events() -> dict:
    d = _ics_dir()
    events = []
    for f in sorted(os.listdir(d)):
        if not f.endswith(".ics"):
            continue
        p = os.path.join(d, f)
        try:
            with open(p, "r", encoding="utf-8") as fh:
                content = fh.read()
            ev = {"file": f, "path": p}
            for line in content.splitlines():
                if line.startswith("SUMMARY:"):
                    ev["title"] = line[len("SUMMARY:"):]
                elif line.startswith("DTSTART:"):
                    ev["start"] = line[len("DTSTART:"):]
                elif line.startswith("DTEND:"):
                    ev["end"] = line[len("DTEND:"):]
                elif line.startswith("LOCATION:"):
                    ev["location"] = line[len("LOCATION:"):]
            events.append(ev)
        except Exception:
            continue
    return {"ok": True, "events": events, "count": len(events)}
