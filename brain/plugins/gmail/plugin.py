"""
Gmail plugin — wraps the Gmail REST API so the agent can:
  - List recent messages
  - Search messages
  - Read a message body
  - Create a draft

Requires a Gmail OAuth access token in brain_state/tool_config.json.
Users get the token via the standard Google OAuth flow (instructions
in INTEGRATION_GMAIL.md).
"""

import base64
import json
import os
import sys
from typing import Optional
from email.mime.text import MIMEText

from brain.tools import register_tool


def _app_root() -> str:
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def _gmail_token() -> Optional[str]:
    path = os.path.join(_app_root(), "brain_state", "tool_config.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        return cfg.get("gmail", {}).get("access_token")
    except Exception:
        return None


def _gmail_get(path: str, params: Optional[dict] = None) -> dict:
    import urllib.parse, urllib.request, urllib.error
    token = _gmail_token()
    if not token:
        return {"ok": False, "error": "Gmail token not configured"}
    url = f"https://gmail.googleapis.com/gmail/v1/users/me/{path}"
    if params:
        url += "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return {"ok": True, "data": json.loads(r.read().decode("utf-8"))}
    except urllib.error.HTTPError as e:
        return {"ok": False, "error": f"HTTP {e.code}: {e.reason}"}
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}


def _gmail_post(path: str, body: dict) -> dict:
    import urllib.request, urllib.error
    token = _gmail_token()
    if not token:
        return {"ok": False, "error": "Gmail token not configured"}
    url = f"https://gmail.googleapis.com/gmail/v1/users/me/{path}"
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, method="POST",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return {"ok": True, "data": json.loads(r.read().decode("utf-8"))}
    except urllib.error.HTTPError as e:
        return {"ok": False, "error": f"HTTP {e.code}: {e.reason}"}
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}


@register_tool(
    name="gmail.list_inbox",
    description="List recent Gmail inbox messages.",
    params={"max_results": "int — default 20"},
    required=[],
    category="communication",
)
def gmail_list_inbox(max_results: int = 20) -> dict:
    return _gmail_get("messages", {"maxResults": max_results, "labelIds": "INBOX"})


@register_tool(
    name="gmail.search",
    description="Search Gmail using Gmail search syntax (e.g. 'from:someone@example.com unread').",
    params={
        "query": "str — Gmail search string",
        "max_results": "int — default 20",
    },
    required=["query"],
    category="communication",
)
def gmail_search(query: str, max_results: int = 20) -> dict:
    return _gmail_get("messages", {"q": query, "maxResults": max_results})


@register_tool(
    name="gmail.read_message",
    description="Read a Gmail message by ID (returns headers + body).",
    params={"message_id": "str — Gmail message ID"},
    required=["message_id"],
    category="communication",
)
def gmail_read_message(message_id: str) -> dict:
    return _gmail_get(f"messages/{message_id}", {"format": "full"})


@register_tool(
    name="gmail.create_draft",
    description="Create a Gmail draft (does NOT send — stays in drafts folder).",
    params={
        "to": "str — recipient email",
        "subject": "str — subject",
        "body": "str — plain-text body",
    },
    required=["to", "subject", "body"],
    category="communication",
)
def gmail_create_draft(to: str, subject: str, body: str) -> dict:
    msg = MIMEText(body)
    msg["to"] = to
    msg["subject"] = subject
    raw = base64.urlsafe_b64encode(msg.as_bytes()).decode()
    return _gmail_post("drafts", {"message": {"raw": raw}})
