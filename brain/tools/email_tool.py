"""
Email tool — send real emails via SMTP.

Config lives in brain_state/tool_config.json:
    {
      "smtp": {
        "host": "smtp.gmail.com",
        "port": 587,
        "username": "...",
        "password": "...",
        "from_addr": "..."
      }
    }
"""

from __future__ import annotations

import json
import os
import smtplib
import sys
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Optional

from brain.tools import register_tool


def _app_root() -> str:
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _load_smtp_config() -> Optional[dict]:
    path = os.path.join(_app_root(), "brain_state", "tool_config.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f).get("smtp")
    except Exception:
        return None


@register_tool(
    name="email.send",
    description="Send an email via SMTP. Requires SMTP config in tool_config.json.",
    params={
        "to": "str — recipient email",
        "subject": "str — email subject line",
        "body": "str — email body (plain text or HTML)",
        "html": "bool — treat body as HTML (default: false)",
        "cc": "str — optional CC recipients, comma-separated",
        "bcc": "str — optional BCC recipients, comma-separated",
    },
    required=["to", "subject", "body"],
    category="communication",
    dangerous=True,  # Sends real email
)
def email_send(
    to: str,
    subject: str,
    body: str,
    html: bool = False,
    cc: str = "",
    bcc: str = "",
) -> dict:
    cfg = _load_smtp_config()
    if cfg is None:
        return {
            "ok": False,
            "error": "SMTP not configured. Add smtp section to brain_state/tool_config.json",
            "hint": "See docs/TOOL_CONFIG.md for the schema.",
        }

    required_keys = ["host", "port", "username", "password", "from_addr"]
    missing = [k for k in required_keys if k not in cfg]
    if missing:
        return {"ok": False, "error": f"SMTP config missing keys: {missing}"}

    msg = MIMEMultipart("alternative")
    msg["From"] = cfg["from_addr"]
    msg["To"] = to
    msg["Subject"] = subject
    if cc:
        msg["Cc"] = cc
    mimetype = "html" if html else "plain"
    msg.attach(MIMEText(body, mimetype, "utf-8"))

    recipients = [x.strip() for x in to.split(",") if x.strip()]
    if cc:
        recipients += [x.strip() for x in cc.split(",") if x.strip()]
    if bcc:
        recipients += [x.strip() for x in bcc.split(",") if x.strip()]

    try:
        with smtplib.SMTP(cfg["host"], int(cfg["port"]), timeout=30) as server:
            server.starttls()
            server.login(cfg["username"], cfg["password"])
            server.sendmail(cfg["from_addr"], recipients, msg.as_string())
        return {
            "ok": True,
            "to": recipients,
            "subject": subject,
            "from": cfg["from_addr"],
        }
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}


@register_tool(
    name="email.draft",
    description="Draft an email (does not send). Stores in brain_state/drafts/ for review.",
    params={
        "to": "str — recipient email",
        "subject": "str — subject line",
        "body": "str — email body",
    },
    required=["to", "subject", "body"],
    category="communication",
)
def email_draft(to: str, subject: str, body: str) -> dict:
    import time
    drafts_dir = os.path.join(_app_root(), "brain_state", "drafts")
    os.makedirs(drafts_dir, exist_ok=True)
    fname = f"draft_{int(time.time())}.json"
    path = os.path.join(drafts_dir, fname)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            {"to": to, "subject": subject, "body": body, "created_at": time.time()},
            f, indent=2,
        )
    return {"ok": True, "draft_path": path, "draft_id": fname}
