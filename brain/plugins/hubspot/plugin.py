"""
HubSpot plugin — connect your CRM to the brain.

Exposes: list/search/get contacts, deals, companies. Creates + updates
supported for contacts specifically (expand as needed).
"""

import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, Optional

from brain.tools import register_tool


def _app_root() -> str:
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def _token() -> Optional[str]:
    path = os.path.join(_app_root(), "brain_state", "tool_config.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f).get("hubspot", {}).get("access_token")
    except Exception:
        return None


BASE = "https://api.hubapi.com"


def _request(method: str, path: str, body: Any = None, params: Optional[dict] = None) -> dict:
    token = _token()
    if not token:
        return {"ok": False, "error": "HubSpot not configured — add access_token to tool_config.json"}
    url = f"{BASE}{path}"
    if params:
        url += "?" + urllib.parse.urlencode(params)
    data = json.dumps(body).encode("utf-8") if body is not None else None
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
    }
    if body is not None:
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            raw = r.read().decode("utf-8")
            return {"ok": True, "data": json.loads(raw) if raw else {}}
    except urllib.error.HTTPError as e:
        body_text = ""
        try: body_text = e.read().decode("utf-8", errors="replace")
        except Exception: pass
        return {"ok": False, "error": f"HTTP {e.code}: {e.reason}", "body": body_text}
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}


# ---------- Contacts ----------

@register_tool(
    name="hubspot.list_contacts",
    description="List HubSpot contacts (most recent first).",
    params={"limit": "int — default 100, max 100"},
    required=[],
    category="crm",
)
def list_contacts(limit: int = 100) -> dict:
    return _request("GET", "/crm/v3/objects/contacts", params={"limit": min(limit, 100)})


@register_tool(
    name="hubspot.search_contacts",
    description="Search HubSpot contacts by email, name, or any filter.",
    params={
        "filters": "list[dict] — HubSpot filter objects [{propertyName, operator, value}]",
        "limit": "int — default 50",
    },
    required=["filters"],
    category="crm",
)
def search_contacts(filters: list, limit: int = 50) -> dict:
    body = {
        "filterGroups": [{"filters": filters}],
        "limit": min(limit, 100),
    }
    return _request("POST", "/crm/v3/objects/contacts/search", body=body)


@register_tool(
    name="hubspot.create_contact",
    description="Create a new HubSpot contact.",
    params={
        "email": "str — required",
        "firstname": "str",
        "lastname": "str",
        "company": "str",
        "phone": "str",
    },
    required=["email"],
    category="crm",
    dangerous=True,
)
def create_contact(email: str, firstname: str = "", lastname: str = "",
                   company: str = "", phone: str = "") -> dict:
    props: Dict[str, str] = {"email": email}
    if firstname: props["firstname"] = firstname
    if lastname: props["lastname"] = lastname
    if company: props["company"] = company
    if phone: props["phone"] = phone
    return _request("POST", "/crm/v3/objects/contacts", body={"properties": props})


# ---------- Deals ----------

@register_tool(
    name="hubspot.list_deals",
    description="List HubSpot deals (pipeline entries).",
    params={"limit": "int — default 100"},
    required=[],
    category="crm",
)
def list_deals(limit: int = 100) -> dict:
    return _request("GET", "/crm/v3/objects/deals", params={"limit": min(limit, 100)})


# ---------- Companies ----------

@register_tool(
    name="hubspot.list_companies",
    description="List HubSpot companies.",
    params={"limit": "int — default 100"},
    required=[],
    category="crm",
)
def list_companies(limit: int = 100) -> dict:
    return _request("GET", "/crm/v3/objects/companies", params={"limit": min(limit, 100)})
