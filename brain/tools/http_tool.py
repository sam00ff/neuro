"""
HTTP tool — call any REST API.

Stdlib only (urllib) so no extra deps. Supports GET / POST / PUT / PATCH / DELETE.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, Optional

from brain.tools import register_tool


def _do_request(
    method: str,
    url: str,
    headers: Optional[Dict[str, str]] = None,
    params: Optional[Dict[str, Any]] = None,
    body: Any = None,
    timeout: int = 30,
) -> dict:
    if params:
        sep = "&" if "?" in url else "?"
        url = f"{url}{sep}{urllib.parse.urlencode(params)}"

    data = None
    hdrs = dict(headers or {})
    if body is not None:
        if isinstance(body, (dict, list)):
            data = json.dumps(body).encode("utf-8")
            hdrs.setdefault("Content-Type", "application/json")
        elif isinstance(body, str):
            data = body.encode("utf-8")
        else:
            data = bytes(body)

    req = urllib.request.Request(url, data=data, headers=hdrs, method=method.upper())
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            ctype = resp.headers.get("Content-Type", "")
            out: Dict[str, Any] = {
                "ok": True,
                "status": resp.status,
                "headers": dict(resp.headers),
                "url": resp.geturl(),
            }
            if "application/json" in ctype:
                try:
                    out["body"] = json.loads(raw.decode("utf-8"))
                except Exception:
                    out["body"] = raw.decode("utf-8", errors="replace")
            else:
                try:
                    out["body"] = raw.decode("utf-8", errors="replace")
                except Exception:
                    out["body"] = f"<binary {len(raw)} bytes>"
            return out
    except urllib.error.HTTPError as e:
        body_text = ""
        try:
            body_text = e.read().decode("utf-8", errors="replace")
        except Exception:
            pass
        return {
            "ok": False,
            "status": e.code,
            "error": f"HTTP {e.code}: {e.reason}",
            "body": body_text,
        }
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}


@register_tool(
    name="http.get",
    description="Send an HTTP GET request to any URL.",
    params={
        "url": "str — full URL",
        "headers": "dict — optional request headers",
        "params": "dict — optional query string parameters",
        "timeout": "int — seconds (default: 30)",
    },
    required=["url"],
    category="network",
)
def http_get(
    url: str,
    headers: Optional[Dict[str, str]] = None,
    params: Optional[Dict[str, Any]] = None,
    timeout: int = 30,
) -> dict:
    return _do_request("GET", url, headers=headers, params=params, timeout=timeout)


@register_tool(
    name="http.post",
    description="Send an HTTP POST request with a JSON or form body.",
    params={
        "url": "str — full URL",
        "body": "any — request body (dict becomes JSON, str sent raw)",
        "headers": "dict — optional headers",
        "timeout": "int — seconds (default: 30)",
    },
    required=["url"],
    category="network",
    dangerous=True,
)
def http_post(
    url: str,
    body: Any = None,
    headers: Optional[Dict[str, str]] = None,
    timeout: int = 30,
) -> dict:
    return _do_request("POST", url, headers=headers, body=body, timeout=timeout)


@register_tool(
    name="http.put",
    description="Send an HTTP PUT request.",
    params={
        "url": "str",
        "body": "any",
        "headers": "dict",
        "timeout": "int",
    },
    required=["url"],
    category="network",
    dangerous=True,
)
def http_put(
    url: str,
    body: Any = None,
    headers: Optional[Dict[str, str]] = None,
    timeout: int = 30,
) -> dict:
    return _do_request("PUT", url, headers=headers, body=body, timeout=timeout)


@register_tool(
    name="http.delete",
    description="Send an HTTP DELETE request.",
    params={
        "url": "str",
        "headers": "dict",
        "timeout": "int",
    },
    required=["url"],
    category="network",
    dangerous=True,
)
def http_delete(
    url: str,
    headers: Optional[Dict[str, str]] = None,
    timeout: int = 30,
) -> dict:
    return _do_request("DELETE", url, headers=headers, timeout=timeout)
