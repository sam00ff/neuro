"""
Shopify plugin — wraps Shopify Admin API 2024-07.

Use case: connect your Shopify store so the brain can answer things
like "what did we sell most this week?" or "who ordered but never
came back?" — using your live order, customer, and product data.
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


def _cfg() -> Optional[Dict[str, str]]:
    path = os.path.join(_app_root(), "brain_state", "tool_config.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f).get("shopify")
    except Exception:
        return None


def _shopify_get(path: str, params: Optional[dict] = None) -> dict:
    cfg = _cfg()
    if not cfg or "store" not in cfg or "access_token" not in cfg:
        return {"ok": False, "error": "Shopify not configured (need store + access_token)"}
    url = f"https://{cfg['store']}/admin/api/2024-07/{path}"
    if params:
        url += "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={
        "X-Shopify-Access-Token": cfg["access_token"],
        "Accept": "application/json",
    })
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return {"ok": True, "data": json.loads(r.read().decode("utf-8"))}
    except urllib.error.HTTPError as e:
        return {"ok": False, "error": f"HTTP {e.code}: {e.reason}"}
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}


@register_tool(
    name="shopify.list_orders",
    description="List recent Shopify orders.",
    params={
        "status": "str — 'any' | 'open' | 'closed' | 'cancelled' (default: 'any')",
        "limit": "int — max 250 (default: 50)",
        "since": "str — ISO 8601 datetime filter (e.g. 2026-04-01T00:00:00Z)",
    },
    required=[],
    category="commerce",
)
def shopify_list_orders(status: str = "any", limit: int = 50, since: str = "") -> dict:
    params: Dict[str, Any] = {"status": status, "limit": min(limit, 250)}
    if since:
        params["created_at_min"] = since
    return _shopify_get("orders.json", params)


@register_tool(
    name="shopify.list_products",
    description="List Shopify products.",
    params={"limit": "int — default 50, max 250"},
    required=[],
    category="commerce",
)
def shopify_list_products(limit: int = 50) -> dict:
    return _shopify_get("products.json", {"limit": min(limit, 250)})


@register_tool(
    name="shopify.list_customers",
    description="List Shopify customers.",
    params={
        "limit": "int — default 50",
        "query": "str — optional search query (name, email)",
    },
    required=[],
    category="commerce",
)
def shopify_list_customers(limit: int = 50, query: str = "") -> dict:
    params: Dict[str, Any] = {"limit": min(limit, 250)}
    if query:
        params["query"] = query
    return _shopify_get("customers/search.json" if query else "customers.json", params)


@register_tool(
    name="shopify.get_order",
    description="Get details on a specific Shopify order.",
    params={"order_id": "str — Shopify order ID"},
    required=["order_id"],
    category="commerce",
)
def shopify_get_order(order_id: str) -> dict:
    return _shopify_get(f"orders/{order_id}.json")


@register_tool(
    name="shopify.shop_info",
    description="Get store configuration info.",
    params={},
    required=[],
    category="commerce",
)
def shopify_shop_info() -> dict:
    return _shopify_get("shop.json")
