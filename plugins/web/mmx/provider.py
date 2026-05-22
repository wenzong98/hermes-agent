"""MiniMax mmx CLI web search provider — plugin form.

Subclasses :class:`agent.web_search_provider.WebSearchProvider`.

Requires ``mmx`` CLI (npm install -g mmx-cli) and authentication
(``mmx auth login --api-key sk-cp-...``).

Config keys::

    web:
      search_backend: "mmx"     # or fallback via backend: "mmx"

This provider only supports search (no extract / crawl).
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
from typing import Any, Dict

from agent.web_search_provider import WebSearchProvider

logger = logging.getLogger(__name__)


class MMXWebSearchProvider(WebSearchProvider):
    """Search via the ``mmx`` CLI."""

    @property
    def name(self) -> str:
        return "mmx"

    @property
    def display_name(self) -> str:
        return "MiniMax Search (mmx)"

    def is_available(self) -> bool:
        """Return True when ``mmx`` is on PATH and authenticated."""
        if not shutil.which("mmx"):
            return False
        try:
            result = subprocess.run(
                ["mmx", "auth", "status"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            return result.returncode == 0 and '"method": "api-key"' in result.stdout
        except Exception:
            return False

    def supports_search(self) -> bool:
        return True

    def supports_extract(self) -> bool:
        return False

    def supports_crawl(self) -> bool:
        return False

    def search(self, query: str, limit: int = 5) -> Dict[str, Any]:
        """Execute a search via ``mmx search query``."""
        cmd = ["mmx", "search", "query", "--q", query, "--output", "json"]
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30,
                env={**os.environ, "PATH": os.environ.get("PATH", "")},
            )
        except subprocess.TimeoutExpired:
            logger.warning("mmx search timed out for query: %s", query)
            return {"success": False, "error": "mmx search timed out"}
        except Exception as exc:
            logger.warning("mmx search failed: %s", exc)
            return {"success": False, "error": f"mmx invocation failed: {exc}"}

        if proc.returncode != 0:
            logger.warning("mmx search non-zero exit %d: %s", proc.returncode, proc.stderr)
            return {"success": False, "error": f"mmx exited with code {proc.returncode}: {proc.stderr.strip()}"}

        try:
            raw = json.loads(proc.stdout)
        except json.JSONDecodeError as exc:
            logger.warning("mmx output not valid JSON: %s", exc)
            return {"success": False, "error": f"mmx returned non-JSON: {proc.stdout[:200]}"}

        organic = raw.get("organic", [])
        results = []
        for i, item in enumerate(organic[:limit]):
            results.append({
                "title": item.get("title", ""),
                "url": item.get("link", ""),
                "description": item.get("snippet", ""),
                "position": i + 1,
            })

        logger.info("mmx search '%s': %d results", query, len(results))
        return {"success": True, "data": {"web": results}}

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": self.display_name,
            "badge": "MiniMax",
            "tag": "Requires mmx CLI and MiniMax Token Plan API key.",
            "env_vars": [],
        }
