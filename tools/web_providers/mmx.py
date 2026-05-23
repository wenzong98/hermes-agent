"""MiniMax mmx CLI web search provider.

Uses the ``mmx`` CLI (npm install -g mmx-cli) for search.
Requires a MiniMax Token Plan API key configured via ``mmx auth login``.

Configuration::

    # Authenticate first (one-time):
    mmx auth login --api-key sk-cp-...

    # Use mmx for search in ~/.hermes/config.yaml:
    web:
      search_backend: "mmx"
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
from typing import Any, Dict

from tools.web_providers.base import WebSearchProvider

logger = logging.getLogger(__name__)


class MMXSearchProvider(WebSearchProvider):
    """Search via the ``mmx`` CLI.

    Requires ``mmx`` to be installed (``npm install -g mmx-cli``) and
    authenticated (``mmx auth login``).
    """

    def provider_name(self) -> str:
        return "mmx"

    def is_configured(self) -> bool:
        """Return True when ``mmx`` is on PATH and authenticated."""
        if not shutil.which("mmx"):
            return False
        # Quick auth check — mmx auth status exits 0 when keyed
        result = subprocess.run(
            ["mmx", "auth", "status"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.returncode == 0 and '"method": "api-key"' in result.stdout

    def search(self, query: str, limit: int = 5) -> Dict[str, Any]:
        """Execute a search via ``mmx search query``.

        Returns normalized results::

            {
                "success": True,
                "data": {
                    "web": [
                        {"title": str, "url": str, "description": str, "position": int},
                        ...
                    ]
                }
            }

        On failure returns ``{"success": False, "error": str}``.
        """
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
