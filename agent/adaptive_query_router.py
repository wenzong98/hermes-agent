"""Hermes host adapter for ``hermes-adaptive-router``.

This module keeps the historical ``agent.adaptive_query_router`` import surface
stable while moving the actual routing logic into the standalone package.
Hermes-specific behavior that remains here:

- load ``config.yaml`` through ``hermes_cli.config.load_config()``
- persist routing events to a local JSONL file for lightweight observability
"""

from __future__ import annotations

import json
import os
import threading
import time as _time
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, Optional

from hermes_adaptive_router import (
    AdaptiveQueryRoutingConfig,
    QueryRoute,
    build_adaptive_query_routing_prompt as _build_prompt_pkg,
    classify_query as _classify_pkg,
    load_adaptive_query_routing_config as _load_router_config,
)
from hermes_constants import get_hermes_home

_HAS_PKG = True
_persist_path = str(get_hermes_home() / "adaptive_router_history.jsonl")
_persist_lock = threading.Lock()


def set_persistence_path(p: str) -> None:
    global _persist_path
    _persist_path = p


def _write(ev: dict[str, Any]) -> None:
    """Thread-safe direct write with fsync."""
    with _persist_lock:
        p = Path(os.path.expanduser(_persist_path))
        p.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(p, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(ev, ensure_ascii=False) + "\n")
                fh.flush()
                os.fsync(fh.fileno())
        except Exception:
            pass


def _event(
    query: str,
    ds: str,
    cx: str,
    strat: str,
    conf: float,
    reason: str,
    lat_ms: float,
) -> dict[str, Any]:
    return {
        "query": query,
        "route": {
            "datasource": ds,
            "complexity": cx,
            "retrieval_strategy": strat,
            "confidence": round(conf, 2),
            "reason": reason,
        },
        "timestamp": _time.time(),
        "latency_ms": round(lat_ms, 3),
    }


def _load_hermes_raw_config() -> Mapping[str, Any]:
    try:
        from hermes_cli.config import load_config

        config = load_config()
    except Exception:
        return {}
    return config if isinstance(config, Mapping) else {}


def _normalize_config_value(value: Any) -> Any:
    """Decode bytes recursively so host config remains stable across loaders."""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, Mapping):
        return {k: _normalize_config_value(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return tuple(_normalize_config_value(v) for v in value)
    if isinstance(value, list):
        return [_normalize_config_value(v) for v in value]
    if isinstance(value, set):
        return {_normalize_config_value(v) for v in value}
    return value


def load_adaptive_query_routing_config(
    cfg: Optional[Mapping[str, Any] | AdaptiveQueryRoutingConfig] = None,
) -> AdaptiveQueryRoutingConfig:
    if isinstance(cfg, AdaptiveQueryRoutingConfig):
        return cfg
    raw_config = _load_hermes_raw_config() if cfg is None else cfg
    return _load_router_config(_normalize_config_value(raw_config))


def classify_query(
    query: str,
    config: Optional[AdaptiveQueryRoutingConfig] = None,
    *,
    available_tools: Optional[Iterable[str]] = None,
) -> QueryRoute:
    t0 = _time.perf_counter()
    route = _classify_pkg(
        query,
        config or load_adaptive_query_routing_config(),
        available_tools=available_tools,
    )
    elapsed = (_time.perf_counter() - t0) * 1000
    _write(
        _event(
            query,
            route.datasource,
            route.complexity,
            route.retrieval_strategy,
            route.confidence,
            getattr(route, "reason", ""),
            elapsed,
        )
    )
    return route


def build_adaptive_query_routing_prompt(
    available_tools: Iterable[str],
    config: Optional[AdaptiveQueryRoutingConfig] = None,
) -> str:
    return _build_prompt_pkg(
        available_tools,
        config or load_adaptive_query_routing_config(),
    )


__all__ = [
    "AdaptiveQueryRoutingConfig",
    "QueryRoute",
    "build_adaptive_query_routing_prompt",
    "classify_query",
    "load_adaptive_query_routing_config",
    "set_persistence_path",
]
