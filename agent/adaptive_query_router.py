"""Adaptive query routing for Hermes — thin shim over hermes-adaptive-router.

This module re-exports the standalone package APIs so that existing Hermes
code (system_prompt.py, tavily provider, tests) can import from
``agent.adaptive_query_router`` without modification.

If hermes-adaptive-router is not installed, falls back to a minimal inline
implementation so the agent does not crash on import.

Persistent event tracking
------------------------
Every call to classify_query() writes one JSON line to
~/.hermes/adaptive_router_history.jsonl (buffered — flushes every 10 events
or on interpreter exit).  No external dependencies.
"""

from __future__ import annotations

import atexit
import json
import os
import threading
import time as _time
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

# --------------------------------------------------------------------------- #
# Buffered JSONL writer — module state
# --------------------------------------------------------------------------- #

_buf = []       # list[dict]
_buf_lock = threading.Lock()
_persist_path = str(Path.home() / ".hermes" / "adaptive_router_history.jsonl")
_writer_on = False


def set_persistence_path(p: str) -> None:
    global _persist_path
    _persist_path = p


def _write(ev: dict) -> None:
    with _buf_lock:
        p = Path(os.path.expanduser(_persist_path))
        p.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(p, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(ev, ensure_ascii=False) + "\n")
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
) -> dict:
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


# --------------------------------------------------------------------------- #
# Try to load from standalone package
# --------------------------------------------------------------------------- #

try:
    from hermes_adaptive_router import (
        AdaptiveQueryRoutingConfig,
        QueryRoute,
        build_adaptive_query_routing_prompt as _build_prompt_pkg,
        classify_query as _classify_pkg,
        load_adaptive_query_routing_config,
    )
    _HAS_PKG = True
except Exception:
    _HAS_PKG = False

# --------------------------------------------------------------------------- #
# Inline fallback (used when hermes-adaptive-router is not installed)
# --------------------------------------------------------------------------- #

if not _HAS_PKG:
    from dataclasses import dataclass, field
    import re

    _URL_RE = re.compile(r"https?://[^\s<>)\]}'\"]+", re.IGNORECASE)
    _WORD_RE = re.compile(r"[\w\u4e00-\u9fff]+", re.UNICODE)

    _FORCE_WEB_KW = (
        "latest", "current", "today", "now", "recent", "news", "breaking",
        "price", "pricing", "release date", "changelog", "version",
        "search", "look up", "lookup", "web search", "online",
        "最新", "今天", "现在", "近期", "新闻", "价格", "定价",
        "版本", "发布日期", "搜索", "查一下", "查下", "联网", "网上",
    )

    _COMPLEX_KW = (
        "compare", "comparison", "versus", " vs ", "tradeoff", "trade-off",
        "benchmarks", "benchmark", "evaluate", "analysis", "analyze",
        "why", "how", "explain", "strategy", "architecture", "root cause", "multi-step",
        "比较", "对比", "权衡", "基准", "评测", "分析", "为什么", "怎么", "如何",
        "原理", "架构", "根因",
    )

    _DIRECT_KW = (
        "who", "what", "when", "where", "define", "meaning",
        "是谁", "是什么", "什么时候", "在哪里", "定义",
    )

    @dataclass(frozen=True)
    class AdaptiveQueryRoutingConfig:
        enabled: bool = True
        simple_max_words: int = 14
        prefer_search_summary: bool = True
        tavily_answer: str | bool = "advanced"
        force_web_keywords: tuple = field(default_factory=lambda: _FORCE_WEB_KW)
        complex_keywords: tuple = field(default_factory=lambda: _COMPLEX_KW)
        direct_keywords: tuple = field(default_factory=lambda: _DIRECT_KW)
        complex_min_signals: int = 2

    @dataclass(frozen=True)
    class QueryRoute:
        datasource: str
        complexity: str
        retrieval_strategy: str
        confidence: float
        reason: str

    _DEF_CFG = AdaptiveQueryRoutingConfig()

    def _bool(v, default):
        if isinstance(v, bool): return v
        if isinstance(v, str):
            n = v.strip().lower()
            if n in {"1","true","yes","on","enabled","enable"}: return True
            if n in {"0","false","no","off","disabled","disable"}: return False
        return default

    def _int(v, default, mn=1, mx=10000):
        try: return min(max(int(v), mn), mx)
        except: return default

    def _tuple(v, default):
        if not v: return tuple(default)
        if isinstance(v, str): return (v,)
        return tuple(x.strip() for x in v if isinstance(x, str) and x.strip()) or tuple(default)

    def _cfg_section(cfg):
        direct = cfg.get("adaptive_query_routing")
        if isinstance(direct, Mapping): return direct
        web = cfg.get("web", {})
        nested = web.get("adaptive_query_routing") or web.get("adaptive_routing")
        if isinstance(nested, Mapping): return nested
        return {}

    def load_adaptive_query_routing_config(cfg=None):
        if cfg is None:
            try:
                from hermes_cli.config import load_config
                cfg = load_config() or {}
            except: cfg = {}
        sec = _cfg_section(cfg if isinstance(cfg, Mapping) else {})
        dc = _DEF_CFG
        ta = sec.get("tavily_answer", dc.tavily_answer)
        if isinstance(ta, str):
            nt = ta.strip().lower()
            if nt in {"false","off","none","no","0"}: ta = False
            elif nt in {"true","on","yes","1"}: ta = True
            elif nt not in {"basic","advanced"}: ta = dc.tavily_answer
        return AdaptiveQueryRoutingConfig(
            enabled=_bool(sec.get("enabled"), dc.enabled),
            simple_max_words=_int(sec.get("simple_max_words"), dc.simple_max_words, 1, 100),
            prefer_search_summary=_bool(sec.get("prefer_search_summary"), dc.prefer_search_summary),
            tavily_answer=ta,
            force_web_keywords=_tuple(sec.get("force_web_keywords"), dc.force_web_keywords),
            complex_keywords=_tuple(sec.get("complex_keywords"), dc.complex_keywords),
            direct_keywords=_tuple(sec.get("direct_keywords"), dc.direct_keywords),
            complex_min_signals=_int(sec.get("complex_min_signals"), dc.complex_min_signals, 1, 10),
        )

    def _any(text, kws):
        t = text.lower()
        return any(kw.lower() in t for kw in kws)

    def _words(text):
        return len(_WORD_RE.findall(text))

    def _has_url(text):
        return bool(_URL_RE.search(text))

    def _tools(t):
        if t is None: return {"web_search", "web_extract"}
        return {str(x) for x in t if str(x).strip()}

    def _route(ds, cx, strat, reason, conf):
        return QueryRoute(ds, cx, strat, round(min(max(conf, 0.0), 1.0), 2), reason)

    def _classify_fallback(query, config=None, *, available_tools=None):
        cfg = config or load_adaptive_query_routing_config()
        tools = _tools(available_tools)
        text = (query or "").strip()
        text_l = f" {text.lower()} "
        words = _words(text)

        if not text:
            r = _route("direct","simple","none","empty query",0.9)
        elif not cfg.enabled:
            r = _route("direct","simple","none","adaptive routing disabled",0.7)
        elif _has_url(text):
            if "web_extract" in tools:
                r = _route("web_extract","intermediate","single_retrieval","query contains URL",0.95)
            elif "web_search" in tools:
                r = _route("web_search","intermediate","single_retrieval","query contains URL but web_extract unavailable",0.75)
            else:
                r = _route("direct","intermediate","none","query contains URL but web tools unavailable",0.55)
        else:
            web = _any(text_l, cfg.force_web_keywords)
            cx_sig = 0
            if _any(text_l, cfg.complex_keywords): cx_sig += 1
            if words > max(cfg.simple_max_words, 18): cx_sig += 1
            if any(m in text for m in ("?","？")) and words > cfg.simple_max_words: cx_sig += 1
            is_cx = cx_sig >= cfg.complex_min_signals
            if web and "web_search" in tools:
                if is_cx:
                    r = _route("web_search","complex","iterative_retrieval","recency/web + multi-hop complexity",0.86)
                else:
                    r = _route("web_search","intermediate","single_retrieval","recency or explicit web-search intent",0.88)
            elif web:
                r = _route("direct","intermediate","none","web intent but web_search unavailable",0.5)
            elif is_cx:
                r = _route("direct","complex","none","complex reasoning without external-data signal",0.65)
            elif words <= cfg.simple_max_words or _any(text_l, cfg.direct_keywords):
                r = _route("direct","simple","none","simple evergreen query",0.82)
            else:
                r = _route("direct","intermediate","none","no external-data signal",0.7)

        _write(_event(query, r.datasource, r.complexity, r.retrieval_strategy,
                      r.confidence, r.reason, 0.05))
        return r


# --------------------------------------------------------------------------- #
# Public classify_query
# --------------------------------------------------------------------------- #

if _HAS_PKG:
    def classify_query(query, config=None, *, available_tools=None):
        t0 = _time.perf_counter()
        route = _classify_pkg(query, config, available_tools=available_tools)
        elapsed = (_time.perf_counter() - t0) * 1000
        _write(_event(
            query,
            route.datasource,
            route.complexity,
            route.retrieval_strategy,
            route.confidence,
            getattr(route, "reason", ""),
            elapsed,
        ))
        return route
else:
    classify_query = _classify_fallback


# --------------------------------------------------------------------------- #
# System prompt builder
# --------------------------------------------------------------------------- #

def _norm_tools(t):
    if t is None: return {"web_search", "web_extract"}
    return {str(x) for x in t if str(x).strip()}


def build_adaptive_query_routing_prompt(available_tools, config=None):
    cfg = config or load_adaptive_query_routing_config()
    tools = _norm_tools(available_tools)
    has_s = "web_search" in tools
    has_e = "web_extract" in tools
    if not cfg.enabled or not (has_s or has_e):
        return ""
    sl = ("- intermediate / web_search: use for current, recent, news, pricing, "
          "release/version, or explicitly web-search questions." if has_s
          else "- intermediate / web_search: unavailable in this session.")
    el = ("- page-specific / web_extract: use for user-provided URLs, PDFs, or when "
          "search snippets/summaries are insufficient; do not extract every result by default."
          if has_e else "- page-specific / web_extract: unavailable in this session; "
          "rely on search summaries/snippets when possible.")
    tl = ""
    if cfg.prefer_search_summary and has_s:
        tl = (" When the search backend is Tavily and web_search returns a Tavily "
              "`data.answer` AI summary or rich result descriptions, treat that as the "
              "first retrieval layer; answer from it with source URLs when sufficient "
              "instead of reflexively calling web_extract.")
    return (
        "Adaptive query routing (web/retrieval): before using web tools, classify the "
        "user query and pick the cheapest sufficient layer.\n"
        "- simple / direct: answer from model knowledge for stable evergreen facts, "
        "definitions, and straightforward reasoning; do not call web tools just because "
        "they exist.\n"
        f"{sl}\n"
        f"{el}\n"
        "- complex / iterative: for multi-hop current research, compare/investigate with "
        "web_search first, inspect the answer/snippets, then web_extract only selected "
        f"high-value URLs if needed.\n{tl}"
    ).strip()


__all__ = [
    "AdaptiveQueryRoutingConfig",
    "QueryRoute",
    "build_adaptive_query_routing_prompt",
    "classify_query",
    "load_adaptive_query_routing_config",
    "set_persistence_path",
]
