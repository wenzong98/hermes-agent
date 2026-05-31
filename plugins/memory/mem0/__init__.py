"""Mem0 memory plugin — MemoryProvider interface.

Supports two backends:
- Local library mode: MiniMax + FastEmbed + local ChromaDB via ``mem0.Memory``
- Cloud mode: Mem0 Platform API via ``mem0.MemoryClient``

Local mode is activated by ``local_mode: true`` in ``$HERMES_HOME/mem0.json``
or ``MEM0_LOCAL_MODE=true`` in the environment.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, List

from agent.memory_provider import MemoryProvider
from tools.registry import tool_error

logger = logging.getLogger(__name__)

# Circuit breaker: after this many consecutive failures, pause API calls
# for _BREAKER_COOLDOWN_SECS to avoid hammering a down server.
_BREAKER_THRESHOLD = 5
_BREAKER_COOLDOWN_SECS = 120
_DEDUP_THRESHOLD = 0.92


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def _load_config() -> dict:
    """Load config from env vars, with $HERMES_HOME/mem0.json overrides.

    Environment variables provide defaults; mem0.json (if present) overrides
    individual keys. This avoids silent failures when the JSON file exists but
    only stores local-mode settings while secrets live in ``~/.hermes/.env``.
    """
    from hermes_constants import get_hermes_home

    config = {
        # Cloud mode
        "api_key": os.environ.get("MEM0_API_KEY", ""),
        # Shared
        "user_id": os.environ.get("MEM0_USER_ID", "hermes-user"),
        "agent_id": os.environ.get("MEM0_AGENT_ID", "hermes"),
        "rerank": True,
        "keyword_search": False,
        # Local mode
        "local_mode": os.environ.get("MEM0_LOCAL_MODE", "false").lower() == "true",
        "minimax_api_key": os.environ.get("MINIMAX_API_KEY", ""),
        "minimax_base_url": os.environ.get("MINIMAX_BASE_URL", "https://api.minimaxi.com/v1"),
        "minimax_model": os.environ.get("MEM0_MINIMAX_MODEL", "MiniMax-M2.7"),
        "chroma_path": os.environ.get("MEM0_CHROMA_PATH", "/tmp/mem0_chroma"),
        "collection_name": os.environ.get("MEM0_COLLECTION", "hermes_memories"),
        "embedder_model": os.environ.get("MEM0_EMBEDDER_MODEL", "BAAI/bge-base-en-v1.5"),
    }

    config_path = get_hermes_home() / "mem0.json"
    if config_path.exists():
        try:
            file_cfg = json.loads(config_path.read_text(encoding="utf-8"))
            config.update({k: v for k, v in file_cfg.items() if v is not None and v != ""})
        except Exception:
            pass

    return config


# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------

PROFILE_SCHEMA = {
    "name": "mem0_profile",
    "description": (
        "Retrieve all stored memories about the user — preferences, facts, "
        "project context. Fast, no reranking. Use at conversation start."
    ),
    "parameters": {"type": "object", "properties": {}, "required": []},
}

SEARCH_SCHEMA = {
    "name": "mem0_search",
    "description": (
        "Search memories by meaning. Returns relevant facts ranked by similarity. "
        "Set rerank=true for higher accuracy on important queries."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "What to search for."},
            "rerank": {"type": "boolean", "description": "Enable reranking for precision (default: false)."},
            "top_k": {"type": "integer", "description": "Max results (default: 10, max: 50)."},
        },
        "required": ["query"],
    },
}

CONCLUDE_SCHEMA = {
    "name": "mem0_conclude",
    "description": (
        "Store a durable fact about the user. Stored verbatim (no LLM extraction). "
        "Use for explicit preferences, corrections, or decisions."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "conclusion": {"type": "string", "description": "The fact to store."},
        },
        "required": ["conclusion"],
    },
}

FORGET_SCHEMA = {
    "name": "mem0_forget",
    "description": (
        "Delete a specific memory by its ID. Use when user explicitly asks to forget, remove, or delete a stored fact. "
        "Get memory IDs from mem0_profile or mem0_search results."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "memory_id": {"type": "string", "description": "The ID of the memory to delete."},
        },
        "required": ["memory_id"],
    },
}

CURATE_SCHEMA = {
    "name": "mem0_curate",
    "description": (
        "Curate and manage memories: deduplicate, categorize, summarize, and clean up stale entries. "
        "Use when user asks to review, organize, clean up, or manage their memories."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["deduplicate", "summarize", "categorize", "list_stale"],
                "description": "The curation action to perform.",
            },
        },
        "required": ["action"],
    },
}


# ---------------------------------------------------------------------------
# MemoryProvider implementation
# ---------------------------------------------------------------------------

class Mem0MemoryProvider(MemoryProvider):
    """Mem0 memory provider supporting both local and cloud backends."""

    def __init__(self):
        self._config = None
        self._client = None  # cloud MemoryClient
        self._memory = None  # local Memory instance
        self._client_lock = threading.Lock()
        self._api_key = ""
        self._user_id = "hermes-user"
        self._agent_id = "hermes"
        self._rerank = True
        self._local_mode = False
        self._prefetch_result = ""
        self._prefetch_lock = threading.Lock()
        self._prefetch_thread = None
        self._sync_thread = None
        # Circuit breaker state
        self._consecutive_failures = 0
        self._breaker_open_until = 0.0

    @property
    def name(self) -> str:
        return "mem0"

    def is_available(self) -> bool:
        cfg = _load_config()
        if cfg.get("local_mode"):
            return bool(cfg.get("minimax_api_key"))
        return bool(cfg.get("api_key"))

    def save_config(self, values, hermes_home):
        """Write config to $HERMES_HOME/mem0.json."""
        config_path = Path(hermes_home) / "mem0.json"
        existing = {}
        if config_path.exists():
            try:
                existing = json.loads(config_path.read_text())
            except Exception:
                pass
        existing.update(values)
        config_path.write_text(json.dumps(existing, indent=2))

    def get_config_schema(self):
        return [
            {"key": "local_mode", "description": "Use local Mem0 library mode (MiniMax + FastEmbed + ChromaDB)", "default": "false", "choices": ["true", "false"]},
            {"key": "api_key", "description": "Mem0 Platform API key (cloud mode)", "secret": True, "required": False, "env_var": "MEM0_API_KEY", "url": "https://app.mem0.ai"},
            {"key": "minimax_api_key", "description": "MiniMax API key (local mode)", "secret": True, "required": False, "env_var": "MINIMAX_API_KEY"},
            {"key": "minimax_base_url", "description": "MiniMax API base URL (OpenAI-compatible /v1 endpoint)", "default": "https://api.minimaxi.com/v1"},
            {"key": "chroma_path", "description": "Local ChromaDB path", "default": "/tmp/mem0_chroma"},
            {"key": "collection_name", "description": "ChromaDB collection name", "default": "hermes_memories"},
            {"key": "user_id", "description": "User identifier", "default": "hermes-user"},
            {"key": "agent_id", "description": "Agent identifier", "default": "hermes"},
            {"key": "rerank", "description": "Enable reranking for recall", "default": "true", "choices": ["true", "false"]},
        ]

    def _local_memory_config(self) -> Dict[str, Any]:
        cfg = self._config or _load_config()
        return {
            "llm": {
                "provider": "minimax",
                "config": {
                    "model": cfg.get("minimax_model", "MiniMax-M2.7"),
                    "api_key": cfg.get("minimax_api_key", "") or os.environ.get("MINIMAX_API_KEY", ""),
                    "minimax_base_url": cfg.get("minimax_base_url", "https://api.minimaxi.com/v1"),
                },
            },
            "embedder": {
                "provider": "fastembed",
                "config": {
                    "model": cfg.get("embedder_model", "BAAI/bge-base-en-v1.5"),
                },
            },
            "vector_store": {
                "provider": "chroma",
                "config": {
                    "collection_name": cfg.get("collection_name", "hermes_memories"),
                    "path": cfg.get("chroma_path", "/tmp/mem0_chroma"),
                },
            },
        }

    def _get_local_memory(self):
        """Thread-safe local Memory accessor with lazy initialization."""
        with self._client_lock:
            if self._memory is not None:
                return self._memory
            try:
                from mem0 import Memory
            except ImportError as e:
                raise RuntimeError(
                    "mem0 local mode requires mem0ai/chromadb/fastembed. Run: pip install mem0ai chromadb fastembed"
                ) from e
            self._memory = Memory.from_config(self._local_memory_config())
            return self._memory

    def _get_cloud_client(self):
        """Thread-safe cloud client accessor with lazy initialization."""
        with self._client_lock:
            if self._client is not None:
                return self._client
            try:
                from mem0 import MemoryClient
                self._client = MemoryClient(api_key=self._api_key)
                return self._client
            except ImportError as e:
                raise RuntimeError("mem0 package not installed. Run: pip install mem0ai") from e

    def _get_client(self):
        """Return the active backend client (local Memory or cloud MemoryClient)."""
        if self._local_mode:
            return self._get_local_memory()
        return self._get_cloud_client()

    def _is_breaker_open(self) -> bool:
        """Return True if the circuit breaker is tripped (too many failures)."""
        if self._consecutive_failures < _BREAKER_THRESHOLD:
            return False
        if time.monotonic() >= self._breaker_open_until:
            # Cooldown expired — reset and allow a retry
            self._consecutive_failures = 0
            return False
        return True

    def _record_success(self):
        self._consecutive_failures = 0

    def _record_failure(self):
        self._consecutive_failures += 1
        if self._consecutive_failures >= _BREAKER_THRESHOLD:
            self._breaker_open_until = time.monotonic() + _BREAKER_COOLDOWN_SECS
            logger.warning(
                "Mem0 circuit breaker tripped after %d consecutive failures. "
                "Pausing API calls for %ds.",
                self._consecutive_failures,
                _BREAKER_COOLDOWN_SECS,
            )

    def initialize(self, session_id: str, **kwargs) -> None:
        self._config = _load_config()
        self._local_mode = bool(self._config.get("local_mode", False))
        self._api_key = self._config.get("api_key", "")
        # Prefer gateway-provided user_id for per-user memory scoping;
        # fall back to config/env default for CLI (single-user) sessions.
        self._user_id = kwargs.get("user_id") or self._config.get("user_id", "hermes-user")
        self._agent_id = self._config.get("agent_id", "hermes")
        self._rerank = self._config.get("rerank", True)

        # Eagerly validate local mode so startup fails loudly if dependencies/config are wrong.
        if self._local_mode:
            self._get_local_memory()
            logger.info(
                "Mem0 local mode initialized (MiniMax + FastEmbed + ChromaDB at %s)",
                self._config.get("chroma_path", "/tmp/mem0_chroma"),
            )

    def _read_filters(self) -> Dict[str, Any]:
        """Filters for search/get_all — scoped to user only for cross-session recall."""
        return {"user_id": self._user_id}

    def _write_filters(self) -> Dict[str, Any]:
        """Filters for add — scoped to user + agent for attribution."""
        return {"user_id": self._user_id, "agent_id": self._agent_id}

    @staticmethod
    def _unwrap_results(response: Any) -> list:
        """Normalize Mem0 API response — v2 wraps results in {"results": [...]}.

        The local ``Memory`` backend also returns the same shape in mem0ai v2.
        """
        if isinstance(response, dict):
            return response.get("results", [])
        if isinstance(response, list):
            return response
        return []

    def _local_get_all_memories(self) -> list[dict[str, Any]]:
        """Read all local-mode memories directly from Chroma.

        Mem0's ``get_all`` defaults to ``top_k=20``, which truncates larger
        memory sets. Reading from the underlying collection preserves the full
        set for profile dumps.
        """
        memory = self._get_local_memory()
        collection = memory.vector_store.collection
        raw = collection.get(where=self._read_filters(), include=["metadatas"])
        results = []
        for meta in raw.get("metadatas", []) or []:
            if not meta:
                continue
            text = meta.get("data") or meta.get("memory") or ""
            if not text:
                continue
            results.append({"memory": text})
        return results

    def _local_semantic_search(self, query: str, top_k: int) -> list[dict[str, Any]]:
        """Run local semantic search directly against Chroma using the configured embedder.

        The Mem0 local ``search()`` path currently produces poor rankings for
        this setup even when the stored embeddings are good. Querying Chroma
        directly with the same embedder yields correct nearest neighbours.
        """
        memory = self._get_local_memory()
        embedding = memory.embedding_model.embed(query, memory_action="search")
        raw = memory.vector_store.collection.query(
            query_embeddings=[[float(x) for x in embedding]],
            n_results=top_k,
            where=self._read_filters(),
            include=["metadatas", "distances"],
        )

        metadatas = (raw.get("metadatas") or [[]])[0]
        distances = (raw.get("distances") or [[]])[0]
        results = []
        for meta, distance in zip(metadatas, distances):
            if not meta:
                continue
            text = meta.get("data") or meta.get("memory") or ""
            if not text:
                continue
            score = 0.0 if distance is None else max(0.0, 1.0 - float(distance))
            results.append({"memory": text, "score": score})
        return results

    def system_prompt_block(self) -> str:
        backend = "local" if self._local_mode else "cloud"
        return (
            "# Mem0 Memory\n"
            f"Active ({backend}). User: {self._user_id}.\n"
            "Use mem0_search to find memories, mem0_conclude to store facts, "
            "mem0_forget to delete a memory, mem0_profile for a full overview, "
            "mem0_curate to deduplicate/categorize/summarize memories."
        )

    def _run_curation(self, action: str) -> Dict[str, Any]:
        """Run a curation action on local-mode memories."""
        collection = self._get_local_memory().vector_store.collection
        raw = collection.get(where=self._read_filters(), include=["metadatas"])
        metadatas = raw.get("metadatas") or []
        if not metadatas:
            return {"action": action, "result": "No memories found.", "count": 0}

        if action == "categorize":
            categories = {
                "config": [], "preference": [], "fact": [], "skill": [],
                "portfolio": [], "path": [], "other": [],
            }
            _CAT_KEYWORDS = {
                "config": ["模型", "model", "provider", "API", "配置", "config", "proxy", "代理", "端口"],
                "preference": ["偏好", "风格", "喜欢", "要求", "注重", "prefers", "偏好"],
                "fact": ["住在", "名字", "是", "有", "路径", "挂载", "目录", "lives in", "name is"],
                "skill": ["skill", "脚本", "命令", "安装", "已装", "yt-dlp"],
                "portfolio": ["基金", "ETF", "QDII", "定投", "持仓", "投资", "策略", "portfolio"],
                "path": ["归档", "博主", "视频", "目录"],
            }
            for meta in metadatas:
                text = (meta.get("data") or meta.get("memory") or "").strip()
                if not text:
                    continue
                matched = False
                for cat, keywords in _CAT_KEYWORDS.items():
                    if any(kw.lower() in text.lower() for kw in keywords):
                        categories[cat].append(text)
                        matched = True
                        break
                if not matched:
                    categories["other"].append(text)

            result_lines = [f"# Memory Categorization ({sum(len(v) for v in categories.values())} total)"]
            for cat, items in categories.items():
                if items:
                    result_lines.append(f"\n## {cat.title()} ({len(items)})")
                    for item in items:
                        result_lines.append(f"- {item[:120]}")
            return {"action": action, "result": "\n".join(result_lines), "categories": {k: len(v) for k, v in categories.items() if v}}

        elif action == "deduplicate":
            from collections import defaultdict as _dd
            groups = _dd(list)
            for meta in metadatas:
                text = (meta.get("data") or meta.get("memory") or "").strip()
                key = text[:100]
                groups[key].append(meta.get("id", ""))
            dup_groups = {k: v for k, v in groups.items() if len(v) > 1}
            dup_count = sum(len(v) - 1 for v in dup_groups.values())
            result_lines = [f"# Dedup Analysis: {len(metadatas)} memories, {len(dup_groups)} duplicate groups, {dup_count} removable"]
            for key, ids in sorted(dup_groups.items()):
                result_lines.append(f"\n- \"{key[:80]}\" x{len(ids)} (keep newest, delete {len(ids)-1})")
            return {"action": action, "duplicate_groups": len(dup_groups), "removable_count": dup_count, "result": "\n".join(result_lines)}

        elif action == "summarize":
            all_texts = [(meta.get("data") or meta.get("memory") or "").strip() for meta in metadatas]
            all_texts = [t for t in all_texts if t]
            summary_parts = [
                f"Total memories: {len(all_texts)}",
                f"\n## User Profile Summary",
            ]
            user_facts = [t for t in all_texts if any(kw in t for kw in ["enzo", "用户", "User"])]
            config_items = [t for t in all_texts if any(kw in t for kw in ["模型", "model", "配置", "config", "代理"])]
            pref_items = [t for t in all_texts if any(kw in t for kw in ["偏好", "风格", "prefers", "要求"])]
            path_items = [t for t in all_texts if any(kw in t for kw in ["路径", "path", "目录", "挂载"])]
            portfolio_items = [t for t in all_texts if any(kw in t for kw in ["ETF", "基金", "定投", "portfolio"])]

            if user_facts:
                summary_parts.append("\n### Facts")
                summary_parts.extend(f"- {t[:100]}" for t in user_facts[:10])
            if config_items:
                summary_parts.append("\n### Config/Tools")
                summary_parts.extend(f"- {t[:100]}" for t in config_items[:10])
            if pref_items:
                summary_parts.append("\n### Preferences")
                summary_parts.extend(f"- {t[:100]}" for t in pref_items[:10])
            if path_items:
                summary_parts.append("\n### Paths")
                summary_parts.extend(f"- {t[:100]}" for t in path_items[:5])
            if portfolio_items:
                summary_parts.append("\n### Portfolio")
                summary_parts.extend(f"- {t[:100]}" for t in portfolio_items[:5])

            return {"action": action, "result": "\n".join(summary_parts)}

        elif action == "list_stale":
            import datetime as _dt
            now_ts = _dt.datetime.now(_dt.timezone.utc).isoformat()
            stale = []
            for meta in metadatas:
                created = meta.get("created_at", "") or ""
                text = (meta.get("data") or meta.get("memory") or "").strip()
                if created:
                    try:
                        age_days = (_dt.datetime.now(_dt.timezone.utc) - _dt.datetime.fromisoformat(created)).days
                        if age_days > 14:
                            stale.append({"id": meta.get("id", ""), "text": text[:80], "age_days": age_days})
                    except (ValueError, TypeError):
                        pass
            stale.sort(key=lambda x: x["age_days"], reverse=True)
            result_lines = [f"# Stale Memories (>14 days): {len(stale)}"]
            for s in stale[:20]:
                result_lines.append(f"- [{s['id'][:12]}] {s['age_days']}d old: {s['text']}")
            return {"action": action, "stale_count": len(stale), "result": "\n".join(result_lines)}

        return {"action": action, "error": f"Unknown curation action: {action}"}

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        if self._prefetch_thread and self._prefetch_thread.is_alive():
            self._prefetch_thread.join(timeout=3.0)
        with self._prefetch_lock:
            result = self._prefetch_result
            self._prefetch_result = ""
        if not result:
            return ""
        return f"## Mem0 Memory\n{result}"

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        if self._is_breaker_open():
            return

        def _run():
            try:
                client = self._get_client()
                results = self._unwrap_results(
                    client.search(
                        query=query,
                        filters=self._read_filters(),
                        rerank=self._rerank,
                        top_k=5,
                    )
                )
                if results:
                    lines = [r.get("memory", "") for r in results if r.get("memory")]
                    with self._prefetch_lock:
                        self._prefetch_result = "\n".join(f"- {l}" for l in lines)
                self._record_success()
            except Exception as e:
                self._record_failure()
                logger.debug("Mem0 prefetch failed: %s", e)

        self._prefetch_thread = threading.Thread(target=_run, daemon=True, name="mem0-prefetch")
        self._prefetch_thread.start()

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "") -> None:
        """Send the turn to Mem0 for fact extraction (non-blocking).

        Includes a semantic dedup pre-check: if a very similar memory already
        exists (score > 0.92), the sync is skipped to avoid storing duplicates.
        """
        if self._is_breaker_open():
            return

        def _sync():
            try:
                client = self._get_client()
                messages = [
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": assistant_content},
                ]
                if self._local_mode:
                    results = self._local_semantic_search(query=user_content, top_k=3)
                    for r in results:
                        if r.get("score", 0) > _DEDUP_THRESHOLD:
                            logger.debug(
                                "sync_turn skipped: similar memory exists (score=%.2f) "
                                "for content=%.80s",
                                r["score"], user_content,
                            )
                            self._record_success()
                            return
                client.add(messages, **self._write_filters())
                self._record_success()
            except Exception as e:
                self._record_failure()
                logger.warning("Mem0 sync failed: %s", e)

        # Wait for any previous sync before starting a new one
        if self._sync_thread and self._sync_thread.is_alive():
            self._sync_thread.join(timeout=5.0)

        self._sync_thread = threading.Thread(target=_sync, daemon=True, name="mem0-sync")
        self._sync_thread.start()

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [PROFILE_SCHEMA, SEARCH_SCHEMA, CONCLUDE_SCHEMA, FORGET_SCHEMA, CURATE_SCHEMA]

    def handle_tool_call(self, tool_name: str, args: dict, **kwargs) -> str:
        if self._is_breaker_open():
            return json.dumps(
                {
                    "error": "Mem0 API temporarily unavailable (multiple consecutive failures). Will retry automatically."
                }
            )

        try:
            client = self._get_client()
        except Exception as e:
            return tool_error(str(e))

        if tool_name == "mem0_profile":
            try:
                memories = (
                    self._local_get_all_memories()
                    if self._local_mode
                    else self._unwrap_results(client.get_all(filters=self._read_filters()))
                )
                self._record_success()
                if not memories:
                    return json.dumps({"result": "No memories stored yet."})
                lines = [m.get("memory", "") for m in memories if m.get("memory")]
                return json.dumps({"result": "\n".join(lines), "count": len(lines)})
            except Exception as e:
                self._record_failure()
                return tool_error(f"Failed to fetch profile: {e}")

        elif tool_name == "mem0_search":
            query = args.get("query", "")
            if not query:
                return tool_error("Missing required parameter: query")
            rerank = args.get("rerank", False)
            top_k = min(int(args.get("top_k", 10)), 50)
            try:
                results = (
                    self._local_semantic_search(query=query, top_k=top_k)
                    if self._local_mode
                    else self._unwrap_results(
                        client.search(
                            query=query,
                            filters=self._read_filters(),
                            rerank=rerank,
                            top_k=top_k,
                        )
                    )
                )
                self._record_success()
                if not results:
                    return json.dumps({"result": "No relevant memories found."})
                items = [{"memory": r.get("memory", ""), "score": r.get("score", 0)} for r in results]
                return json.dumps({"results": items, "count": len(items)})
            except Exception as e:
                self._record_failure()
                return tool_error(f"Search failed: {e}")

        elif tool_name == "mem0_conclude":
            conclusion = args.get("conclusion", "")
            if not conclusion:
                return tool_error("Missing required parameter: conclusion")
            try:
                client.add(
                    [{"role": "user", "content": conclusion}],
                    **self._write_filters(),
                    infer=False,
                )
                self._record_success()
                return json.dumps({"result": "Fact stored."})
            except Exception as e:
                self._record_failure()
                return tool_error(f"Failed to store: {e}")

        elif tool_name == "mem0_forget":
            memory_id = args.get("memory_id", "")
            if not memory_id:
                return tool_error("Missing required parameter: memory_id")
            try:
                if self._local_mode:
                    collection = self._get_local_memory().vector_store.collection
                    collection.delete(ids=[memory_id])
                else:
                    client.delete(memory_id)
                self._record_success()
                return json.dumps({"result": f"Memory {memory_id[:12]}... deleted."})
            except Exception as e:
                self._record_failure()
                return tool_error(f"Failed to delete: {e}")

        elif tool_name == "mem0_curate":
            action = args.get("action", "")
            if not action:
                return tool_error("Missing required parameter: action")
            if not self._local_mode:
                return tool_error("mem0_curate is only available in local mode.")
            try:
                result = self._run_curation(action)
                self._record_success()
                return json.dumps(result, ensure_ascii=False)
            except Exception as e:
                self._record_failure()
                return tool_error(f"Curation failed: {e}")

        return tool_error(f"Unknown tool: {tool_name}")

    def shutdown(self) -> None:
        for t in (self._prefetch_thread, self._sync_thread):
            if t and t.is_alive():
                t.join(timeout=5.0)
        with self._client_lock:
            self._client = None
            self._memory = None


def register(ctx) -> None:
    """Register Mem0 as a memory provider plugin."""
    ctx.register_memory_provider(Mem0MemoryProvider())
