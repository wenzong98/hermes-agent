"""
Task Complexity Router — 任务复杂度路由

在 LLM 调用前根据 query / tool 信号判断任务复杂度，
自动选择合适的模型 tier：

- simple  → 轻量快速模型（MiniMax-M2.7 高速版 / deepseek-v4-flash）
- complex → 高质量模型（kimi-k2.6 / GPT-5）

当前行为：
1. 优先走纯启发式规则；只有规则未命中时，才会调用 MiniMax 高速版仲裁
2. 可接受显式的调用方 override，并保证其优先于自动路由
3. 工具级控制：特定工具可强制升级到 complex tier
4. JSONL 追踪：每次路由决策写入日志，便于分析调优
5. 会更新 agent 的有效主路由（`_primary_runtime`），与 fallback 恢复逻辑协同

集成点：
- agent/conversation_loop.py  ：用户消息入口分类与 effective route 应用
- agent/tool_executor.py      ：工具执行前按需升级复杂度
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
import threading
from contextvars import ContextVar
from copy import deepcopy
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class Complexity(Enum):
    UNKNOWN = "unknown"
    SIMPLE = "simple"
    COMPLEX = "complex"
    # 强制 tier，不走分类器
    FORCE_SIMPLE = "force_simple"   # 读取类工具
    FORCE_COMPLEX = "force_complex" # 代码实现类工具


@dataclass
class RouteDecision:
    complexity: Complexity
    primary_signal: str       # 触发决策的主要信号
    matched_rules: list[str]  # 命中的规则列表
    suggested_model: str      # 建议模型
    suggested_provider: str   # 建议 provider
    reason: str               # 人类可读原因
    latency_ms: float = 0.0


@dataclass(frozen=True)
class EffectiveModelRoute:
    """Single source of truth for the active model route of a turn."""

    provider: str
    model: str
    decision: RouteDecision
    source: str
    normalized_query: str
    user_override: Optional[str] = None


@dataclass
class RoutingEvent:
    """写入 JSONL 的每次路由记录"""
    timestamp: str
    query: str
    complexity: str
    primary_signal: str
    matched_rules: list
    suggested_model: str
    suggested_provider: str
    reason: str
    tool_name: Optional[str] = None
    override: bool = False    # 用户是否手动指定模型
    latency_ms: float = 0.0

    def to_json_line(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


# ---------------------------------------------------------------------------
# Complexity signals — 基于50条 LLM 标注样本重新设计的启发式规则
# ---------------------------------------------------------------------------

# 匹配则 → complex（优先级高）
_COMPLEX_PATTERNS: list[str] = [
    # 代码构建/配置文件
    r"dockerfile", r"makefile", r"单元测试",
    # 创建项目/系统
    r"写一个.*爬虫", r"创建一个.*项目", r"帮我创建一个",
    r"写一个.*项目", r"写一个.*系统", r"写一个.*应用",
    r"算法", r"架构", r"系统设计",
    # 调研分析
    r"调研", r"研究",
    # 代码 debug/review/重构
    r"\bdebug\b",
    r"帮我review", r"帮我分析这段代码",
    r"帮我优化这段", r"帮我.*修复", r"帮我重构这段",
    r"内存泄漏", r"性能.*优化",
    # 部署/迁移
    r"部署", r"迁移数据库",
    # 多动作组合
    r"验证.*修复", r"深度验证", r"并.*修复",
    r"增加更多.*测试", r"写一个.*正则",
]

# 匹配则 → simple（优先级低，只有 complex 未命中时才生效）
_SIMPLE_PATTERNS: list[str] = [
    # 简单 CLI 命令
    r"^list", r"^ls", r"^cat", r"^head", r"^pwd", r"^whoami", r"^date", r"^uptime",
    # 读/查看动作
    r"^查看", r"^显示", r"^列出", r"^读", r"^统计",
    r"^给我看看", r"^看看",
    # 查询类
    r"^查一下", r"^查下", r"^帮我查",
    # 问答题
    r"是什么", r"什么意思", r"哪里", r"哪个",
    # 简单文件操作
    r"复制.*目录", r"移动.*目录", r"整理.*文件", r"压缩.*文件夹", r"清理缓存",
    # 简单生成
    r"生成密钥", r"生成证书", r"缩小图片", r"调整尺寸",
    # 状态检查
    r"磁盘使用", r"内存占用", r"网络连接", r"配置信息",
    # 简单脚本
    r"帮我写shell脚本",
    # 更新操作
    r"更新.*版本", r"升级.*依赖",
]

# 仲裁模型配置（MiniMax 高速版）
_ARBITRATION_PROVIDER = "minimax"
_ARBITRATION_MODEL = "MiniMax-M2.7-highspeed"

# 工具复杂度映射 — 已知工具的难度级别
TOOL_COMPLEXITY_MAP: dict[str, Complexity] = {
    # 强制 simple
    "read_file": Complexity.FORCE_SIMPLE,
    "search_files": Complexity.FORCE_SIMPLE,
    "terminal": Complexity.FORCE_SIMPLE,      # 但内容决定一切，下游会覆盖
    "list_directory": Complexity.FORCE_SIMPLE,
    "session_search": Complexity.FORCE_SIMPLE,
    "web_search": Complexity.FORCE_SIMPLE,
    "web_extract": Complexity.FORCE_SIMPLE,
    "get_weather": Complexity.FORCE_SIMPLE,
    "skill_view": Complexity.FORCE_SIMPLE,
    "skills_list": Complexity.FORCE_SIMPLE,
    "cronjob": Complexity.FORCE_SIMPLE,      # list 操作为主
    "send_message": Complexity.FORCE_SIMPLE,
    "mcp_*": Complexity.FORCE_SIMPLE,       # MCP 工具大部分是读取

    # 强制 complex
    "execute_code": Complexity.FORCE_COMPLEX,
    "write_file": Complexity.FORCE_COMPLEX,
    "patch": Complexity.FORCE_COMPLEX,
    "delegate_task": Complexity.FORCE_COMPLEX,
    "browser_navigate": Complexity.FORCE_COMPLEX,  # 浏览器操作
    "browser_click": Complexity.FORCE_COMPLEX,
    "browser_type": Complexity.FORCE_COMPLEX,
    "image_generate": Complexity.FORCE_COMPLEX,
    "video_gen": Complexity.FORCE_COMPLEX,
    "text_to_speech": Complexity.FORCE_COMPLEX,
    "skill_manage": Complexity.FORCE_COMPLEX,  # 写操作
    "memory": Complexity.FORCE_COMPLEX,         # 写操作
    "todo": Complexity.FORCE_COMPLEX,           # 写操作
    "plan": Complexity.FORCE_COMPLEX,
    "spike": Complexity.FORCE_COMPLEX,
}


def _resolve_complexity(complexity: Complexity) -> Complexity:
    """Map FORCE_* to base complexity so MODEL_TIERS lookup works."""
    if complexity == Complexity.FORCE_COMPLEX:
        return Complexity.COMPLEX
    if complexity == Complexity.FORCE_SIMPLE:
        return Complexity.SIMPLE
    return complexity


def get_tool_complexity(tool_name: str) -> Complexity:
    """
    查询工具的复杂度级别。

    支持前缀通配（如 ``mcp_*``），返回 FORCE_COMPLEX / FORCE_SIMPLE
    或对应的 Complexity 枚举值。未匹配到的工具返回 Complexity.UNKNOWN。
    """
    for pattern, complexity in TOOL_COMPLEXITY_MAP.items():
        if pattern.endswith("*"):
            prefix = pattern[:-1]
            if tool_name.startswith(prefix):
                return complexity
        elif tool_name == pattern:
            return complexity
    return Complexity.UNKNOWN


def get_tool_tier_for_upgrade(tool_name: str) -> Complexity | None:
    """
    Return the base Complexity tier a tool demands, or None if the tool
    doesn't force an upgrade from the simple tier.

    This collapses FORCE_COMPLEX → COMPLEX and FORCE_SIMPLE → SIMPLE
    so callers can use it directly for MODEL_TIERS lookup and comparison.
    """
    tc = get_tool_complexity(tool_name)
    if tc in (Complexity.FORCE_COMPLEX, Complexity.FORCE_SIMPLE):
        return _resolve_complexity(tc)
    if tc == Complexity.COMPLEX:
        return Complexity.COMPLEX
    return None  # tool doesn't force any tier — leave current model unchanged


# 模型分级配置（可从 config 覆盖）
_DEFAULT_MODEL_TIERS = {
    Complexity.SIMPLE: {
        "default": {
            "provider": "minimax",
            "model": "MiniMax-M2.7-highspeed",
        },
        "fallback": [
            {"provider": "deepseek", "model": "deepseek-v4-flash"},
        ],
    },
    Complexity.COMPLEX: {
        "default": {
            "provider": "kimi",
            "model": "kimi-k2.6",
        },
        "fallback": [
            {"provider": "anthropic", "model": "claude-sonnet-4-20250514"},
            {"provider": "openai", "model": "gpt-4o"},
        ],
    },
}

MODEL_TIERS = deepcopy(_DEFAULT_MODEL_TIERS)

_INLINE_MODEL_OVERRIDE_RE = re.compile(
    r"^\s*@(?P<override>[A-Za-z0-9._:-]+(?:/[A-Za-z0-9._:-]+)?)\s+(?P<rest>.+?)\s*$"
)


# ---------------------------------------------------------------------------
# Core classifier — 纯启发式，无 LLM 调用
# ---------------------------------------------------------------------------

def estimate_tokens(text: str) -> int:
    """估算 token 数：中文≈1.5字/token，英文≈3.5字符/token"""
    chinese_chars = len(re.findall(r"[\u4e00-\u9fff]", text))
    non_chinese = len(text) - chinese_chars
    return int(chinese_chars * 0.67) + int(non_chinese * 0.29)


def _make_decision(
    complexity: Complexity,
    primary_signal: str,
    matched_rules: list[str],
    reason: str,
    latency_ms: float,
) -> RouteDecision:
    """构建 RouteDecision 并填充 suggested_model/provider。"""
    resolved = _resolve_complexity(complexity)
    tier = MODEL_TIERS.get(resolved, MODEL_TIERS[Complexity.SIMPLE])
    default = tier["default"]
    return RouteDecision(
        complexity=complexity,
        primary_signal=primary_signal,
        matched_rules=matched_rules,
        suggested_model=default["model"],
        suggested_provider=default["provider"],
        reason=reason,
        latency_ms=latency_ms,
    )


# ---------------------------------------------------------------------------
# 低置信度仲裁 — 调用 MiniMax 高速版二次判断
# ---------------------------------------------------------------------------

def _arbitrate_with_minimax(query: str) -> tuple[Complexity, str]:
    """
    当启发式分类置信度低于阈值时，调用 MiniMax 高速版进行仲裁。

    使用流式输出 + 早停：一旦在 text delta 中检测到 simple/complex 即返回，
    无需等待完整的 thinking 推理过程。

    返回 (complexity, reason)。
    如果仲裁失败，返回 (Complexity.SIMPLE, "仲裁失败，默认 simple")。
    """
    try:
        from hermes_cli.auth import resolve_api_key_provider_credentials
        import os

        # 解析 MiniMax 凭证
        creds = resolve_api_key_provider_credentials("minimax")
        api_key = creds.get("api_key") or os.environ.get("MINIMAX_API_KEY", "")
        base_url = creds.get("base_url") or os.environ.get("MINIMAX_BASE_URL", "")
        if not base_url:
            base_url = "https://api.minimaxi.com/anthropic"

        from anthropic import Anthropic
        client = Anthropic(api_key=api_key, base_url=base_url)

        stream = client.messages.stream(
            model=_ARBITRATION_MODEL,
            max_tokens=200,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "你是一个任务复杂度判断助手。"
                        "用户输入是一个需要 AI 处理的任务描述。"
                        "请判断这个任务应该使用「简单模型」还是「复杂模型」处理。"
                        "简单模型：适合查看、列出、检查状态、简单问答等轻量任务。"
                        "复杂模型：适合代码实现、架构设计、调试优化、调研分析等重任务。"
                        "只回答一个单词：simple 或 complex。不要解释。"
                    ),
                },
                {
                    "role": "user",
                    "content": f"任务描述：{query}\n\n判断结果（simple/complex）：",
                },
            ],
        )
        ms = stream.__enter__()
        try:
            for event in ms:
                if event.type == "content_block_delta":
                    if hasattr(event.delta, "text") and event.delta.text:
                        text = event.delta.text.strip().lower()
                        if "simple" in text or "complex" in text:
                            raw = text
                            if "complex" in raw:
                                return Complexity.COMPLEX, f"MiniMax仲裁(流式): 判定为复杂任务 (raw={raw})"
                            else:
                                return Complexity.SIMPLE, f"MiniMax仲裁(流式): 判定为简单任务 (raw={raw})"
        finally:
            # 强制关闭 HTTP 连接（避免等完整思考过程）
            try:
                ms._raw_stream.response.close()
            except Exception:
                pass
            stream.__exit__(None, None, None)

        # 流式结束未命中
        return Complexity.SIMPLE, "MiniMax仲裁(流式): 输出不含判定词，默认 simple"

    except Exception as exc:
        logging.debug("[TaskComplexityRouter] MiniMax 仲裁失败: %s", exc)
        return Complexity.SIMPLE, f"MiniMax仲裁失败: {exc}"


def classify_query(query: str, tool_name: Optional[str] = None) -> RouteDecision:
    """
    核心分类函数。输入用户消息和可选的工具名，
    返回 RouteDecision（包含 complexity、suggested_model 等）。

    基于 50 条 LLM 标注样本重新设计的纯启发式规则，
    无额外 LLM 调用，零延迟开销。
    """
    start = time.monotonic()
    query = query.strip()

    # ── 工具名强制覆盖（不走规则）────────────────────────────────────────
    if tool_name:
        for pattern, complexity in TOOL_COMPLEXITY_MAP.items():
            if pattern.endswith("*"):
                prefix = pattern[:-1]
                if tool_name.startswith(prefix):
                    return _make_decision(
                        complexity,
                        primary_signal=f"tool:{tool_name}",
                        matched_rules=[f"TOOL_MAP:{pattern}"],
                        reason=f"工具 {tool_name} 映射为 {complexity.value}",
                        latency_ms=(time.monotonic() - start) * 1000,
                    )
            elif tool_name == pattern:
                return _make_decision(
                    complexity,
                    primary_signal=f"tool:{tool_name}",
                    matched_rules=[f"TOOL_MAP:{pattern}"],
                    reason=f"工具 {tool_name} 映射为 {complexity.value}",
                    latency_ms=(time.monotonic() - start) * 1000,
                )

    # ── 启发式规则判断 ──────────────────────────────────────────────────
    q_lower = query.lower()

    # 先检查 complex 规则（优先级高）
    for pat in _COMPLEX_PATTERNS:
        if re.search(pat, q_lower):
            return _make_decision(
                Complexity.COMPLEX,
                primary_signal="heuristic_complex",
                matched_rules=[pat],
                reason=f"命中复杂规则 [{pat}]",
                latency_ms=(time.monotonic() - start) * 1000,
            )

    # 再检查 simple 规则
    for pat in _SIMPLE_PATTERNS:
        if re.search(pat, q_lower):
            return _make_decision(
                Complexity.SIMPLE,
                primary_signal="heuristic_simple",
                matched_rules=[pat],
                reason=f"命中简单规则 [{pat}]",
                latency_ms=(time.monotonic() - start) * 1000,
            )

    # ── 无规则匹配 → 触发 LLM 仲裁 ──────────────────────────────────────
    # 启发式规则未命中，交给 MiniMax 高速版二次判断
    latency_ms = (time.monotonic() - start) * 1000
    complexity, reason = _arbitrate_with_minimax(query)
    return _make_decision(
        complexity,
        primary_signal="llm_arbiter",
        matched_rules=["ARBITRATION"],
        reason=reason,
        latency_ms=latency_ms,
    )


# ---------------------------------------------------------------------------
# Terminal / CLI 工具内容二次分类
# ---------------------------------------------------------------------------

def classify_terminal_content(command: str) -> RouteDecision:
    """
    对于 terminal 工具，仅凭工具名无法判断复杂度，
    需要分析命令内容。
    例：ls /tmp → simple；python -m pytest tests/ → complex
    """
    cmd_lower = command.lower().strip()

    # 明确复杂命令
    COMPLEX_COMMANDS = {
        "pytest", "python -m", "python3 -m", "cargo build", "cargo test",
        "make build", "make install", "npm install", "npm run", "yarn",
        "git commit", "git push", "git merge", "git rebase",
        "docker build", "docker run", "kubectl", "helm",
        "terraform apply", "ansible", "vagrant",
        "curl -X POST", "curl -X PUT", "wget", "ssh",
        "ffmpeg", "convert", "magick",
        "mongosh", "psql", "mysql",
        "bundle exec", "rake", "gradle",
    }

    SIMPLE_COMMANDS = {
        "ls", "ll", "la", "dir", "pwd", "cd",
        "cat", "head", "tail", "less", "more", "grep", "rg", "find",
        "echo", "printenv", "env", "which", "whoami", "date",
        "ps", "kill", "killall", "top", "htop",
        "df", "du", "free", "uptime",
        "curl -s", "curl -I", "wget -q",
        "git status", "git log", "git diff", "git show",
        "git branch", "git remote -v",
    }

    matched_complex = [c for c in COMPLEX_COMMANDS if c in cmd_lower]
    matched_simple = [c for c in SIMPLE_COMMANDS if c in cmd_lower]

    if matched_complex:
        return _make_decision(
            Complexity.COMPLEX,
            primary_signal=f"terminal_command:{matched_complex[0]}",
            matched_rules=[f"COMPLEX_CMD:{matched_complex}"],
            reason=f"终端命令推断为复杂（{matched_complex[0]}）",
            latency_ms=0.0,
        )
    if matched_simple:
        return _make_decision(
            Complexity.SIMPLE,
            primary_signal=f"terminal_command:{matched_simple[0]}",
            matched_rules=[f"SIMPLE_CMD:{matched_simple}"],
            reason=f"终端命令推断为简单（{matched_simple[0]}）",
            latency_ms=0.0,
        )

    # 其他命令默认 simple
    return _make_decision(
        Complexity.SIMPLE,
        primary_signal="terminal_default",
        matched_rules=["TERMINAL_DEFAULT_SIMPLE"],
        reason="终端命令无法明确分类，默认简单",
        latency_ms=0.0,
    )


# ---------------------------------------------------------------------------
# JSONL 持久化 — 线程安全直写
# ---------------------------------------------------------------------------

_ROUTING_LOG_PATH = Path(os.path.expanduser("~/.hermes/task_complexity_router.jsonl"))
_ROUTING_LOG_LOCK = threading.Lock()

# ---------------------------------------------------------------------------
# Per-query routing context — contextvars-based, safe across async boundaries
# ---------------------------------------------------------------------------

_routing_ctx: ContextVar[dict | None] = ContextVar(
    "task_complexity_router_ctx",
    default=None,
)

_CONFIG_LOCK = threading.Lock()
_CONFIG_FINGERPRINT: str | None = None


def get_current_routing_context() -> dict:
    """Return the current query routing context dict, or empty dict if none."""
    ctx = _routing_ctx.get()
    return dict(ctx) if isinstance(ctx, dict) else {}


def set_current_routing_context(ctx: dict) -> None:
    """Store the current query routing context for consumption by web providers."""
    _routing_ctx.set(dict(ctx))


def clear_current_routing_context() -> None:
    _routing_ctx.set(None)


def _compute_config_fingerprint(config: dict) -> str:
    try:
        return json.dumps(
            config.get("task_complexity_router", {}) or {},
            ensure_ascii=False,
            sort_keys=True,
        )
    except Exception:
        return ""


def ensure_router_config_loaded(config: Optional[dict] = None) -> None:
    """Load task router config once per effective config fingerprint."""
    global _CONFIG_FINGERPRINT

    explicit_config = config is not None
    if config is None:
        try:
            from hermes_cli.config import load_config

            config = load_config() or {}
        except Exception:
            config = {}

    section = {}
    if isinstance(config, dict):
        section = config.get("task_complexity_router", {}) or {}
    if not explicit_config and not section and _CONFIG_FINGERPRINT is not None:
        return

    fingerprint = _compute_config_fingerprint(config if isinstance(config, dict) else {})
    with _CONFIG_LOCK:
        if fingerprint == _CONFIG_FINGERPRINT:
            return
        configure_from_dict(config if isinstance(config, dict) else {})
        _CONFIG_FINGERPRINT = fingerprint


def _ensure_log_dir():
    _ROUTING_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)


def record_routing_event(
    query: str,
    decision: RouteDecision,
    tool_name: Optional[str] = None,
    override: bool = False,
) -> None:
    """线程安全地写入一条路由记录到 JSONL。"""
    event = RoutingEvent(
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime()),
        query=query[:500],  # 截断防止过大
        complexity=decision.complexity.value,
        primary_signal=decision.primary_signal,
        matched_rules=decision.matched_rules,
        suggested_model=decision.suggested_model,
        suggested_provider=decision.suggested_provider,
        reason=decision.reason,
        tool_name=tool_name,
        override=override,
        latency_ms=round(decision.latency_ms, 2),
    )
    try:
        _ensure_log_dir()
        with _ROUTING_LOG_LOCK:
            with open(_ROUTING_LOG_PATH, "a", encoding="utf-8") as f:
                f.write(event.to_json_line() + "\n")
                f.flush()
                os.fsync(f.fileno())
    except Exception:
        # 写入失败不阻塞主流程，仅记录
        pass


def _extract_user_override(
    query: str,
    user_override: Optional[str] = None,
) -> tuple[Optional[str], str]:
    """Resolve explicit override from caller or inline `@provider/model` prefix."""
    if user_override:
        return user_override.strip(), query
    match = _INLINE_MODEL_OVERRIDE_RE.match(query)
    if not match:
        return None, query
    override = (match.group("override") or "").strip()
    rest = (match.group("rest") or "").strip()
    return (override or None), rest


def _build_routing_context(route: EffectiveModelRoute) -> dict:
    return {
        "complexity": route.decision.complexity.value,
        "primary_signal": route.decision.primary_signal,
        "reason": route.decision.reason,
        "provider": route.provider,
        "model": route.model,
        "route_source": route.source,
        "user_override": route.user_override,
    }


def _classify_without_side_effects(
    query: str,
    tool_name: Optional[str] = None,
    tool_args: Optional[dict] = None,
) -> RouteDecision:
    decision = classify_query(query, tool_name)

    # Terminal 内容二次判断
    if tool_name == "terminal" and tool_args:
        cmd = tool_args.get("command", "")
        if cmd:
            terminal_decision = classify_terminal_content(cmd)
            # terminal 二次判断覆盖原有的 simple 结论
            if terminal_decision.complexity == Complexity.COMPLEX:
                decision = terminal_decision

    return decision


def resolve_effective_model_route(
    query: str,
    tool_name: Optional[str] = None,
    tool_args: Optional[dict] = None,
    user_override: Optional[str] = None,
    record: bool = True,
    config: Optional[dict] = None,
) -> EffectiveModelRoute:
    """Resolve the effective route for the current turn or tool invocation."""
    ensure_router_config_loaded(config)

    raw_query = query or ""
    resolved_override, normalized_query = _extract_user_override(raw_query, user_override)

    if resolved_override:
        if "/" in resolved_override:
            provider, model = resolved_override.split("/", 1)
        else:
            # Keep the current provider when callers only supply a model name.
            provider, model = "", resolved_override
        decision = RouteDecision(
            complexity=Complexity.UNKNOWN,
            primary_signal="user_override",
            matched_rules=[],
            suggested_model=model,
            suggested_provider=provider,
            reason=f"显式路由覆盖: {resolved_override}",
        )
        route = EffectiveModelRoute(
            provider=provider,
            model=model,
            decision=decision,
            source="user_override",
            normalized_query=normalized_query,
            user_override=resolved_override,
        )
    else:
        decision = _classify_without_side_effects(
            normalized_query,
            tool_name=tool_name,
            tool_args=tool_args,
        )
        route = EffectiveModelRoute(
            provider=decision.suggested_provider,
            model=decision.suggested_model,
            decision=decision,
            source="task_router",
            normalized_query=normalized_query,
            user_override=None,
        )

    if record:
        record_routing_event(
            normalized_query or raw_query,
            route.decision,
            tool_name,
            override=bool(route.user_override),
        )

    set_current_routing_context(_build_routing_context(route))
    return route


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def classify(
    query: str,
    tool_name: Optional[str] = None,
    tool_args: Optional[dict] = None,
    record: bool = True,
) -> RouteDecision:
    """
    主入口。对用户消息或工具调用进行复杂度分类。

    Args:
        query: 用户原始消息，或工具名+参数的描述
        tool_name: 工具名（可选，有则优先查 TOOL_COMPLEXITY_MAP）
        tool_args: 工具参数（可选，用于 terminal 等工具的二次判断）
        record: 是否写入 JSONL 日志（默认 True）

    Returns:
        RouteDecision 对象
    """
    ensure_router_config_loaded()
    decision = _classify_without_side_effects(query, tool_name, tool_args)

    if record:
        record_routing_event(query, decision, tool_name)

    # Publish routing context for web providers to read
    set_current_routing_context({
        "complexity": decision.complexity.value,
        "primary_signal": decision.primary_signal,
        "reason": decision.reason,
        "provider": decision.suggested_provider,
        "model": decision.suggested_model,
        "route_source": "task_router",
        "user_override": None,
    })

    return decision


def get_model_for_task(
    query: str,
    tool_name: Optional[str] = None,
    tool_args: Optional[dict] = None,
    user_override: Optional[str] = None,  # 用户手动指定的模型
    config: Optional[dict] = None,
) -> tuple[str, str, RouteDecision]:
    """
    返回 (provider, model, decision)。

    如果 user_override 存在，直接返回用户指定的模型，跳过分层逻辑。
    """
    route = resolve_effective_model_route(
        query,
        tool_name=tool_name,
        tool_args=tool_args,
        user_override=user_override,
        record=True,
        config=config,
    )
    return route.provider, route.model, route.decision


# ---------------------------------------------------------------------------
# 配置加载（支持从 config.yaml 覆盖 MODEL_TIERS）
# ---------------------------------------------------------------------------

_CONFIGURED = False
_CUSTOM_TIERS: dict = {}


def configure_from_dict(config: dict) -> None:
    """从 dict（如从 config.yaml 读取的）加载自定义配置。"""
    global _CONFIGURED, _CUSTOM_TIERS
    tier_config = config.get("task_complexity_router", {})
    global MODEL_TIERS
    MODEL_TIERS = deepcopy(_DEFAULT_MODEL_TIERS)
    _CUSTOM_TIERS = tier_config or {}
    if not tier_config:
        _CONFIGURED = False
        return

    # 允许覆盖 simple tier 模型
    simple_cfg = tier_config.get("simple_tier", {})
    if simple_cfg:
        MODEL_TIERS[Complexity.SIMPLE]["default"] = {
            "provider": simple_cfg.get("provider", "minimax"),
            "model": simple_cfg.get("model", "MiniMax-M2.7-highspeed"),
        }
        if isinstance(simple_cfg.get("fallback"), list):
            MODEL_TIERS[Complexity.SIMPLE]["fallback"] = deepcopy(simple_cfg["fallback"])

    # 允许覆盖 complex tier 模型
    complex_cfg = tier_config.get("complex_tier", {})
    if complex_cfg:
        MODEL_TIERS[Complexity.COMPLEX]["default"] = {
            "provider": complex_cfg.get("provider", "kimi"),
            "model": complex_cfg.get("model", "kimi-k2.6"),
        }
        if isinstance(complex_cfg.get("fallback"), list):
            MODEL_TIERS[Complexity.COMPLEX]["fallback"] = deepcopy(complex_cfg["fallback"])

    _CONFIGURED = True


def _apply_model_route(
    agent,
    provider: str,
    model: str,
    *,
    reason: str,
    source: str,
) -> bool:
    """Apply a resolved route to the agent and persist it as the new primary runtime."""
    new_provider = (provider or getattr(agent, "provider", "") or "").strip()
    new_model = (model or getattr(agent, "model", "") or "").strip()
    if not new_provider or not new_model:
        return False

    # 跳过同 provider+model 的切换
    current_provider = (getattr(agent, "provider", "") or "").strip().lower()
    current_model = (getattr(agent, "model", "") or "").strip().lower()
    if current_provider == new_provider.lower() and current_model == new_model.lower():
        return False

    try:
        from agent.auxiliary_client import resolve_provider_client

        new_client, resolved_model = resolve_provider_client(
            new_provider, model=new_model, raw_codex=False
        )
        if new_client is None:
            logging.warning(
                "[TaskComplexityRouter] 切换模型失败: provider=%s model=%s 不可用",
                new_provider, new_model,
            )
            return False

        # 更新核心属性
        agent.model = resolved_model or new_model
        agent.provider = new_provider

        # 获取 base_url 和 api_key
        base_url = getattr(new_client, "_base_url", None) or getattr(new_client, "base_url", None) or ""
        agent.base_url = str(base_url)

        # api_key 可能存在 client 或其 config 中
        api_key = ""
        if hasattr(new_client, "api_key") and new_client.api_key:
            api_key = new_client.api_key
        elif hasattr(new_client, "_api_key"):
            api_key = new_client._api_key
        elif hasattr(new_client, "api_key") and isinstance(new_client.api_key, str):
            api_key = new_client.api_key
        agent.api_key = api_key

        # 更新 client_kwargs
        new_client_kwargs = {}
        if base_url:
            new_client_kwargs["base_url"] = str(base_url)
        if api_key:
            new_client_kwargs["api_key"] = api_key
        agent._client_kwargs = new_client_kwargs

        # 清除 transport 缓存，强制重建
        if hasattr(agent, "_transport_cache"):
            agent._transport_cache.clear()

        # 清除 API client 缓存，触发重新构造
        agent.client = None

        # Update _primary_runtime so fallback restore and cached-agent reuse
        # treat this route as the current preferred primary runtime.
        agent._primary_runtime = {
            "model": agent.model,
            "provider": agent.provider,
            "base_url": agent.base_url,
            "api_mode": getattr(agent, "api_mode", "chat_completions"),
            "api_key": agent.api_key,
            "client_kwargs": dict(agent._client_kwargs),
            "use_prompt_caching": getattr(agent, "_use_prompt_caching", False),
            "use_native_cache_layout": getattr(agent, "_use_native_cache_layout", False),
        }
        if getattr(agent, "api_mode", "") == "anthropic_messages":
            agent._primary_runtime.update({
                "anthropic_api_key": getattr(agent, "_anthropic_api_key", ""),
                "anthropic_base_url": getattr(agent, "_anthropic_base_url", ""),
                "is_anthropic_oauth": getattr(agent, "_is_anthropic_oauth", False),
            })
        agent._fallback_activated = False
        agent._fallback_index = 0
        # Invalidate cached system prompt so it rebuilds with new model next turn
        agent._cached_system_prompt = None

        logging.info(
            "[TaskComplexityRouter] 模型切换: source=%s provider=%s model=%s reason=%s",
            source, new_provider, agent.model, reason,
        )
        return True

    except Exception as exc:
        logging.warning(
            "[TaskComplexityRouter] 切换模型异常: provider=%s model=%s error=%s",
            new_provider, new_model, exc,
        )
        return False


def activate_effective_route(agent, route: EffectiveModelRoute) -> bool:
    """Apply the resolved effective route to the agent."""
    return _apply_model_route(
        agent,
        route.provider or getattr(agent, "provider", ""),
        route.model or getattr(agent, "model", ""),
        reason=route.decision.reason,
        source=route.source,
    )


def activate_model_tier(agent, decision: RouteDecision) -> bool:
    """
    将 agent 切换到指定 complexity tier 的模型。

    流程：
    1. 从 MODEL_TIERS 取出对应 tier 的默认模型
    2. 通过 resolve_provider_client 构造新 client（自动处理 key/endpoint）
    3. 更新 agent.model / agent.provider / agent.base_url / agent.api_key / agent._client_kwargs
    4. 清除 _transport_cache 强制重建

    注意：
    - 会同步更新 `_primary_runtime`，让 fallback 恢复逻辑使用新的首选主路由
    - 调用方确保 decision 不是 UNKNOWN / FORCE_* 类型
    """
    if decision.complexity in (Complexity.UNKNOWN, Complexity.FORCE_SIMPLE, Complexity.FORCE_COMPLEX):
        return False

    return _apply_model_route(
        agent,
        decision.suggested_provider,
        decision.suggested_model,
        reason=decision.reason,
        source=decision.primary_signal,
    )


# ---------------------------------------------------------------------------
# CLI 快速测试
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_queries = [
        "查看配置",
        "帮我 list 一下当前目录",
        "如何实现 LLM 路由功能？",
        "写一个 Python 函数来解析 JSON",
        "调研 adaptive rag 的实现方案",
        "ls /tmp",
        "帮我读一下 config.yaml 的内容",
        "用 Claude 实现一个决策树分类器",
        "优化一下这段代码的性能",
        "什么是 HTTP",
    ]

    print("\n=== Task Complexity Router 测试 ===\n")
    for q in test_queries:
        d = classify(q)
        print(f"[{d.complexity.value:>12}] {d.suggested_provider}/{d.suggested_model} | {d.reason}")
        print(f"            query: {q}")
        print()
