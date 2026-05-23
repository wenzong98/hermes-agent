"""
Task Complexity Router — 任务复杂度路由

在 LLM 调用前根据启发式规则判断任务复杂度，
自动选择合适的模型 tier：

- simple  → 轻量快速模型（MiniMax-M2.7 高速版 / deepseek-v4-flash）
- complex → 高质量模型（kimi-k2.6 / claude-sonnet-4 / GPT-4）

设计原则：
1. 零额外 LLM 调用 — 纯启发式判断，无 latency 开销
2. 可干预 — 用户可通过 @mention 指定模型，覆盖自动路由
3. 工具级控制 — 特定工具强制走 complex 或 simple tier
4. JSONL 追踪 — 每次路由决策写入日志，便于分析调优

集成点：
- agent/conversation_loop.py  ：用户消息入口分类
- model_tools.py/handle_function_call：工具执行前拦截
- agent/tool_executor.py     ：工具执行复杂度记录
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
import threading
from dataclasses import dataclass, field, asdict
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
    confidence: float          # 0.0–1.0
    primary_signal: str       # 触发决策的主要信号
    matched_rules: list[str]  # 命中的规则列表
    suggested_model: str      # 建议模型
    suggested_provider: str   # 建议 provider
    reason: str               # 人类可读原因
    latency_ms: float = 0.0


@dataclass
class RoutingEvent:
    """写入 JSONL 的每次路由记录"""
    timestamp: str
    query: str
    complexity: str
    confidence: float
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
# Complexity signals — 启发式规则
# ---------------------------------------------------------------------------

# 强信号：出现这些词 → complex（置信度高）
_COMPLEX_KEYWORDS = {
    # 规划和架构
    "规划", "设计", "架构", "架构设计", "方案", "解决方案",
    "plan", "design", "architect", "architecture", "solution",
    # 代码开发
    "实现", "开发", "写代码", "编程", "代码", "debug", "调试",
    "implement", "develop", "coding", "code", "program",
    "refactor", "重构", "优化性能", "optimize",
    "写一个", "写段", "写个", "帮我写", "create function",
    "class ", "def ", "function", "algorithm",
    "bug", "error", "exception", "traceback",
    # 研究和分析
    "调研", "研究", "分析", "对比", "评估", "调研报告",
    "research", "analyze", "analyse", "compare", "evaluate",
    "investigate", "调查",
    # 复杂操作
    "部署", "上线", "迁移", "集成", "对接",
    "deploy", "deploy", "migration", "integration",
}

# 弱信号：出现这些词 → simple（但整体偏 complex 时会覆盖）
_SIMPLE_KEYWORDS = {
    # 读取状态
    "查看", "显示", "列出", "获取", "读取", "检查", "状态",
    "show", "list", "get", "read", "check", "status", "display",
    "what is", "where is", "who is", "how many", "多少",
    # 简单问答题
    "什么是", "怎么", "如何", "why is", "when did",
    "解释", "说明", "讲讲",
    # 配置相关
    "配置", "config", "setting", "配置信息",
}

# Token count 阈值
_TOKEN_COUNT_SIMPLE_MAX = 150      # 中文约 100 字符，英文约 150 词
_TOKEN_COUNT_LIKELY_COMPLEX_MIN = 300  # 超过这个几乎肯定是复杂任务

# 低置信度仲裁阈值：低于此值时调用 MiniMax 高速版二次判断
_ARBITRATION_CONFIDENCE_THRESHOLD = 0.72
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
MODEL_TIERS = {
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
    confidence: float,
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
        confidence=confidence,
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

    返回 (complexity, reason)。
    如果仲裁失败，返回 (Complexity.SIMPLE, "仲裁失败，默认 simple")。
    """
    try:
        from agent.auxiliary_client import call_llm

        arb_messages = [
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
        ]

        resp = call_llm(
            task="task_complexity_arbitration",
            provider=_ARBITRATION_PROVIDER,
            model=_ARBITRATION_MODEL,
            messages=arb_messages,
            temperature=0.0,
            max_tokens=10,
            timeout=10.0,
        )
        raw = (resp.choices[0].message.content or "").strip().lower()

        if "complex" in raw:
            return Complexity.COMPLEX, f"MiniMax仲裁: 判定为复杂任务 (raw={raw})"
        elif "simple" in raw:
            return Complexity.SIMPLE, f"MiniMax仲裁: 判定为简单任务 (raw={raw})"
        else:
            # 无法解析，默认 simple
            return Complexity.SIMPLE, f"MiniMax仲裁: 输出不可解析 (raw={raw})，默认 simple"

    except Exception as exc:
        logging.debug("[TaskComplexityRouter] MiniMax 仲裁失败: %s", exc)
        return Complexity.SIMPLE, f"MiniMax仲裁失败: {exc}"


def classify_query(query: str, tool_name: Optional[str] = None) -> RouteDecision:
    """
    核心分类函数。输入用户消息和可选的工具名，
    返回 RouteDecision（包含 complexity、suggested_model 等）。

    流程：
    1. 先走纯启发式分类（零 LLM 调用）
    2. 如果置信度 < _ARBITRATION_CONFIDENCE_THRESHOLD，调用 MiniMax 高速版仲裁
    3. 返回最终决策
    """
    start = time.monotonic()
    query = query.strip()
    tokens = estimate_tokens(query)
    query_lower = query.lower()

    # ── 1. 工具名强制覆盖 ────────────────────────────────────────────────
    if tool_name:
        for pattern, complexity in TOOL_COMPLEXITY_MAP.items():
            if pattern.endswith("*"):
                prefix = pattern[:-1]
                if tool_name.startswith(prefix):
                    return _make_decision(
                        complexity,
                        confidence=1.0,
                        primary_signal=f"tool:{tool_name}",
                        matched_rules=[f"TOOL_MAP:{pattern}"],
                        reason=f"工具 {tool_name} 映射为 {complexity.value}",
                        latency_ms=time.monotonic() - start,
                    )
            elif tool_name == pattern:
                return _make_decision(
                    complexity,
                    confidence=1.0,
                    primary_signal=f"tool:{tool_name}",
                    matched_rules=[f"TOOL_MAP:{pattern}"],
                    reason=f"工具 {tool_name} 映射为 {complexity.value}",
                    latency_ms=time.monotonic() - start,
                )

    # ── 2. Token count 强信号 ─────────────────────────────────────────────
    if tokens >= _TOKEN_COUNT_LIKELY_COMPLEX_MIN:
        return _make_decision(
            Complexity.COMPLEX,
            confidence=0.95,
            primary_signal=f"token_count:{tokens}",
            matched_rules=[f"COMPLEX_BY_TOKENS({tokens}>={_TOKEN_COUNT_LIKELY_COMPLEX_MIN})"],
            reason=f"输入过长（≈{tokens} tokens），推断为复杂任务",
            latency_ms=time.monotonic() - start,
        )

    # ── 3.5. 短输入特殊处理：防止「帮我+写/创建」被误判 ─────────────────────
    if tokens <= _TOKEN_COUNT_SIMPLE_MAX:
        # 跳过以 # 前缀开头的查询（话题标签/提及等，不应作为复杂度判断依据）
        query_stripped = query.lstrip()
        if query_stripped.startswith("#"):
            # 标签类输入：查询标签内容本身属于 simple
            return _make_decision(
                Complexity.SIMPLE,
                confidence=0.82,
                primary_signal="hashtag_input",
                matched_rules=["HASHTAG_PREFIX"],
                reason="标签类输入（#开头），判定为简单任务",
                latency_ms=time.monotonic() - start,
            )

        # 「帮我 + 写/创建类动词」→ 强制 complex
        writing_intent = any(
            f"{prefix}{action}" in query
            for prefix in ["帮我", "请帮我", "我想"]
            for action in ["写", "写个", "创建一个", "写段", "写代码", "写个"]
        )
        if writing_intent:
            matched_complex = [kw for kw in _COMPLEX_KEYWORDS if kw in query_lower]
            return _make_decision(
                Complexity.COMPLEX,
                confidence=0.85,
                primary_signal="short_input_writing_intent",
                matched_rules=[f"WRITING_INTENT:{matched_complex}"],
                reason="短输入含「帮我+写/创建」意图，判定为复杂任务",
                latency_ms=time.monotonic() - start,
            )

        # 短输入先倾向 simple，再看关键词
        simple_kw_count = sum(1 for kw in _SIMPLE_KEYWORDS if kw in query_lower)
        complex_kw_count = sum(1 for kw in _COMPLEX_KEYWORDS if kw in query_lower)
        if simple_kw_count > complex_kw_count:
            return _make_decision(
                Complexity.SIMPLE,
                confidence=0.75,
                primary_signal=f"token_count:{tokens}",
                matched_rules=[f"SIMPLE_BY_TOKENS_AND_KEYWORDS(simple={simple_kw_count},complex={complex_kw_count})"],
                reason=f"短输入（≈{tokens} tokens），简单关键词多",
                latency_ms=time.monotonic() - start,
            )

    # ── 3. 关键词计数 ────────────────────────────────────────────────────
    matched_simple = [kw for kw in _SIMPLE_KEYWORDS if kw in query_lower]
    matched_complex = [kw for kw in _COMPLEX_KEYWORDS if kw in query_lower]

    complex_score = len(matched_complex)
    simple_score = len(matched_simple)

    # 权重：complex 关键词命中权重更高
    # 1 个 complex 关键词 ≈ 2 个 simple 关键词
    weighted_complex = complex_score * 2.0
    weighted_simple = simple_score * 1.0

    if weighted_complex > weighted_simple:
        confidence = min(0.5 + (weighted_complex - weighted_simple) * 0.1, 0.95)
        return _make_decision(
            Complexity.COMPLEX,
            confidence=confidence,
            primary_signal="keyword_analysis",
            matched_rules=[f"COMPLEX_KEYWORDS:{matched_complex}", f"SIMPLE_KEYWORDS:{matched_simple}"],
            reason=f"复杂关键词命中（{'/'.join(matched_complex[:3])}），推断为复杂任务",
            latency_ms=time.monotonic() - start,
        )
    elif weighted_simple > weighted_complex:
        confidence = min(0.5 + (weighted_simple - weighted_complex) * 0.15, 0.85)
        return _make_decision(
            Complexity.SIMPLE,
            confidence=confidence,
            primary_signal="keyword_analysis",
            matched_rules=[f"SIMPLE_KEYWORDS:{matched_simple}", f"COMPLEX_KEYWORDS:{matched_complex}"],
            reason=f"简单关键词命中（{'/'.join(matched_simple[:3])}），推断为简单任务",
            latency_ms=time.monotonic() - start,
        )

    # ── 4. 默认策略 ────────────────────────────────────────────────────────
    # 两者差不多或都没命中，按 simple 处理（节省成本）
    preliminary = _make_decision(
        Complexity.SIMPLE,
        confidence=0.6,
        primary_signal="default_fallback",
        matched_rules=["DEFAULT_SIMPLE_FALLBACK"],
        reason="无法明确分类，默认路由到简单模型（可手动指定覆盖）",
        latency_ms=time.monotonic() - start,
    )

    # ── 5. 低置信度仲裁 ────────────────────────────────────────────────────
    # 当启发式分类置信度不足时，调用 MiniMax 高速版二次判断
    if preliminary.confidence < _ARBITRATION_CONFIDENCE_THRESHOLD:
        arb_complexity, arb_reason = _arbitrate_with_minimax(query)
        arb_confidence = 0.82  # 仲裁后的置信度固定为 0.82
        return _make_decision(
            arb_complexity,
            confidence=arb_confidence,
            primary_signal="minimax_arbitration",
            matched_rules=["ARBITRATION_TRIGGERED"] + preliminary.matched_rules,
            reason=arb_reason,
            latency_ms=time.monotonic() - start,
        )

    return preliminary


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
            confidence=0.9,
            primary_signal=f"terminal_command:{matched_complex[0]}",
            matched_rules=[f"COMPLEX_CMD:{matched_complex}"],
            reason=f"终端命令推断为复杂（{matched_complex[0]}）",
            latency_ms=0.0,
        )
    if matched_simple:
        return _make_decision(
            Complexity.SIMPLE,
            confidence=0.85,
            primary_signal=f"terminal_command:{matched_simple[0]}",
            matched_rules=[f"SIMPLE_CMD:{matched_simple}"],
            reason=f"终端命令推断为简单（{matched_simple[0]}）",
            latency_ms=0.0,
        )

    # 其他命令默认 simple
    return _make_decision(
        Complexity.SIMPLE,
        confidence=0.6,
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
# Per-query routing context — thread-local, consumed by web providers
# ---------------------------------------------------------------------------

_routing_ctx = threading.local()


def get_current_routing_context() -> dict:
    """Return the current query routing context dict, or empty dict if none."""
    ctx = getattr(_routing_ctx, "value", None)
    return ctx if ctx is not None else {}


def set_current_routing_context(ctx: dict) -> None:
    """Store the current query routing context for consumption by web providers."""
    _routing_ctx.value = ctx


def clear_current_routing_context() -> None:
    _routing_ctx.value = None


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
        confidence=decision.confidence,
        primary_signal=decision.primary_signal,
        matched_rules=decision.matched_rules,
        suggested_model=decision.suggested_model,
        suggested_provider=decision.suggested_provider,
        reason=decision.reason,
        tool_name=tool_name,
        override=override,
        latency_ms=round(decision.latency_ms * 1000, 2),
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
    decision = classify_query(query, tool_name)

    # Terminal 内容二次判断
    if tool_name == "terminal" and tool_args:
        cmd = tool_args.get("command", "")
        if cmd:
            terminal_decision = classify_terminal_content(cmd)
            # terminal 二次判断覆盖原有的 simple 结论
            if terminal_decision.complexity == Complexity.COMPLEX:
                decision = terminal_decision

    if record:
        record_routing_event(query, decision, tool_name)

    # Publish routing context for web providers to read
    set_current_routing_context({
        "complexity": decision.complexity.value,
        "confidence": decision.confidence,
        "primary_signal": decision.primary_signal,
        "reason": decision.reason,
    })

    return decision


def get_model_for_task(
    query: str,
    tool_name: Optional[str] = None,
    tool_args: Optional[dict] = None,
    user_override: Optional[str] = None,  # 用户手动指定的模型
) -> tuple[str, str, RouteDecision]:
    """
    返回 (provider, model, decision)。

    如果 user_override 存在，直接返回用户指定的模型，跳过分层逻辑。
    """
    if user_override:
        # 格式支持： "kimi/kimi-k2.6" 或 "kimi-k2.6"（自动补 provider）
        if "/" in user_override:
            provider, model = user_override.split("/", 1)
        else:
            provider, model = "user_specified", user_override
        decision = RouteDecision(
            complexity=Complexity.UNKNOWN,
            confidence=1.0,
            primary_signal="user_override",
            matched_rules=[],
            suggested_model=model,
            suggested_provider=provider,
            reason=f"用户手动指定: {user_override}",
        )
        return provider, model, decision

    decision = classify(query, tool_name, tool_args)
    return decision.suggested_provider, decision.suggested_model, decision


# ---------------------------------------------------------------------------
# 配置加载（支持从 config.yaml 覆盖 MODEL_TIERS）
# ---------------------------------------------------------------------------

_CONFIGURED = False
_CUSTOM_TIERS: dict = {}


def configure_from_dict(config: dict) -> None:
    """从 dict（如从 config.yaml 读取的）加载自定义配置。"""
    global _CONFIGURED, _CUSTOM_TIERS
    tier_config = config.get("task_complexity_router", {})
    if not tier_config:
        return

    global MODEL_TIERS
    _CUSTOM_TIERS = tier_config

    # 允许覆盖 simple tier 模型
    simple_cfg = tier_config.get("simple_tier", {})
    if simple_cfg:
        MODEL_TIERS[Complexity.SIMPLE]["default"] = {
            "provider": simple_cfg.get("provider", "minimax"),
            "model": simple_cfg.get("model", "MiniMax-M2.7-highspeed"),
        }

    # 允许覆盖 complex tier 模型
    complex_cfg = tier_config.get("complex_tier", {})
    if complex_cfg:
        MODEL_TIERS[Complexity.COMPLEX]["default"] = {
            "provider": complex_cfg.get("provider", "kimi"),
            "model": complex_cfg.get("model", "kimi-k2.6"),
        }

    _CONFIGURED = True


def activate_model_tier(agent, decision: RouteDecision) -> bool:
    """
    将 agent 切换到指定 complexity tier 的模型。

    流程：
    1. 从 MODEL_TIERS 取出对应 tier 的默认模型
    2. 通过 resolve_provider_client 构造新 client（自动处理 key/endpoint）
    3. 更新 agent.model / agent.provider / agent.base_url / agent.api_key / agent._client_kwargs
    4. 清除 _transport_cache 强制重建

    注意：
    - 不修改 _primary_runtime（那是 fallback/自动降级用的，不是我们手动切换）
    - 调用方确保 decision 不是 UNKNOWN / FORCE_* 类型
    """
    if decision.complexity in (Complexity.UNKNOWN, Complexity.FORCE_SIMPLE, Complexity.FORCE_COMPLEX):
        return False

    tier = MODEL_TIERS[decision.complexity]
    new_provider = decision.suggested_provider
    new_model = decision.suggested_model

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

        # Update _primary_runtime so automatic fallback restoration doesn't
        # undo a deliberate task-complexity routing decision mid-session.
        # Match the shape of what create_openai_client saves (agent_runtime_helpers).
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
        # Invalidate cached system prompt so it rebuilds with new model next turn
        agent._cached_system_prompt = None

        logging.info(
            "[TaskComplexityRouter] 模型切换: complexity=%s provider=%s model=%s reason=%s",
            decision.complexity.value, new_provider, agent.model, decision.reason,
        )
        return True

    except Exception as exc:
        logging.warning(
            "[TaskComplexityRouter] 切换模型异常: provider=%s model=%s error=%s",
            new_provider, new_model, exc,
        )
        return False


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
        print(f"[{d.complexity.value:>12}] {d.confidence:.2f} | {d.suggested_provider}/{d.suggested_model} | {d.reason}")
        print(f"            query: {q}")
        print()
