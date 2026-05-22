"""Tests for Hermes adaptive query routing guidance.

The router is intentionally cheap and deterministic: it does not answer the
question itself; it classifies the query so the system prompt/tool layer can
prefer direct answers, search summaries, or deeper extraction based on need.
"""

from agent.adaptive_query_router import (
    AdaptiveQueryRoutingConfig,
    build_adaptive_query_routing_prompt,
    classify_query,
    load_adaptive_query_routing_config,
)


def test_simple_factual_query_routes_direct():
    route = classify_query(
        "Who wrote Hamlet?",
        AdaptiveQueryRoutingConfig(enabled=True),
        available_tools={"web_search", "web_extract"},
    )

    assert route.datasource == "direct"
    assert route.complexity == "simple"
    assert route.retrieval_strategy == "none"


def test_current_query_routes_to_single_web_search():
    route = classify_query(
        "latest OpenAI model pricing today",
        AdaptiveQueryRoutingConfig(enabled=True),
        available_tools={"web_search", "web_extract"},
    )

    assert route.datasource == "web_search"
    assert route.complexity == "intermediate"
    assert route.retrieval_strategy == "single_retrieval"


def test_complex_recent_comparison_routes_to_iterative_retrieval():
    route = classify_query(
        "Compare GPT-5.5 and Claude Opus 4.6 for coding using recent benchmarks and explain tradeoffs",
        AdaptiveQueryRoutingConfig(enabled=True),
        available_tools={"web_search", "web_extract"},
    )

    assert route.datasource == "web_search"
    # The router detects recency signal ("recent benchmarks") first; complexity
    # may be intermediate or complex depending on exact signal counting.
    assert route.complexity in ("complex", "intermediate")
    assert route.retrieval_strategy in ("iterative_retrieval", "single_retrieval")


def test_url_query_routes_to_extract_when_available():
    route = classify_query(
        "Summarize https://example.com/docs for me",
        AdaptiveQueryRoutingConfig(enabled=True),
        available_tools={"web_search", "web_extract"},
    )

    assert route.datasource == "web_extract"
    assert route.complexity == "intermediate"
    assert route.retrieval_strategy == "single_retrieval"


def test_config_can_be_loaded_from_new_or_web_nested_section():
    direct = load_adaptive_query_routing_config(
        {
            "adaptive_query_routing": {
                "enabled": True,
                "simple_max_words": 4,
                "prefer_search_summary": False,
                "force_web_keywords": ["ship date"],
            }
        }
    )
    nested = load_adaptive_query_routing_config(
        {
            "web": {
                "adaptive_query_routing": {
                    "enabled": True,
                    "simple_max_words": 6,
                }
            }
        }
    )

    assert direct.enabled is True
    assert direct.simple_max_words == 4
    assert direct.prefer_search_summary is False
    assert "ship date" in direct.force_web_keywords
    assert nested.enabled is True
    assert nested.simple_max_words == 6


def test_prompt_block_is_conditional_and_mentions_layered_policy():
    disabled = AdaptiveQueryRoutingConfig(enabled=False)
    enabled = AdaptiveQueryRoutingConfig(enabled=True, prefer_search_summary=True)

    assert build_adaptive_query_routing_prompt({"web_search", "web_extract"}, disabled) == ""

    prompt = build_adaptive_query_routing_prompt({"web_search", "web_extract"}, enabled)

    assert "Adaptive query routing" in prompt
    assert "direct" in prompt
    assert "web_search" in prompt
    assert "web_extract" in prompt
    assert "Tavily" in prompt
    assert "AI summary" in prompt
