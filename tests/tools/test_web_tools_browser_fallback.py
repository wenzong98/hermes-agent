import json
from unittest.mock import AsyncMock

import pytest


class _FakeAsyncExtractProvider:
    name = "fake-extract"
    display_name = "Fake Extract"

    def __init__(self, results):
        self._results = results

    def supports_extract(self):
        return True

    async def extract(self, urls, format=None):
        return list(self._results)


@pytest.mark.asyncio
async def test_web_extract_does_not_fallback_for_normal_content(monkeypatch):
    from tools import web_tools

    provider = _FakeAsyncExtractProvider(
        [
            {
                "url": "https://example.com/ok",
                "title": "Normal page",
                "content": "A" * 300,
                "error": None,
            }
        ]
    )
    browser_fallback = AsyncMock()

    monkeypatch.setattr(web_tools, "_get_extract_backend", lambda: "fake")
    monkeypatch.setattr(web_tools, "is_safe_url", lambda url: True)
    monkeypatch.setattr(web_tools, "check_auxiliary_model", lambda: False)
    monkeypatch.setattr(web_tools, "_extract_with_browser", browser_fallback)

    import agent.web_search_registry as registry

    monkeypatch.setattr(registry, "get_provider", lambda backend: provider)
    monkeypatch.setattr(registry, "get_active_extract_provider", lambda: provider)

    result = json.loads(
        await web_tools.web_extract_tool(
            ["https://example.com/ok"], use_llm_processing=False
        )
    )

    assert result["results"][0]["content"] == "A" * 300
    assert "extraction_method" not in result["results"][0]
    browser_fallback.assert_not_awaited()


@pytest.mark.asyncio
async def test_web_extract_falls_back_to_browser_for_dynamic_placeholder(monkeypatch):
    from tools import web_tools

    provider = _FakeAsyncExtractProvider(
        [
            {
                "url": "https://example.com/dynamic",
                "title": "Dynamic shell",
                "content": "Please enable JavaScript to continue.",
                "error": None,
            }
        ]
    )
    browser_fallback = AsyncMock(
        return_value={
            "url": "https://example.com/dynamic",
            "title": "Rendered page",
            "content": "Rendered browser content " * 20,
            "raw_content": "Rendered browser content " * 20,
            "error": None,
        }
    )

    monkeypatch.setattr(web_tools, "_get_extract_backend", lambda: "fake")
    monkeypatch.setattr(web_tools, "is_safe_url", lambda url: True)
    monkeypatch.setattr(web_tools, "check_auxiliary_model", lambda: False)
    monkeypatch.setattr(web_tools, "_extract_with_browser", browser_fallback)

    import agent.web_search_registry as registry

    monkeypatch.setattr(registry, "get_provider", lambda backend: provider)
    monkeypatch.setattr(registry, "get_active_extract_provider", lambda: provider)

    result = json.loads(
        await web_tools.web_extract_tool(
            ["https://example.com/dynamic"], use_llm_processing=False
        )
    )

    assert result["results"][0]["title"] == "Rendered page"
    assert result["results"][0]["content"].startswith("Rendered browser content")
    assert result["results"][0]["extraction_method"] == "browser"
    assert result["results"][0]["fallback_reason"] == "dynamic_placeholder"
    browser_fallback.assert_awaited_once_with("https://example.com/dynamic")


@pytest.mark.asyncio
async def test_web_extract_only_falls_back_for_failed_items_in_batch(monkeypatch):
    from tools import web_tools

    provider = _FakeAsyncExtractProvider(
        [
            {
                "url": "https://example.com/ok",
                "title": "Normal page",
                "content": "B" * 300,
                "error": None,
            },
            {
                "url": "https://example.com/bad",
                "title": "Broken page",
                "content": "",
                "error": "provider timed out",
            },
        ]
    )
    browser_fallback = AsyncMock(
        return_value={
            "url": "https://example.com/bad",
            "title": "Recovered page",
            "content": "Recovered browser content " * 20,
            "raw_content": "Recovered browser content " * 20,
            "error": None,
        }
    )

    monkeypatch.setattr(web_tools, "_get_extract_backend", lambda: "fake")
    monkeypatch.setattr(web_tools, "is_safe_url", lambda url: True)
    monkeypatch.setattr(web_tools, "check_auxiliary_model", lambda: False)
    monkeypatch.setattr(web_tools, "_extract_with_browser", browser_fallback)

    import agent.web_search_registry as registry

    monkeypatch.setattr(registry, "get_provider", lambda backend: provider)
    monkeypatch.setattr(registry, "get_active_extract_provider", lambda: provider)

    result = json.loads(
        await web_tools.web_extract_tool(
            ["https://example.com/ok", "https://example.com/bad"],
            use_llm_processing=False,
        )
    )

    assert result["results"][0]["url"] == "https://example.com/ok"
    assert result["results"][0]["content"] == "B" * 300
    assert "extraction_method" not in result["results"][0]

    assert result["results"][1]["url"] == "https://example.com/bad"
    assert result["results"][1]["title"] == "Recovered page"
    assert result["results"][1]["extraction_method"] == "browser"
    assert result["results"][1]["fallback_reason"] == "extract_error"
    assert browser_fallback.await_count == 1
    browser_fallback.assert_awaited_once_with("https://example.com/bad")
