"""Regression tests for Hermes' local mem0 mode.

These cover the historical bytedance setup:
- memory.provider: mem0
- ~/.hermes/mem0.json contains local_mode + chroma_path
- MiniMax API key comes from env
- mem0 should use Memory.from_config(...) + local Chroma, not MemoryClient cloud mode
"""

import json
import sys
from types import ModuleType, SimpleNamespace

import pytest

import plugins.memory.mem0 as mem0_plugin
from plugins.memory.mem0 import Mem0MemoryProvider


class FakeCollection:
    def __init__(self):
        self.get_calls = []
        self.query_calls = []

    def get(self, *, where=None, include=None):
        self.get_calls.append({"where": where, "include": include})
        return {"metadatas": [{"data": "local profile hit", "user_id": where.get("user_id") if where else None}]}

    def query(self, *, query_embeddings, n_results, where=None, include=None):
        self.query_calls.append(
            {
                "query_embeddings": query_embeddings,
                "n_results": n_results,
                "where": where,
                "include": include,
            }
        )
        return {
            "metadatas": [[{"data": "local search hit", "user_id": where.get("user_id") if where else None}]],
            "distances": [[0.1]],
        }


class FakeEmbeddingModel:
    def __init__(self):
        self.calls = []

    def embed(self, text, memory_action=None):
        self.calls.append((text, memory_action))
        return [0.1, 0.2, 0.3]


class FakeLocalMemory:
    def __init__(self):
        self.add_calls = []
        self.collection = FakeCollection()
        self.embedding_model = FakeEmbeddingModel()
        self.vector_store = SimpleNamespace(collection=self.collection)

    def add(self, messages, **kwargs):
        self.add_calls.append({"messages": messages, **kwargs})
        return {"results": [{"memory": messages[0]["content"], "event": "ADD"}]}


def _local_cfg(tmp_path):
    return {
        "local_mode": True,
        "minimax_api_key": "test-minimax-key",
        "minimax_base_url": "https://api.minimaxi.com/anthropic",
        "chroma_path": str(tmp_path / "mem0_chroma"),
        "collection_name": "hermes_memories",
        "user_id": "hermes-user",
        "agent_id": "hermes",
        "rerank": True,
    }


def test_local_mode_is_available_with_minimax_key(monkeypatch, tmp_path):
    monkeypatch.setattr(mem0_plugin, "_load_config", lambda: _local_cfg(tmp_path))
    provider = Mem0MemoryProvider()
    assert provider.is_available() is True


def test_local_mode_uses_memory_from_config_not_cloud_client(monkeypatch, tmp_path):
    fake_local = FakeLocalMemory()
    captured = {}

    class DummyMemory:
        @classmethod
        def from_config(cls, config):
            captured["config"] = config
            return fake_local

    class FailMemoryClient:
        def __init__(self, *args, **kwargs):
            raise AssertionError("cloud MemoryClient should not be used in local_mode")

    fake_mem0 = ModuleType("mem0")
    setattr(fake_mem0, "Memory", DummyMemory)
    setattr(fake_mem0, "MemoryClient", FailMemoryClient)

    monkeypatch.setattr(mem0_plugin, "_load_config", lambda: _local_cfg(tmp_path))
    monkeypatch.setitem(sys.modules, "mem0", fake_mem0)

    provider = Mem0MemoryProvider()
    provider.initialize("test-session")
    client = provider._get_client()

    assert provider._local_mode is True
    assert client is fake_local
    assert captured["config"]["llm"]["provider"] == "minimax"
    assert captured["config"]["llm"]["config"]["minimax_base_url"] == "https://api.minimaxi.com/anthropic"
    assert captured["config"]["vector_store"]["provider"] == "chroma"
    assert captured["config"]["vector_store"]["config"]["path"] == str(tmp_path / "mem0_chroma")


def test_local_mode_tool_calls_go_through_local_memory(monkeypatch, tmp_path):
    fake_local = FakeLocalMemory()

    class DummyMemory:
        @classmethod
        def from_config(cls, config):
            return fake_local

    fake_mem0 = ModuleType("mem0")
    setattr(fake_mem0, "Memory", DummyMemory)

    monkeypatch.setattr(mem0_plugin, "_load_config", lambda: _local_cfg(tmp_path))
    monkeypatch.setitem(sys.modules, "mem0", fake_mem0)

    provider = Mem0MemoryProvider()
    provider.initialize("test-session", user_id="tg_user_1")

    profile = json.loads(provider.handle_tool_call("mem0_profile", {}))
    search = json.loads(provider.handle_tool_call("mem0_search", {"query": "dark mode", "top_k": 3}))
    conclude = json.loads(provider.handle_tool_call("mem0_conclude", {"conclusion": "user prefers dark mode"}))

    assert profile["count"] == 1
    assert "local profile hit" in profile["result"]
    assert search["count"] == 1
    assert search["results"][0]["memory"] == "local search hit"
    assert conclude["result"] == "Fact stored."

    assert fake_local.collection.get_calls[0]["where"] == {"user_id": "tg_user_1"}
    assert fake_local.collection.get_calls[0]["include"] == ["metadatas"]
    assert fake_local.collection.query_calls[0]["where"] == {"user_id": "tg_user_1"}
    assert fake_local.collection.query_calls[0]["n_results"] == 3
    assert fake_local.collection.query_calls[0]["include"] == ["metadatas", "distances"]
    assert fake_local.collection.query_calls[0]["query_embeddings"][0] == pytest.approx([0.1, 0.2, 0.3])
    assert fake_local.embedding_model.calls == [("dark mode", "search")]
    add_call = fake_local.add_calls[0]
    assert add_call["user_id"] == "tg_user_1"
    assert add_call["agent_id"] == "hermes"
    assert add_call["infer"] is False
