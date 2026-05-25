"""Tests for the Mem0 memory plugin local-mode helpers."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

_repo_root = str(Path(__file__).resolve().parents[3])
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from plugins.memory.mem0 import Mem0MemoryProvider


class FakeCollection:
    def __init__(self, *, metadatas=None, query_metadatas=None, query_distances=None):
        self._metadatas = metadatas or []
        self._query_metadatas = query_metadatas or []
        self._query_distances = query_distances or []
        self.where_get = None
        self.get_include = None
        self.query_args = None

    def get(self, *, where=None, include=None):
        self.where_get = where
        self.get_include = include
        return {"metadatas": self._metadatas}

    def query(self, *, query_embeddings, n_results, where=None, include=None):
        self.query_args = {
            "query_embeddings": query_embeddings,
            "n_results": n_results,
            "where": where,
            "include": include,
        }
        return {
            "metadatas": [self._query_metadatas],
            "distances": [self._query_distances],
        }


class FakeEmbeddingModel:
    def __init__(self):
        self.calls = []

    def embed(self, text, memory_action=None):
        self.calls.append((text, memory_action))
        return np.array([0.1, 0.2, 0.3], dtype=np.float32)


def _make_provider(collection: FakeCollection):
    provider = Mem0MemoryProvider()
    provider._local_mode = True
    provider._user_id = "u-1"
    provider._agent_id = "agent-1"
    cast(Any, provider)._memory = SimpleNamespace(
        vector_store=SimpleNamespace(collection=collection),
        embedding_model=FakeEmbeddingModel(),
    )
    return provider


def test_mem0_profile_reads_all_local_memories_without_truncation():
    collection = FakeCollection(
        metadatas=[{"data": f"memory {i}", "user_id": "u-1"} for i in range(25)]
    )
    provider = _make_provider(collection)

    result = json.loads(provider.handle_tool_call("mem0_profile", {}))

    assert result["count"] == 25
    assert "memory 24" in result["result"]
    assert collection.where_get == {"user_id": "u-1"}
    assert collection.get_include == ["metadatas"]


def test_mem0_search_uses_direct_local_vector_query_for_ranked_results():
    collection = FakeCollection(
        query_metadatas=[
            {"data": "CAPE 投资规则与 QDII 配置", "user_id": "u-1"},
            {"data": "openclaw 长超时偏好", "user_id": "u-1"},
        ],
        query_distances=[0.2, 0.8],
    )
    provider = _make_provider(collection)

    result = json.loads(
        provider.handle_tool_call(
            "mem0_search", {"query": "enzo 投资策略 CAPE", "top_k": 2, "rerank": True}
        )
    )

    assert result["count"] == 2
    assert result["results"][0]["memory"] == "CAPE 投资规则与 QDII 配置"
    assert result["results"][0]["score"] == pytest.approx(0.8)
    assert result["results"][1]["score"] == pytest.approx(0.2)

    assert collection.query_args is not None
    assert collection.query_args["n_results"] == 2
    assert collection.query_args["where"] == {"user_id": "u-1"}
    assert collection.query_args["include"] == ["metadatas", "distances"]
    assert collection.query_args["query_embeddings"][0] == pytest.approx([0.1, 0.2, 0.3])
    assert all(isinstance(x, float) for x in collection.query_args["query_embeddings"][0])
    fake_memory = cast(Any, provider)._memory
    assert fake_memory.embedding_model.calls == [("enzo 投资策略 CAPE", "search")]
