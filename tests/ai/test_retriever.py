"""
tests/ai/test_retriever.py

Tests for ai/rag/retriever.py
These test the factory logic and filter building WITHOUT calling Mistral APIs.

Run with:  cd lexmind && pytest tests/ai/test_retriever.py -v
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from ai.rag.retriever import _build_filter, get_retriever


# ── Filter builder tests ──────────────────────────────────────────────────────

def test_filter_none_when_no_args():
    """No args -> no filter (search all docs)."""
    result = _build_filter()
    assert result is None


def test_filter_document_id_only():
    result = _build_filter(document_id="doc-123")
    assert result == {"document_id": {"$eq": "doc-123"}}


def test_filter_clause_type_only():
    result = _build_filter(clause_type="payment")
    assert result == {"clause_type": {"$eq": "payment"}}


def test_filter_both_uses_and():
    result = _build_filter(document_id="doc-123", clause_type="payment")
    assert result["$and"] is not None
    assert len(result["$and"]) == 2


def test_filter_no_negative_noise():
    """Empty strings should not create filters."""
    result = _build_filter(document_id=None, clause_type=None)
    assert result is None


# ── Factory tests ─────────────────────────────────────────────────────────────

def test_unknown_strategy_falls_back_to_mmr(monkeypatch):
    """Unknown strategy should not raise — falls back to MMR with a warning."""
    created = []

    def mock_mmr(document_id=None, k=None):
        created.append("mmr")
        return "mmr_retriever"

    monkeypatch.setattr("ai.rag.retriever.get_mmr_retriever", mock_mmr)

    result = get_retriever(strategy="nonexistent_strategy")
    assert result == "mmr_retriever"
    assert created == ["mmr"]


def test_factory_routes_mmr(monkeypatch):
    monkeypatch.setattr("ai.rag.retriever.get_mmr_retriever",
                        lambda **kw: "mmr")
    assert get_retriever(strategy="mmr") == "mmr"


def test_factory_routes_similarity(monkeypatch):
    monkeypatch.setattr("ai.rag.retriever.get_similarity_retriever",
                        lambda **kw: "sim")
    assert get_retriever(strategy="similarity") == "sim"


def test_factory_routes_multi_query(monkeypatch):
    monkeypatch.setattr("ai.rag.retriever.get_multi_query_retriever",
                        lambda **kw: "mq")
    assert get_retriever(strategy="multi_query") == "mq"


def test_factory_clause_type_requires_clause_type():
    """Passing strategy=clause_type without clause_type should raise ValueError."""
    with pytest.raises(ValueError, match="clause_type is required"):
        get_retriever(strategy="clause_type", clause_type=None)


def test_factory_clause_type_passes_through(monkeypatch):
    received = {}

    def mock_clause(clause_type, document_id=None, k=None):
        received["clause_type"] = clause_type
        return "ct_retriever"

    monkeypatch.setattr("ai.rag.retriever.get_clause_type_retriever", mock_clause)
    result = get_retriever(strategy="clause_type", clause_type="payment")
    assert result == "ct_retriever"
    assert received["clause_type"] == "payment"
    