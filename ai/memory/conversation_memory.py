"""
ai/memory/conversation_memory.py

Per-document conversation memory.
Each document gets its own chat history so users can "chat with a contract".

Storage: in-memory dict (dev) — replace with Redis for production.
Key    : document_id  →  list of LangChain messages

Used by the /chat FastAPI route and the RAG chain.
"""

import logging
from typing import Optional

from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

logger = logging.getLogger(__name__)


# Session key = "{document_id}:{user_id}"
_store: dict[str, ChatMessageHistory] = {}


def _session_key(document_id: str, user_id: str = "default") -> str:
    return f"{document_id}:{user_id}"


def get_history(document_id: str, user_id: str = "default") -> ChatMessageHistory:
    """
    Get (or create) the chat history for a document+user session.

    Args:
        document_id: the contract being chatted with
        user_id:     user session ID (from Node JWT)

    Returns:
        ChatMessageHistory — a mutable LangChain message store
    """
    key = _session_key(document_id, user_id)
    if key not in _store:
        logger.info(f"Creating new chat session: {key}")
        _store[key] = ChatMessageHistory()
    return _store[key]


def add_turn(
    document_id: str,
    user_id: str,
    question: str,
    answer: str,
) -> None:
    """
    Append a human/AI turn to the session history.

    Args:
        document_id: contract ID
        user_id:     user session ID
        question:    user's message
        answer:      LexMind's response
    """
    history = get_history(document_id, user_id)
    history.add_user_message(question)
    history.add_ai_message(answer)
    logger.debug(f"Turn added to session {_session_key(document_id, user_id)}")


def get_messages(document_id: str, user_id: str = "default") -> list[BaseMessage]:
    """Return all messages in a session (for passing to LLM context)."""
    return get_history(document_id, user_id).messages


def clear_history(document_id: str, user_id: str = "default") -> None:
    """Clear the chat history for a document session."""
    key = _session_key(document_id, user_id)
    if key in _store:
        _store[key].clear()
        logger.info(f"Cleared chat history for session: {key}")


def list_sessions() -> list[str]:
    """Return all active session keys (for debugging / admin)."""
    return list(_store.keys())