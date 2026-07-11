"""
ai/api/routes/chat.py

POST /chat
Chat with a specific contract — uses RAG + conversation memory.
Each document_id + user_id gets its own persistent chat history.
"""

import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ai.rag.rag_chain             import ask
from ai.guardrails.guardrail_service import protect_query, protect_output
from ai.memory.conversation_memory import add_turn, get_messages, clear_history

router = APIRouter()
logger = logging.getLogger(__name__)


class ChatRequest(BaseModel):
    document_id: str
    user_id:     str = "default"
    question:    str
    strategy:    str = "hybrid"


class ChatResponse(BaseModel):
    document_id: str
    user_id:     str
    question:    str
    answer:      str
    turn_number: int
    cache_hit:   bool = False
    warnings:    list = []


class ClearRequest(BaseModel):
    document_id: str
    user_id:     str = "default"


@router.post("", response_model=ChatResponse)
async def chat_with_document(req: ChatRequest):
    """
    Ask a question about a specific contract.
    Maintains conversation history per document+user session.
    """
    if not req.question.strip():
        raise HTTPException(status_code=422, detail="Question cannot be empty.")

    logger.info(
        f"Chat: doc={req.document_id}  user={req.user_id}  "
        f"q='{req.question[:60]}'"
    )

    try:
        # Use hybrid by default for chat — best of dense + sparse
        # Falls back to mmr if strategy unrecognised
        # Input guard — validate and sanitise query
        guard = protect_query(req.question)
        if not guard["allowed"]:
            return error(res, guard["blocked_reason"], 400)

        valid = {"hybrid", "reranked", "mmr", "similarity", "multi_query"}
        strategy = req.strategy if req.strategy in valid else "hybrid"
        result = await ask(
            question=req.question,
            document_id=req.document_id,
            strategy=strategy,
        )
        raw_answer = result["answer"]

        # Output guard — redact PII, check for hallucination signals
        output_guard = protect_output(raw_answer)
        answer = output_guard["text"]

        # Store this turn in memory
        add_turn(
            document_id=req.document_id,
            user_id=req.user_id,
            question=req.question,
            answer=answer,
        )

        # Count turns (each turn = 2 messages: human + AI)
        history = get_messages(req.document_id, req.user_id)
        turn_number = len(history) // 2

    except Exception as e:
        logger.exception(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")

    return ChatResponse(
        document_id=req.document_id,
        user_id=req.user_id,
        question=req.question,
        answer=answer,
        turn_number=turn_number,
    )


@router.delete("/clear")
async def clear_chat_history(req: ClearRequest):
    """Clear conversation history for a document+user session."""
    clear_history(req.document_id, req.user_id)
    return {"status": "cleared", "document_id": req.document_id, "user_id": req.user_id}
