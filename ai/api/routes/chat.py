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
from ai.memory.conversation_memory import add_turn, get_messages, clear_history

router = APIRouter()
logger = logging.getLogger(__name__)


class ChatRequest(BaseModel):
    document_id: str
    user_id:     str = "default"
    question:    str


class ChatResponse(BaseModel):
    document_id: str
    user_id:     str
    question:    str
    answer:      str
    turn_number: int


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
        result = await ask(
            question=req.question,
            document_id=req.document_id,
        )
        answer = result["answer"]

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