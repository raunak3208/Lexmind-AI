"""
ai/api/routes/research.py

GET  /research/stream?topic=...   SSE stream — yields agent status events in real time
POST /research                    Non-streaming — runs full pipeline, returns JSON result
"""

import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from ai.agents.research.research_orchestrator import (
    research_stream_generator,
    run_research_async,
)

router = APIRouter()
logger = logging.getLogger(__name__)


class ResearchRequest(BaseModel):
    topic: str


@router.get("/stream")
async def stream_research(topic: str):
    """
    SSE endpoint — streams agent status events as they happen.
    Connect from Node backend or frontend with EventSource.

    Query param: topic (string)
    Events: search, reader, writer, critic, system
    """
    if not topic or not topic.strip():
        raise HTTPException(status_code=422, detail="topic query param is required")

    logger.info(f"SSE research stream started: topic='{topic[:60]}'")
    return EventSourceResponse(research_stream_generator(topic))


@router.post("")
async def run_research(req: ResearchRequest):
    """
    Non-streaming research endpoint.
    Runs full pipeline and returns complete report as JSON.
    Used by Node backend to save result to MongoDB.
    """
    if not req.topic.strip():
        raise HTTPException(status_code=422, detail="topic is required")

    logger.info(f"Research POST: topic='{req.topic[:60]}'")

    try:
        result = await run_research_async(req.topic)
    except Exception as e:
        logger.exception(f"Research pipeline failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    return {"status": "completed", "result": result}