"""
ai/agents/research/research_orchestrator.py

Entry point for the ARIA research pipeline.
Now delegates to the LangGraph state machine in ai/graph/research_graph.py.

Both functions kept for backward compatibility —
all existing callers (ai/api/routes/research.py) work unchanged.
"""

import logging
from typing import AsyncGenerator

logger = logging.getLogger(__name__)


async def research_stream_generator(topic: str) -> AsyncGenerator[dict, None]:
    """
    Stream ARIA research progress as SSE events via LangGraph.
    Replaces old sequential generator.
    """
    from ai.graph.research_graph import stream_research_graph

    logger.info(f"Research stream START (LangGraph)  topic='{topic[:60]}'")

    async for event in stream_research_graph(topic):
        yield event


async def run_research_async(topic: str) -> dict:
    """
    Run full ARIA research pipeline via LangGraph (non-streaming).
    Returns final result dict for Node to save to MongoDB.
    """
    from ai.graph.research_graph import run_research_graph

    logger.info(f"Research async START (LangGraph)  topic='{topic[:60]}'")

    result = await run_research_graph(topic)

    logger.info(f"Research async DONE (LangGraph)  topic='{topic[:40]}'")
    return result