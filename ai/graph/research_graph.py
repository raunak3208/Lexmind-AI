"""
ai/graph/research_graph.py

LangGraph state machine for ARIA research pipeline.

Graph structure:

  START
    |
    v
  search  ──fail──> error ──> END
    |
    v
  reader  ──fail──> writer  (non-fatal, uses search results only)
    |
    v
  writer  ──fail──> error ──> END
    |
    v
  critic
    |
    v
  END

Replaces the old sequential pipeline in research_orchestrator.py.
"""

import time
import json
import logging
from typing import AsyncGenerator

from langgraph.graph import StateGraph, END

from ai.graph.states import ResearchState
from ai.graph.nodes import (
    search_node,
    reader_node,
    writer_node,
    critic_node,
    error_node,
)
from ai.graph.edges import (
    route_after_search,
    route_after_reader,
    route_after_writer,
)

logger = logging.getLogger(__name__)


def build_research_graph():
    """
    Build and compile the ARIA research LangGraph.
    Returns compiled graph ready for ainvoke() or astream().
    """
    graph = StateGraph(ResearchState)

    graph.add_node("search", search_node)
    graph.add_node("reader", reader_node)
    graph.add_node("writer", writer_node)
    graph.add_node("critic", critic_node)
    graph.add_node("handle_error",  error_node)

    graph.set_entry_point("search")

    graph.add_edge("critic", END)
    graph.add_edge("handle_error",  END)

    graph.add_conditional_edges(
        "search",
        route_after_search,
        {"reader": "reader", "handle_error": "handle_error"},
    )
    graph.add_conditional_edges(
        "reader",
        route_after_reader,
        {"writer": "writer"},
    )
    graph.add_conditional_edges(
        "writer",
        route_after_writer,
        {"critic": "critic", "handle_error": "handle_error"},
    )

    compiled = graph.compile()
    logger.info("Research graph compiled")
    return compiled


_research_graph = None


def get_research_graph():
    """Return singleton compiled research graph."""
    global _research_graph
    if _research_graph is None:
        _research_graph = build_research_graph()
    return _research_graph


async def run_research_graph(topic: str) -> dict:
    """
    Run the full ARIA research graph and return final result.
    Replaces run_research_async() in research_orchestrator.py.

    Args:
        topic: research topic string

    Returns:
        dict with title, summary, findings, analysis, sources, critic, meta
    """
    graph = get_research_graph()

    initial_state: ResearchState = {
        "topic":           topic,
        "status":          "running",
        "current_node":    "start",
        "retry_count":     0,
        "started_at":      time.time(),
        "search_results":  None,
        "scraped_content": None,
        "report":          None,
        "feedback":        None,
        "handle_error":           None,
        "elapsed":         None,
    }

    logger.info(f"Research graph START  topic='{topic[:60]}'")

    final_state = await graph.ainvoke(initial_state)

    if final_state.get("status") == "failed":
        raise RuntimeError(
            f"Research graph failed at '{final_state.get('current_node')}': "
            f"{final_state.get('handle_error', 'unknown error')}"
        )

    report   = final_state["report"]
    feedback = final_state["feedback"]
    elapsed  = final_state.get("elapsed", 0)

    logger.info(
        f"Research graph DONE  topic='{topic[:40]}'  "
        f"score={feedback.score}/10  elapsed={elapsed}s"
    )

    return {
        "title":    report.title,
        "summary":  report.summary,
        "findings": report.findings,
        "analysis": report.analysis,
        "sources":  [s.model_dump() for s in report.sources],
        "critic":   feedback.model_dump(),
        "meta": {
            "topic":      topic,
            "word_count": len(report.summary.split()),
            "elapsed":    elapsed,
        },
    }


NODE_MESSAGES = {
    "search": ("search", "Querying search indexes..."),
    "reader": ("reader", "Reading and scraping source content..."),
    "writer": ("writer", "Synthesizing research into report..."),
    "critic": ("critic", "Reviewing and scoring the report..."),
    "handle_error":  ("system", "Pipeline encountered an error"),
}


async def stream_research_graph(topic: str) -> AsyncGenerator[dict, None]:
    """
    Stream research graph progress as SSE-compatible events.
    Replaces research_stream_generator() in research_orchestrator.py.

    Yields:
        dict with event and data keys for SSE
    """
    graph = get_research_graph()

    initial_state: ResearchState = {
        "topic":           topic,
        "status":          "running",
        "current_node":    "start",
        "retry_count":     0,
        "started_at":      time.time(),
        "search_results":  None,
        "scraped_content": None,
        "report":          None,
        "feedback":        None,
        "handle_error":           None,
        "elapsed":         None,
    }

    final_report   = None
    final_feedback = None

    async for event in graph.astream(initial_state):
        for node_name, node_state in event.items():
            agent_name, message = NODE_MESSAGES.get(node_name, ("system", node_name))
            status = node_state.get("status", "running")

            if status == "completed" and node_name == "critic":
                final_report   = node_state.get("report")
                final_feedback = node_state.get("feedback")
                detail = f"Score: {final_feedback.score}/10" if final_feedback else "Done"
            elif status == "failed":
                detail = node_state.get("handle_error", "Unknown error")
            else:
                detail = f"Processing {node_name}..."

            yield {
                "event": "message",
                "data": json.dumps({
                    "agent":   agent_name,
                    "status":  "complete" if status == "completed" else status,
                    "message": message,
                    "detail":  detail,
                }),
            }

    # Emit final full report for Node to save
    if final_report and final_feedback:
        report_text = f"{final_report.summary} {final_report.analysis}"
        yield {
            "event": "message",
            "data": json.dumps({
                "agent":  "system",
                "status": "complete",
                "message": "Pipeline finished",
                "report": {
                    "title":    final_report.title,
                    "summary":  final_report.summary,
                    "findings": final_report.findings,
                    "analysis": final_report.analysis,
                    "sources":  [s.model_dump() for s in final_report.sources],
                    "critic":   final_feedback.model_dump(),
                    "meta": {
                        "topic":      topic,
                        "word_count": len(report_text.split()),
                    },
                },
            }),
        }