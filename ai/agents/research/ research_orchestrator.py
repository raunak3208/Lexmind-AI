"""
ai/agents/research/research_orchestrator.py

Async SSE orchestrator — adapted from your server.py research_generator().
Yields real-time status events as each agent completes.
Used by the FastAPI /research/stream route.

Also exposes run_research_async() for saving final result to MongoDB
via the Node backend.
"""

import json
import logging
from typing import AsyncGenerator

from ai.agents.research.agents import (
    build_search_agent,
    build_reader_agent,
    writer_chain,
    critic_chain,
)

logger = logging.getLogger(__name__)


def _make_event(agent: str, status: str, message: str, detail: str = "") -> dict:
    return {
        "event": "message",
        "data": json.dumps({
            "agent":   agent,
            "status":  status,
            "message": message,
            "detail":  detail,
        }),
    }


async def research_stream_generator(topic: str) -> AsyncGenerator[dict, None]:
    """
    Async generator that yields SSE events as each ARIA agent completes.
    Consumed by FastAPI EventSourceResponse.
    """
    state = {}

    try:
        # Step 1 — Search Agent
        yield _make_event("search", "running", "Querying search indexes...", f"Searching: {topic}")

        search_agent = build_search_agent()
        search_result = await search_agent.ainvoke({
            "messages": [("user", f"Find recent, reliable and detailed information about: {topic}")]
        })
        state["search_results"] = search_result["messages"][-1].content

        yield _make_event("search", "complete", "Search complete", "Retrieved search results")

        # Step 2 — Reader Agent
        yield _make_event("reader", "running", "Reading and parsing source content...", "Extracting content from sources")

        reader_agent = build_reader_agent()
        reader_result = await reader_agent.ainvoke({
            "messages": [("user",
                f"Based on the following search results about '{topic}', "
                f"pick the most relevant URL and scrape it for deeper content.\n\n"
                f"Search Results:\n{state['search_results'][:800]}"
            )]
        })
        state["scraped_content"] = reader_result["messages"][-1].content

        yield _make_event("reader", "complete", "Content extracted", "Finished scraping primary source")

        # Step 3 — Writer Chain
        yield _make_event("writer", "running", "Synthesizing research into report...", "Drafting findings and analysis")

        research_combined = (
            f"SEARCH RESULTS:\n{state['search_results']}\n\n"
            f"DETAILED SCRAPED CONTENT:\n{state['scraped_content']}"
        )

        report_obj = await writer_chain.ainvoke({
            "topic":    topic,
            "research": research_combined,
        })
        state["report"] = report_obj

        yield _make_event(
            "writer", "complete",
            "Report generated",
            f"Generated {len(report_obj.findings)} key findings"
        )

        # Step 4 — Critic Chain
        yield _make_event("critic", "running", "Reviewing and scoring the report...", "Evaluating accuracy and coherence")

        report_text = (
            f"Title: {report_obj.title}\n"
            f"Summary: {report_obj.summary}\n"
            f"Findings: {report_obj.findings}\n"
            f"Analysis: {report_obj.analysis}"
        )

        critic_obj = await critic_chain.ainvoke({"report": report_text})
        state["feedback"] = critic_obj

        yield _make_event(
            "critic", "complete",
            f"Score: {critic_obj.score}/10",
            critic_obj.verdict
        )

        # Final payload — full report for Node to save to MongoDB
        yield {
            "event": "message",
            "data": json.dumps({
                "agent":  "system",
                "status": "complete",
                "message": "Pipeline finished",
                "report": {
                    "title":    report_obj.title,
                    "summary":  report_obj.summary,
                    "findings": report_obj.findings,
                    "analysis": report_obj.analysis,
                    "sources":  [s.model_dump() for s in report_obj.sources],
                    "critic":   critic_obj.model_dump(),
                    "meta": {
                        "topic":      topic,
                        "word_count": len(report_text.split()),
                    },
                },
            }),
        }

    except Exception as e:
        logger.exception(f"Research pipeline error: {e}")
        yield _make_event("system", "error", "Pipeline error", str(e))


async def run_research_async(topic: str) -> dict:
    """
    Run the full pipeline non-streaming and return the final result dict.
    Used when Node backend wants to save the report to MongoDB directly.
    """
    state = {}

    search_agent = build_search_agent()
    search_result = await search_agent.ainvoke({
        "messages": [("user", f"Find recent, reliable and detailed information about: {topic}")]
    })
    state["search_results"] = search_result["messages"][-1].content

    reader_agent = build_reader_agent()
    reader_result = await reader_agent.ainvoke({
        "messages": [("user",
            f"Based on the following search results about '{topic}', "
            f"pick the most relevant URL and scrape it for deeper content.\n\n"
            f"Search Results:\n{state['search_results'][:800]}"
        )]
    })
    state["scraped_content"] = reader_result["messages"][-1].content

    research_combined = (
        f"SEARCH RESULTS:\n{state['search_results']}\n\n"
        f"DETAILED SCRAPED CONTENT:\n{state['scraped_content']}"
    )

    report_obj = await writer_chain.ainvoke({"topic": topic, "research": research_combined})

    report_text = (
        f"Title: {report_obj.title}\nSummary: {report_obj.summary}\n"
        f"Findings: {report_obj.findings}\nAnalysis: {report_obj.analysis}"
    )

    critic_obj = await critic_chain.ainvoke({"report": report_text})

    return {
        "title":    report_obj.title,
        "summary":  report_obj.summary,
        "findings": report_obj.findings,
        "analysis": report_obj.analysis,
        "sources":  [s.model_dump() for s in report_obj.sources],
        "critic":   critic_obj.model_dump(),
        "meta": {
            "topic":      topic,
            "word_count": len(report_text.split()),
        },
    }