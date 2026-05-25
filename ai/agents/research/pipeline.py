"""
ai/agents/research/pipeline.py

ARIA research pipeline — adapted from your working pipeline.py.
Sync version used for direct CLI testing.
Async version (research_orchestrator.py) used by FastAPI SSE route.
"""

import logging
from ai.agents.research.agents import (
    build_search_agent,
    build_reader_agent,
    writer_chain,
    critic_chain,
)

logger = logging.getLogger(__name__)


def run_research_pipeline(topic: str) -> dict:
    state = {}

    logger.info("Step 1 — Search Agent")
    search_agent = build_search_agent()
    search_result = search_agent.invoke({
        "messages": [("user", f"Find recent, reliable and detailed information about: {topic}")]
    })
    state["search_results"] = search_result["messages"][-1].content
    logger.info("Search complete")

    logger.info("Step 2 — Reader Agent")
    reader_agent = build_reader_agent()
    reader_result = reader_agent.invoke({
        "messages": [("user",
            f"Based on the following search results about '{topic}', "
            f"pick the most relevant URL and scrape it for deeper content.\n\n"
            f"Search Results:\n{state['search_results'][:800]}"
        )]
    })
    state["scraped_content"] = reader_result["messages"][-1].content
    logger.info("Scraping complete")

    logger.info("Step 3 — Writer Chain")
    research_combined = (
        f"SEARCH RESULTS:\n{state['search_results']}\n\n"
        f"DETAILED SCRAPED CONTENT:\n{state['scraped_content']}"
    )
    state["report"] = writer_chain.invoke({
        "topic": topic,
        "research": research_combined,
    })
    logger.info("Report drafted")

    logger.info("Step 4 — Critic Chain")
    report_text = (
        f"Title: {state['report'].title}\n"
        f"Summary: {state['report'].summary}\n"
        f"Findings: {state['report'].findings}\n"
        f"Analysis: {state['report'].analysis}"
    )
    state["feedback"] = critic_chain.invoke({"report": report_text})
    logger.info(f"Critic score: {state['feedback'].score}/10")

    return state


if __name__ == "__main__":
    topic = input("Enter a research topic: ")
    result = run_research_pipeline(topic)
    print("\nReport:", result["report"].title)
    print("Score:", result["feedback"].score, "/10")
    print("Verdict:", result["feedback"].verdict)