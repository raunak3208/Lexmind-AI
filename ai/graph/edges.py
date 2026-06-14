"""
ai/graph/edges.py

Conditional routing logic for LangGraph pipelines.

Edge functions receive the current state and return the name
of the next node to execute. This is where branching happens:
  - if extraction failed → go to error node
  - if risk score is critical → flag for retry
  - if all good → continue to next step

Contract pipeline routing:
  route_after_extraction    failed → error | ok → classifier
  route_after_risk          failed → error | ok → augment_risk
  route_after_summarizer    failed → error | ok → finalize

Research pipeline routing:
  route_after_search        failed → error | ok → reader
  route_after_reader        failed → error | ok → writer
  route_after_writer        failed → error | ok → critic
"""

import logging

logger = logging.getLogger(__name__)


# ── CONTRACT PIPELINE EDGES ───────────────────────────────────────────────────

def route_after_extraction(state: dict) -> str:
    """
    After extraction: if failed go to error, else classify.
    Extraction failure is fatal — cannot continue without clauses.
    """
    if state.get("status") == "failed" or not state.get("extraction"):
        logger.warning("Routing to error: extraction failed")
        return "handle_error"
    return "classifier"


def route_after_classifier(state: dict) -> str:
    """
    After classifier: always continue to risk even if classifier failed.
    Classifier failure is non-fatal — we use original clause types.
    """
    if state.get("status") == "failed":
        logger.warning("Classifier failed — continuing with original clause types")
        # Reset status so pipeline continues
        state["status"] = "running"
    return "risk"


def route_after_risk(state: dict) -> str:
    """
    After risk scoring: if failed go to error, else augment with tools.
    """
    if state.get("status") == "failed" or not state.get("risk_report"):
        logger.warning("Routing to error: risk agent failed")
        return "handle_error"
    return "augment_risk"


def route_after_augment(state: dict) -> str:
    """After augmenting risk: always go to summarizer."""
    return "summarizer"


def route_after_summarizer(state: dict) -> str:
    """
    After summarizer: if failed go to error, else finalize.
    """
    if state.get("status") == "failed" or not state.get("summary"):
        logger.warning("Routing to error: summarizer failed")
        return "handle_error"
    return "finalize"


# ── RESEARCH PIPELINE EDGES ───────────────────────────────────────────────────

def route_after_search(state: dict) -> str:
    """After search: if failed go to error, else read."""
    if state.get("status") == "failed" or not state.get("search_results"):
        logger.warning("Routing to error: search failed")
        return "handle_error"
    return "reader"


def route_after_reader(state: dict) -> str:
    """
    After reader: if failed, still continue to writer with search results only.
    Scraping is non-fatal — writer can work from search snippets alone.
    """
    if state.get("status") == "failed":
        logger.warning("Reader failed — continuing to writer with search results only")
        state["scraped_content"] = "Scraping failed — using search snippets only."
        state["status"] = "running"
    return "writer"


def route_after_writer(state: dict) -> str:
    """After writer: if failed go to error, else critique."""
    if state.get("status") == "failed" or not state.get("report"):
        logger.warning("Routing to error: writer failed")
        return "handle_error"
    return "critic"