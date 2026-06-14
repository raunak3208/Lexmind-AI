"""
ai/graph/contract_graph.py

LangGraph state machine for contract analysis.

Graph structure:

  START
    |
    v
  pre_analysis  (tools — no LLM)
    |
    v
  extractor  ──fail──> error ──> END
    |
    v
  classifier  ──fail──> risk  (non-fatal, continues)
    |
    v
  risk  ──fail──> error ──> END
    |
    v
  augment_risk  (merge tool findings into risk report)
    |
    v
  summarizer  ──fail──> error ──> END
    |
    v
  finalize
    |
    v
  END

Replaces the old sequential run_full_pipeline() in orchestrator.py.
Adds: typed state, conditional routing, error handling, progress tracking.
"""

import time
import logging
from langgraph.graph import StateGraph, END

from ai.graph.states import ContractAnalysisState
from ai.graph.nodes import (
    pre_analysis_node,
    extractor_node,
    classifier_node,
    risk_node,
    augment_risk_node,
    summarizer_node,
    graph_build_node,
    finalize_node,
    error_node,
)
from ai.graph.edges import (
    route_after_extraction,
    route_after_classifier,
    route_after_risk,
    route_after_augment,
    route_after_summarizer,
)

logger = logging.getLogger(__name__)


def build_contract_graph():
    """
    Build and compile the contract analysis LangGraph.
    Call once and reuse the compiled graph.

    Returns:
        compiled LangGraph app ready for ainvoke()
    """
    graph = StateGraph(ContractAnalysisState)

    # Register all nodes
    graph.add_node("pre_analysis",  pre_analysis_node)
    graph.add_node("extractor",     extractor_node)
    graph.add_node("classifier",    classifier_node)
    graph.add_node("risk",          risk_node)
    graph.add_node("augment_risk",  augment_risk_node)
    graph.add_node("summarizer",    summarizer_node)
    graph.add_node("finalize",      finalize_node)
    graph.add_node("handle_error",         error_node)

    # Entry point
    graph.set_entry_point("pre_analysis")

    # Fixed edges (always go to next node)
    graph.add_edge("pre_analysis", "extractor")
    graph.add_edge("augment_risk", "summarizer")
    graph.add_edge("finalize",     END)
    graph.add_edge("handle_error",        END)

    # Conditional edges (routing based on state)
    graph.add_conditional_edges(
        "extractor",
        route_after_extraction,
        {"classifier": "classifier", "handle_error": "handle_error"},
    )
    graph.add_conditional_edges(
        "classifier",
        route_after_classifier,
        {"risk": "risk"},
    )
    graph.add_conditional_edges(
        "risk",
        route_after_risk,
        {"augment_risk": "augment_risk", "handle_error": "handle_error"},
    )
    graph.add_conditional_edges(
        "summarizer",
        route_after_summarizer,
        {"finalize": "finalize", "handle_error": "handle_error"},
    )

    compiled = graph.compile()
    logger.info("Contract analysis graph compiled")
    return compiled


# Singleton — compiled once on first import
_contract_graph = None


def get_contract_graph():
    """Return singleton compiled contract graph."""
    global _contract_graph
    if _contract_graph is None:
        _contract_graph = build_contract_graph()
    return _contract_graph


async def run_contract_graph(
    contract_text: str,
    document_id: str,
    filename: str,
):
    """
    Run the full contract analysis graph.
    Replaces run_full_pipeline() in orchestrator.py.

    Args:
        contract_text: full extracted text from the contract
        document_id:   MongoDB document ID
        filename:      original filename

    Returns:
        FullAnalysisResult
    """
    graph = get_contract_graph()

    initial_state: ContractAnalysisState = {
        "document_id":   document_id,
        "filename":      filename,
        "contract_text": contract_text,
        "status":        "running",
        "retry_count":   0,
        "current_node":  "start",
        "started_at":    time.time(),
        "tool_findings": {},
        "extraction":    None,
        "risk_report":   None,
        "summary":       None,
        "result":        None,
        "handle_error":         None,
        "elapsed":       None,
    }

    logger.info(
        f"Contract graph START  doc={document_id}  "
        f"file={filename}  text_len={len(contract_text)}"
    )

    final_state = await graph.ainvoke(initial_state)

    if final_state.get("status") == "failed":
        raise RuntimeError(
            f"Contract analysis graph failed at node "
            f"'{final_state.get('current_node')}': "
            f"{final_state.get('handle_error', 'unknown error')}"
        )

    logger.info(
        f"Contract graph DONE  doc={document_id}  "
        f"elapsed={final_state.get('elapsed')}s"
    )

    return final_state["result"]


async def stream_contract_graph(
    contract_text: str,
    document_id: str,
    filename: str,
):
    """
    Stream contract graph progress node by node.
    Yields dicts with current node name and status.
    Used by FastAPI SSE route for real-time progress.

    Yields:
        dict with keys: node, status, message
    """
    graph = get_contract_graph()

    initial_state: ContractAnalysisState = {
        "document_id":   document_id,
        "filename":      filename,
        "contract_text": contract_text,
        "status":        "running",
        "retry_count":   0,
        "current_node":  "start",
        "started_at":    time.time(),
        "tool_findings": {},
        "extraction":    None,
        "risk_report":   None,
        "summary":       None,
        "result":        None,
        "handle_error":         None,
        "elapsed":       None,
    }

    NODE_MESSAGES = {
        "pre_analysis":  "Running pre-analysis tools against vector store...",
        "extractor":     "Extracting clauses from contract text...",
        "classifier":    "Verifying and correcting clause types...",
        "risk":          "Scoring contract risk...",
        "augment_risk":  "Merging tool findings into risk report...",
        "summarizer":    "Generating plain-English summary...",
        "finalize":      "Assembling final analysis result...",
        "handle_error":         "Pipeline encountered an error",
    }

    async for event in graph.astream(initial_state):
        for node_name, node_state in event.items():
            yield {
                "node":    node_name,
                "status":  node_state.get("status", "running"),
                "message": NODE_MESSAGES.get(node_name, f"Running {node_name}..."),
                "handle_error":   node_state.get("handle_error"),
            }

            if node_name == "finalize" and node_state.get("result"):
                yield {
                    "node":    "complete",
                    "status":  "completed",
                    "message": "Analysis complete",
                    "result":  node_state["result"].model_dump(),
                }