"""
ai/agents/orchestrator.py

Entry point for the contract analysis pipeline.
Now delegates to the LangGraph state machine in ai/graph/contract_graph.py.

run_full_pipeline() is kept for backward compatibility —
all existing callers (ai/api/routes/analyze.py) work unchanged.
"""

import logging
from ai.schemas.document_schema import FullAnalysisResult

logger = logging.getLogger(__name__)
async def run_full_pipeline(
    contract_text: str,
    document_id: str,
    filename: str,
) -> FullAnalysisResult:
    """
    Run the full contract analysis pipeline via LangGraph.

    Graph: pre_analysis → extractor → classifier → risk →
           augment_risk → summarizer → finalize

    Args:
        contract_text: full extracted text from the contract
        document_id:   MongoDB document ID
        filename:      original filename

    Returns:
        FullAnalysisResult
    """
    from ai.graph.contract_graph import run_contract_graph

    logger.info(
        f"Pipeline START (LangGraph)  doc={document_id}  "
        f"file={filename}  text_len={len(contract_text)}"
    )

    result = await run_contract_graph(
        contract_text=contract_text,
        document_id=document_id,
        filename=filename,
    )

    logger.info(f"Pipeline DONE (LangGraph)  doc={document_id}")
    return result