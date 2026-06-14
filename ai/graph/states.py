"""
ai/graph/states.py

Typed state objects for LangGraph pipelines.

ContractAnalysisState  — used by the contract analysis graph
ResearchState          — used by the ARIA research graph

Every node in the graph receives the full state and returns
a partial state dict with only the fields it updates.
LangGraph merges them automatically.
"""

from typing import Optional, Any
from typing_extensions import TypedDict

from ai.schemas.clause_schema import ClauseExtractionResult
from ai.schemas.risk_schema import RiskReport
from ai.schemas.document_schema import ContractSummary, FullAnalysisResult


class ContractAnalysisState(TypedDict, total=False):
    """
    State for the contract analysis LangGraph pipeline.

    Fields are populated progressively as each node runs.
    All fields optional — only the final state has everything filled.
    """
    # Input — set before graph starts
    document_id:    str
    filename:       str
    contract_text:  str

    # Pre-analysis tools output
    tool_findings:  dict

    # Agent outputs
    extraction:     Optional[ClauseExtractionResult]
    risk_report:    Optional[RiskReport]
    summary:        Optional[ContractSummary]

    # Knowledge graph output
    graph_result:   Optional[dict]

    # Final result
    result:         Optional[FullAnalysisResult]

    # Control flow
    status:         str   # running | completed | failed | retrying
    error:          Optional[str]
    retry_count:    int
    current_node:   str

    # Timing
    started_at:     float
    elapsed:        Optional[float]


class ResearchState(TypedDict, total=False):
    """
    State for the ARIA research LangGraph pipeline.

    Fields populated progressively as Search → Reader → Writer → Critic run.
    """
    # Input
    topic: str

    # Agent outputs
    search_results:  Optional[str]
    scraped_content: Optional[str]
    report:          Optional[Any]   # ResearchReport Pydantic object
    feedback:        Optional[Any]   # CriticReview Pydantic object

    # Control flow
    status:       str
    error:        Optional[str]
    current_node: str
    retry_count:  int

    # Timing
    started_at:  float
    elapsed:     Optional[float]