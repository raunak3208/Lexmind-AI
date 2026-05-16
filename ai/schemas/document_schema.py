"""
ai/schemas/document_schema.py

Top-level schema returned by the orchestrator after the full
Extractor → Classifier → Risk → Summarizer pipeline runs.
"""

from typing import Optional
from pydantic import BaseModel, Field
from ai.schemas.clause_schema import ClauseExtractionResult
from ai.schemas.risk_schema   import RiskReport


class ContractSummary(BaseModel):
    document_id:   str
    filename:      str
    contract_type: str        = Field(..., description="e.g. NDA, SaaS Agreement, Employment Contract")
    parties:       list[str]  = Field(..., description="All parties identified in the contract")
    effective_date: Optional[str] = Field(None, description="Contract effective date if found")
    expiry_date:    Optional[str] = Field(None, description="Expiry / termination date if found")
    governing_law:  Optional[str] = Field(None, description="Jurisdiction / governing law")
    key_obligations: list[str]   = Field(default_factory=list, description="Top obligations for each party")
    executive_summary: str       = Field(..., description="Plain-English 5-6 sentence summary of the contract")


class FullAnalysisResult(BaseModel):
    document_id:  str
    filename:     str
    status:       str = "completed"
    extraction:   ClauseExtractionResult
    risk_report:  RiskReport
    summary:      ContractSummary