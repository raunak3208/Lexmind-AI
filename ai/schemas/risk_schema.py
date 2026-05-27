"""
ai/schemas/risk_schema.py

Pydantic models for risk scoring output.
The Risk Agent returns a RiskReport per document.
"""

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class RiskLevel(str, Enum):
    LOW      = "low"
    MEDIUM   = "medium"
    HIGH     = "high"
    CRITICAL = "critical"


class RiskFlag(BaseModel):
    flag_id:     str       = Field(..., description="Unique ID e.g. R-001")
    clause_id:   str       = Field(..., description="Which clause triggered this flag")
    risk_level:  RiskLevel = Field(..., description="Severity of this risk")
    category:    str       = Field(..., description="Type of risk: ambiguity | one-sided | missing-clause | punitive | other")
    description: str       = Field(..., description="Plain-English explanation of the risk")
    suggestion:  str       = Field(..., description="Recommended fix or negotiation point")
    flagged_text: Optional[str] = Field(None, description="Exact text that triggered the flag")


class RiskReport(BaseModel):
    document_id:   str
    filename:      str
    overall_risk:  RiskLevel
    risk_score:    int       = Field(..., ge=0, le=100, description="0=safe, 100=extremely risky")
    total_flags:   int
    flags:         list[RiskFlag]
    summary:       str       = Field(..., description="2-3 sentence executive risk summary")


