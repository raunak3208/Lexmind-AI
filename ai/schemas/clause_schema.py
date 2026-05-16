"""
ai/schemas/clause_schema.py

Pydantic models for extracted clause data.
The Extractor Agent returns a list of ExtractedClause objects.
"""

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class ClauseType(str, Enum):
    PAYMENT          = "payment"
    TERMINATION      = "termination"
    CONFIDENTIALITY  = "confidentiality"
    LIABILITY        = "liability"
    INDEMNIFICATION  = "indemnification"
    INTELLECTUAL_PROPERTY = "intellectual_property"
    DISPUTE_RESOLUTION    = "dispute_resolution"
    GOVERNING_LAW    = "governing_law"
    FORCE_MAJEURE    = "force_majeure"
    AMENDMENT        = "amendment"
    ASSIGNMENT       = "assignment"
    WARRANTY         = "warranty"
    PENALTY          = "penalty"
    NOTICE           = "notice"
    OTHER            = "other"


class ExtractedClause(BaseModel):
    clause_id:    str          = Field(..., description="Unique ID e.g. C-001")
    clause_type:  ClauseType   = Field(..., description="Category of this clause")
    heading:      Optional[str]= Field(None, description="Original heading in the contract, if any")
    text:         str          = Field(..., description="Exact or near-exact clause text")
    page:         Optional[int]= Field(None, description="Page number where clause appears")
    section:      Optional[str]= Field(None, description="Section number e.g. '4.2'")
    parties_mentioned: list[str] = Field(default_factory=list, description="Party names referenced in this clause")


class ClauseExtractionResult(BaseModel):
    document_id: str
    filename:    str
    total_clauses: int
    clauses:     list[ExtractedClause]