"""
ai/knowledge_graph/schemas.py

Pydantic models for the knowledge graph layer.

Entity types found in legal contracts:
  PARTY           — named parties (Acme Corp, John Doe)
  CLAUSE          — clause types (Payment Clause, Termination Clause)
  OBLIGATION      — what a party must do
  DATE            — effective date, expiry date, notice periods
  AMOUNT          — monetary values ($5,000, net 30)
  JURISDICTION    — governing law, courts
  DEFINED_TERM    — capitalised defined terms ("Confidential Information")
  DOCUMENT        — the contract itself, exhibits, schedules

Relationship types:
  PARTY_TO        — party is a party to the contract
  HAS_OBLIGATION  — party has a specific obligation
  GOVERNED_BY     — contract governed by a jurisdiction
  CONTAINS        — contract/clause contains another clause
  CONNECTED_TO    — generic relationship
  DUE_WITHIN      — obligation/payment due within a time period
  PENALISED_BY    — clause has an associated penalty
  DEFINED_AS      — term is defined as something
  REFERENCES      — clause references another clause

Location: ai/knowledge_graph/schemas.py
"""

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class EntityType(str, Enum):
    PARTY        = "PARTY"
    CLAUSE       = "CLAUSE"
    OBLIGATION   = "OBLIGATION"
    DATE         = "DATE"
    AMOUNT       = "AMOUNT"
    JURISDICTION = "JURISDICTION"
    DEFINED_TERM = "DEFINED_TERM"
    DOCUMENT     = "DOCUMENT"
    OTHER        = "OTHER"


class RelationType(str, Enum):
    PARTY_TO      = "PARTY_TO"
    HAS_OBLIGATION = "HAS_OBLIGATION"
    GOVERNED_BY   = "GOVERNED_BY"
    CONTAINS      = "CONTAINS"
    CONNECTED_TO  = "CONNECTED_TO"
    DUE_WITHIN    = "DUE_WITHIN"
    PENALISED_BY  = "PENALISED_BY"
    DEFINED_AS    = "DEFINED_AS"
    REFERENCES    = "REFERENCES"


class Entity(BaseModel):
    entity_id:   str
    name:        str
    entity_type: EntityType
    description: Optional[str] = None
    source_text: Optional[str] = None
    page:        Optional[int] = None


class Relation(BaseModel):
    relation_id:   str
    source_id:     str
    target_id:     str
    relation_type: RelationType
    label:         Optional[str] = None
    weight:        float = 1.0


class ExtractionResult(BaseModel):
    document_id: str
    filename:    str
    entities:    list[Entity]
    relations:   list[Relation]
    total_entities:  int
    total_relations: int


class GraphQueryResult(BaseModel):
    query:        str
    document_id:  Optional[str]
    answer_nodes: list[dict]
    relationships: list[dict]
    graph_context: str
    total_nodes:   int
