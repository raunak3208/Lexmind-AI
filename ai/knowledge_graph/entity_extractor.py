"""
ai/knowledge_graph/entity_extractor.py

Uses Mistral (free tier) to extract entities and relationships
from contract text and return structured JSON.

Extraction prompt is engineered specifically for legal contracts —
it instructs Mistral to find parties, obligations, amounts, dates,
clauses and the edges connecting them.

Location: ai/knowledge_graph/entity_extractor.py
"""

import json
import logging
from functools import lru_cache

from langchain_mistralai import ChatMistralAI
from langchain_core.messages import SystemMessage, HumanMessage

from ai.config import settings
from ai.knowledge_graph.schemas import (
    Entity, Relation, ExtractionResult,
    EntityType, RelationType,
)

logger = logging.getLogger(__name__)

MAX_TEXT_LENGTH = 24000

EXTRACTION_SYSTEM = """You are a legal knowledge graph extraction engine.
Extract ALL entities and relationships from the contract text.

Entity types:
  PARTY        — named parties, companies, individuals
  CLAUSE       — clause types (payment, termination, confidentiality etc.)
  OBLIGATION   — specific obligations or duties
  DATE         — any date, time period, or deadline
  AMOUNT       — monetary values, fees, penalties
  JURISDICTION — governing law, courts, locations
  DEFINED_TERM — capitalised defined terms like "Confidential Information"
  DOCUMENT     — the contract itself, exhibits, schedules

Relationship types:
  PARTY_TO       — party participates in the contract
  HAS_OBLIGATION — party has an obligation
  GOVERNED_BY    — contract/clause governed by jurisdiction
  CONTAINS       — contract/clause contains another clause
  DUE_WITHIN     — obligation/payment due within a time period
  PENALISED_BY   — breach penalised by amount or action
  DEFINED_AS     — defined term defined as description
  REFERENCES     — clause references another clause
  CONNECTED_TO   — generic connection

OUTPUT FORMAT — respond ONLY with valid JSON, no markdown:
{
  "entities": [
    {
      "entity_id": "E001",
      "name": "Acme Corp",
      "entity_type": "PARTY",
      "description": "The Client party",
      "source_text": "Acme Corp (the Client)"
    }
  ],
  "relations": [
    {
      "relation_id": "R001",
      "source_id": "E001",
      "target_id": "E003",
      "relation_type": "HAS_OBLIGATION",
      "label": "must pay $5000/month"
    }
  ]
}

RULES:
- entity_id sequential: E001, E002, E003...
- relation_id sequential: R001, R002...
- Every entity referenced in relations MUST exist in entities list
- Extract at minimum: all parties, all clause types, all monetary amounts, all dates
- Output ONLY the JSON object. No explanation, no markdown fences.
"""

EXTRACTION_HUMAN = """Extract all entities and relationships from this contract:

{contract_text}
"""


@lru_cache(maxsize=1)
def _get_llm():
    return ChatMistralAI(
        model=settings.mistral_llm_model,
        mistral_api_key=settings.mistral_api_key,
        temperature=0,
        max_retries=3,
    )


def _parse_extraction(
    raw_json: str,
    document_id: str,
    filename: str,
) -> ExtractionResult:
    """Parse LLM JSON output into ExtractionResult."""
    try:
        cleaned = (
            raw_json.strip()
            .removeprefix("```json")
            .removeprefix("```")
            .removesuffix("```")
            .strip()
        )
        data = json.loads(cleaned)
    except json.JSONDecodeError as e:
        logger.error(f"Entity extraction JSON parse error: {e}")
        return ExtractionResult(
            document_id=document_id,
            filename=filename,
            entities=[],
            relations=[],
            total_entities=0,
            total_relations=0,
        )

    entities = []
    for item in data.get("entities", []):
        try:
            etype = EntityType(item.get("entity_type", "OTHER"))
        except ValueError:
            etype = EntityType.OTHER

        entities.append(Entity(
            entity_id=item.get("entity_id", f"E{len(entities)+1:03d}"),
            name=item.get("name", ""),
            entity_type=etype,
            description=item.get("description"),
            source_text=item.get("source_text"),
        ))

    entity_ids = {e.entity_id for e in entities}
    relations = []
    for item in data.get("relations", []):
        src = item.get("source_id", "")
        tgt = item.get("target_id", "")

        # Skip if either end doesn't exist
        if src not in entity_ids or tgt not in entity_ids:
            continue

        try:
            rtype = RelationType(item.get("relation_type", "CONNECTED_TO"))
        except ValueError:
            rtype = RelationType.CONNECTED_TO

        relations.append(Relation(
            relation_id=item.get("relation_id", f"R{len(relations)+1:03d}"),
            source_id=src,
            target_id=tgt,
            relation_type=rtype,
            label=item.get("label"),
            weight=float(item.get("weight", 1.0)),
        ))

    return ExtractionResult(
        document_id=document_id,
        filename=filename,
        entities=entities,
        relations=relations,
        total_entities=len(entities),
        total_relations=len(relations),
    )


async def extract_entities(
    contract_text: str,
    document_id: str,
    filename: str,
) -> ExtractionResult:
    """
    Extract entities and relationships from contract text using Mistral.

    Args:
        contract_text: full contract text
        document_id:   MongoDB document ID
        filename:      original filename

    Returns:
        ExtractionResult with entities and relations lists
    """
    if len(contract_text) > MAX_TEXT_LENGTH:
        logger.warning(
            f"Contract truncated from {len(contract_text)} "
            f"to {MAX_TEXT_LENGTH} for graph extraction"
        )
        contract_text = contract_text[:MAX_TEXT_LENGTH]

    llm = _get_llm()
    messages = [
        SystemMessage(content=EXTRACTION_SYSTEM),
        HumanMessage(content=EXTRACTION_HUMAN.format(contract_text=contract_text)),
    ]

    logger.info(f"Extracting entities for doc={document_id}")
    response = await llm.ainvoke(messages)

    result = _parse_extraction(response.content, document_id, filename)
    logger.info(
        f"Extraction done: {result.total_entities} entities, "
        f"{result.total_relations} relations"
    )
    return result
