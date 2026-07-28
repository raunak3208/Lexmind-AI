"""
ai/knowledge_graph/graph_service.py

Single entry point for all knowledge graph operations.
All routes and graph nodes import from here.

Functions:
  build_document_graph(text, doc_id, filename)  extract entities, build and save graph
  query_graph(query, doc_id)                    graph-enhanced retrieval
  get_party_obligations(party, doc_id)          all obligations for a party
  get_clause_connections(clause_type, doc_id)   connected entities for a clause type
  get_cross_document_query(query)               query across ALL contracts
  delete_document_graph(doc_id)                 cleanup on contract deletion
  get_stats()                                   graph statistics

Location: ai/knowledge_graph/graph_service.py
"""

import logging
from typing import Optional

from ai.config import settings

logger = logging.getLogger(__name__)


async def build_document_graph(
    contract_text: str,
    document_id: str,
    filename: str,
) -> dict:
    """
    Full pipeline: extract entities → build graph → save to disk.

    Called by the LangGraph graph_build_node after analysis completes.

    Args:
        contract_text: full contract text
        document_id:   MongoDB document ID
        filename:      original filename

    Returns:
        dict with total_entities, total_relations, status
    """
    if not settings.enable_knowledge_graph:
        return {"status": "disabled", "total_entities": 0, "total_relations": 0}

    try:
        from ai.knowledge_graph.entity_extractor import extract_entities
        from ai.knowledge_graph.graph_builder import build_and_save_graph

        # Step 1: extract entities and relations via Mistral
        extraction = await extract_entities(contract_text, document_id, filename)

        # Step 2: build NetworkX graph and save to disk
        graph = await build_and_save_graph(extraction, document_id)

        logger.info(
            f"Knowledge graph built: doc={document_id} "
            f"nodes={graph.number_of_nodes()} "
            f"edges={graph.number_of_edges()}"
        )

        return {
            "status":          "completed",
            "document_id":     document_id,
            "total_entities":  graph.number_of_nodes(),
            "total_relations": graph.number_of_edges(),
        }

    except Exception as e:
        logger.error(f"Graph build failed for doc={document_id}: {e}")
        return {"status": "failed", "error": str(e), "total_entities": 0, "total_relations": 0}


def query_graph(
    query: str,
    document_id: Optional[str] = None,
    max_hops: int = 2,
) -> str:
    """
    Retrieve graph context for a query.
    Returns a structured text string describing relevant entities and relationships.
    Used to augment RAG with graph context.

    Args:
        query:       user query
        document_id: restrict to one document (None = all documents)
        max_hops:    how many relationship hops to traverse from matched entities

    Returns:
        Formatted string with graph context — passed to LLM alongside chunks
    """
    if not settings.enable_knowledge_graph:
        return ""

    try:
        from ai.knowledge_graph.graph_store import load_graph, load_global_graph
        from ai.knowledge_graph.graph_retriever import retrieve_graph_context

        graph = load_graph(document_id) if document_id else load_global_graph()

        if graph is None or graph.number_of_nodes() == 0:
            logger.info(f"No graph found for doc={document_id or 'global'}")
            return ""

        context = retrieve_graph_context(query, graph, max_hops=max_hops)
        return context

    except Exception as e:
        logger.warning(f"Graph query failed: {e}")
        return ""


def get_party_obligations(
    party_name: str,
    document_id: Optional[str] = None,
) -> list[dict]:
    """
    Get all obligations for a specific party from the graph.
    The killer feature — impossible with flat RAG.

    Args:
        party_name:  exact or partial party name
        document_id: restrict to one document

    Returns:
        list of obligation dicts: { obligation, related_to, relation }
    """
    if not settings.enable_knowledge_graph:
        return []

    try:
        from ai.knowledge_graph.graph_store import load_graph, load_global_graph
        from ai.knowledge_graph.graph_builder import get_party_obligations as _get

        graph = load_graph(document_id) if document_id else load_global_graph()
        if not graph:
            return []

        return _get(graph, party_name)

    except Exception as e:
        logger.warning(f"get_party_obligations failed: {e}")
        return []


def get_clause_connections(
    clause_type: str,
    document_id: Optional[str] = None,
) -> list[dict]:
    """
    Get all entities connected to a clause type.

    Args:
        clause_type: e.g. "payment", "termination", "confidentiality"
        document_id: restrict to one document

    Returns:
        list of dicts: { clause, connected: [{ entity, type, relation }] }
    """
    if not settings.enable_knowledge_graph:
        return []

    try:
        from ai.knowledge_graph.graph_store import load_graph, load_global_graph
        from ai.knowledge_graph.graph_builder import find_connected_clauses

        graph = load_graph(document_id) if document_id else load_global_graph()
        if not graph:
            return []

        return find_connected_clauses(graph, clause_type)

    except Exception as e:
        logger.warning(f"get_clause_connections failed: {e}")
        return []


def get_cross_document_query(query: str) -> str:
    """
    Query across ALL ingested contracts using the global merged graph.
    Finds entities and relationships that span multiple contracts.

    Example: "Which contracts have payment penalties?"
    → traverses global graph → finds all penalty nodes across all documents

    Args:
        query: natural language query

    Returns:
        Formatted graph context string
    """
    return query_graph(query, document_id=None)


def delete_document_graph(document_id: str) -> None:
    """
    Delete graph for a document when it is removed.
    Called by vector_store.delete_document().
    """
    if not settings.enable_knowledge_graph:
        return

    try:
        from ai.knowledge_graph.graph_store import delete_graph
        delete_graph(document_id)
        logger.info(f"Graph deleted for doc={document_id}")
    except Exception as e:
        logger.warning(f"Graph deletion failed: {e}")


def get_graph_stats(document_id: Optional[str] = None) -> dict:
    """
    Return graph statistics for a document or all documents.
    Used by health check and admin endpoints.
    """
    if not settings.enable_knowledge_graph:
        return {"enabled": False}

    try:
        from ai.knowledge_graph.graph_store import get_graph_stats, load_graph
        stats = get_graph_stats()

        if document_id:
            g = load_graph(document_id)
            stats["document"] = {
                "document_id": document_id,
                "nodes":       g.number_of_nodes() if g else 0,
                "edges":       g.number_of_edges() if g else 0,
            }

        stats["enabled"] = True
        return stats

    except Exception as e:
        return {"enabled": True, "error": str(e)}
