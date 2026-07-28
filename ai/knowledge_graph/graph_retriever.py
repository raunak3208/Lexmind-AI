"""
ai/knowledge_graph/graph_retriever.py

Graph-based retrieval — traverses the knowledge graph to find
relevant entities and relationships for a query.

Two strategies:
  1. entity_search    — find nodes by name/type matching the query keywords
  2. neighbourhood    — expand from matched nodes to get context subgraph

Returns graph_context string — structured text describing relevant
graph nodes and edges — passed to Mistral alongside vector chunks.

Location: ai/knowledge_graph/graph_retriever.py
"""

import re
import logging
from typing import Optional

import networkx as nx

from ai.knowledge_graph.graph_store import load_graph, load_global_graph
from ai.knowledge_graph.graph_builder import (
    get_entity_neighbourhood,
    get_party_obligations,
    find_connected_clauses,
)

logger = logging.getLogger(__name__)

# Keywords that suggest what type of graph query is needed
PARTY_KEYWORDS    = {"party", "parties", "who", "client", "vendor", "consultant", "employer", "employee"}
OBLIGATION_KEYWORDS = {"obligation", "must", "shall", "required", "duty", "responsible"}
CLAUSE_KEYWORDS   = {"clause", "section", "provision", "term", "article"}
AMOUNT_KEYWORDS   = {"payment", "fee", "amount", "cost", "price", "penalty", "damages"}
DATE_KEYWORDS     = {"date", "deadline", "period", "term", "expiry", "effective", "when", "notice"}


def _classify_query(query: str) -> list[str]:
    """Classify query intent to guide graph traversal strategy."""
    lower = query.lower()
    intents = []
    if any(kw in lower for kw in PARTY_KEYWORDS):       intents.append("party")
    if any(kw in lower for kw in OBLIGATION_KEYWORDS):  intents.append("obligation")
    if any(kw in lower for kw in CLAUSE_KEYWORDS):      intents.append("clause")
    if any(kw in lower for kw in AMOUNT_KEYWORDS):      intents.append("amount")
    if any(kw in lower for kw in DATE_KEYWORDS):        intents.append("date")
    return intents or ["general"]


def _extract_entity_mentions(query: str, graph: nx.DiGraph) -> list[str]:
    """
    Find entity names from the graph that appear in the query.
    Case-insensitive substring match.
    """
    mentioned = []
    query_lower = query.lower()
    for _, attrs in graph.nodes(data=True):
        name = attrs.get("name", "")
        if len(name) > 2 and name.lower() in query_lower:
            mentioned.append(name)
    return mentioned


def _graph_to_context_string(subgraph: nx.DiGraph, max_nodes: int = 20) -> str:
    """
    Convert a NetworkX subgraph to a readable context string for the LLM.

    Format:
      ENTITIES:
        [PARTY] Acme Corp — The Client party
        [CLAUSE] Payment Clause — Payment terms section
      RELATIONSHIPS:
        Acme Corp --HAS_OBLIGATION--> Pay $5000/month
        Payment Clause --DUE_WITHIN--> 30 days
    """
    if subgraph.number_of_nodes() == 0:
        return ""

    lines = ["GRAPH CONTEXT (entities and relationships from knowledge graph):"]

    lines.append("\nENTITIES:")
    nodes = list(subgraph.nodes(data=True))[:max_nodes]
    for _, attrs in nodes:
        etype = attrs.get("entity_type", "OTHER")
        name  = attrs.get("name", "")
        desc  = attrs.get("description", "")
        line  = f"  [{etype}] {name}"
        if desc:
            line += f" — {desc}"
        lines.append(line)

    lines.append("\nRELATIONSHIPS:")
    edges = list(subgraph.edges(data=True))
    for src, tgt, attrs in edges:
        src_name = subgraph.nodes[src].get("name", src)
        tgt_name = subgraph.nodes[tgt].get("name", tgt)
        rel_type = attrs.get("relation_type", "CONNECTED_TO")
        label    = attrs.get("label", "")
        line = f"  {src_name} --{rel_type}--> {tgt_name}"
        if label:
            line += f" ({label})"
        lines.append(line)

    return "\n".join(lines)


def retrieve_graph_context(
    query: str,
    document_id: Optional[str] = None,
    depth: int = 2,
) -> str:
    """
    Main graph retrieval function.
    Loads the graph, finds relevant nodes, returns context string.

    Args:
        query:       user question
        document_id: specific document (None = global graph)
        depth:       neighbourhood expansion depth

    Returns:
        Structured text string describing relevant graph context
        Empty string if no graph found or no relevant nodes
    """
    # Load appropriate graph
    if document_id:
        graph = load_graph(document_id)
    else:
        graph = load_global_graph()

    if graph is None or graph.number_of_nodes() == 0:
        logger.info(f"No graph available for doc={document_id or 'global'}")
        return ""

    intents = _classify_query(query)
    mentioned_entities = _extract_entity_mentions(query, graph)

    logger.info(
        f"Graph retrieval: intents={intents}  "
        f"mentioned={mentioned_entities}  "
        f"doc={document_id or 'global'}"
    )

    subgraphs = []

    # Strategy 1 — expand neighbourhood around mentioned entities
    for entity_name in mentioned_entities:
        sub = get_entity_neighbourhood(graph, entity_name, depth=depth)
        if sub.number_of_nodes() > 0:
            subgraphs.append(sub)

    # Strategy 2 — intent-based node lookup
    if "party" in intents and not mentioned_entities:
        party_nodes = [
            nid for nid, attrs in graph.nodes(data=True)
            if attrs.get("entity_type") == "PARTY"
        ]
        if party_nodes:
            party_sub = graph.subgraph(party_nodes).copy()
            subgraphs.append(party_sub)

    if "obligation" in intents:
        obligation_nodes = [
            nid for nid, attrs in graph.nodes(data=True)
            if attrs.get("entity_type") == "OBLIGATION"
        ]
        if obligation_nodes:
            subgraphs.append(graph.subgraph(obligation_nodes).copy())

    if "amount" in intents:
        amount_nodes = [
            nid for nid, attrs in graph.nodes(data=True)
            if attrs.get("entity_type") == "AMOUNT"
        ]
        if amount_nodes:
            subgraphs.append(graph.subgraph(amount_nodes).copy())

    if not subgraphs:
        # Fallback — return high-degree nodes (most connected = most important)
        degrees = sorted(graph.degree(), key=lambda x: x[1], reverse=True)
        top_nodes = [nid for nid, _ in degrees[:10]]
        subgraphs.append(graph.subgraph(top_nodes).copy())

    # Merge all subgraphs
    merged = nx.DiGraph()
    for sub in subgraphs:
        merged = nx.compose(merged, sub)

    context = _graph_to_context_string(merged)
    logger.info(
        f"Graph context: {merged.number_of_nodes()} nodes, "
        f"{len(context)} chars"
    )
    return context


def query_party_obligations(
    party_name: str,
    document_id: Optional[str] = None,
) -> list[dict]:
    """
    Get all obligations for a party — cross-document if no document_id.

    Args:
        party_name:  party to query
        document_id: specific document (None = all documents)

    Returns:
        list of obligation dicts
    """
    if document_id:
        graph = load_graph(document_id)
    else:
        graph = load_global_graph()

    if not graph:
        return []

    return get_party_obligations(graph, party_name)


def query_clause_connections(
    clause_type: str,
    document_id: Optional[str] = None,
) -> list[dict]:
    """
    Find clauses of a type and their connected entities.

    Args:
        clause_type: e.g. "payment", "termination"
        document_id: specific document (None = all documents)

    Returns:
        list of clause connection dicts
    """
    if document_id:
        graph = load_graph(document_id)
    else:
        graph = load_global_graph()

    if not graph:
        return []

    return find_connected_clauses(graph, clause_type)
