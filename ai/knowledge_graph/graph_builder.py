"""
ai/knowledge_graph/graph_builder.py

Builds a NetworkX directed graph from entity extraction results.

Each node = an Entity (party, clause, obligation, amount, date...)
Each edge = a Relation between two entities

Node attributes stored:
  name, entity_type, description, source_text, document_id

Edge attributes stored:
  relation_type, label, weight, document_id

Location: ai/knowledge_graph/graph_builder.py
"""

import logging
from typing import Optional

import networkx as nx

from ai.knowledge_graph.schemas import ExtractionResult, EntityType, RelationType
from ai.knowledge_graph.graph_store import save_graph

logger = logging.getLogger(__name__)


def build_graph(extraction: ExtractionResult) -> nx.DiGraph:
    """
    Build a NetworkX DiGraph from an ExtractionResult.

    Args:
        extraction: output from entity_extractor.extract_entities()

    Returns:
        NetworkX directed graph with entities as nodes, relations as edges
    """
    G = nx.DiGraph()
    G.graph["document_id"] = extraction.document_id
    G.graph["filename"]    = extraction.filename

    # Add nodes
    for entity in extraction.entities:
        G.add_node(
            entity.entity_id,
            name=entity.name,
            entity_type=entity.entity_type.value,
            description=entity.description or "",
            source_text=entity.source_text or "",
            document_id=extraction.document_id,
        )

    # Add edges
    for relation in extraction.relations:
        if relation.source_id in G and relation.target_id in G:
            G.add_edge(
                relation.source_id,
                relation.target_id,
                relation_id=relation.relation_id,
                relation_type=relation.relation_type.value,
                label=relation.label or "",
                weight=relation.weight,
                document_id=extraction.document_id,
            )

    logger.info(
        f"Graph built: doc={extraction.document_id}  "
        f"nodes={G.number_of_nodes()}  edges={G.number_of_edges()}"
    )
    return G


async def build_and_save_graph(
    contract_text: str,
    document_id: str,
    filename: str,
) -> nx.DiGraph:
    """
    Full pipeline: extract entities → build graph → save to disk.

    Args:
        contract_text: full contract text
        document_id:   MongoDB document ID
        filename:      original filename

    Returns:
        NetworkX DiGraph
    """
    from ai.knowledge_graph.entity_extractor import extract_entities

    logger.info(f"Building knowledge graph for doc={document_id}")

    extraction = await extract_entities(
        contract_text=contract_text,
        document_id=document_id,
        filename=filename,
    )

    graph = build_graph(extraction)
    save_graph(document_id, graph)

    logger.info(
        f"Knowledge graph ready: doc={document_id}  "
        f"entities={extraction.total_entities}  "
        f"relations={extraction.total_relations}"
    )
    return graph


def get_entity_neighbourhood(
    graph: nx.DiGraph,
    entity_name: str,
    depth: int = 2,
) -> nx.DiGraph:
    """
    Get a subgraph centred on a named entity up to a given depth.

    Example: get_entity_neighbourhood(G, "Acme Corp", depth=2)
    Returns all nodes within 2 hops of Acme Corp.

    Args:
        graph:       full document graph
        entity_name: entity name to centre on (case-insensitive)
        depth:       number of hops to traverse

    Returns:
        subgraph containing the neighbourhood
    """
    # Find node ID(s) matching the name
    target_ids = [
        nid for nid, attrs in graph.nodes(data=True)
        if attrs.get("name", "").lower() == entity_name.lower()
    ]

    if not target_ids:
        # Fuzzy match — partial name
        target_ids = [
            nid for nid, attrs in graph.nodes(data=True)
            if entity_name.lower() in attrs.get("name", "").lower()
        ]

    if not target_ids:
        logger.info(f"Entity '{entity_name}' not found in graph")
        return nx.DiGraph()

    # Collect all nodes within depth hops
    neighbourhood = set()
    for nid in target_ids:
        # Outgoing neighbours
        out_nodes = nx.single_source_shortest_path_length(
            graph, nid, cutoff=depth
        ).keys()
        # Incoming neighbours (reverse graph)
        rev = graph.reverse()
        in_nodes = nx.single_source_shortest_path_length(
            rev, nid, cutoff=depth
        ).keys()
        neighbourhood.update(out_nodes)
        neighbourhood.update(in_nodes)

    subgraph = graph.subgraph(neighbourhood).copy()
    logger.info(
        f"Neighbourhood of '{entity_name}' (depth={depth}): "
        f"{subgraph.number_of_nodes()} nodes, {subgraph.number_of_edges()} edges"
    )
    return subgraph


def get_party_obligations(
    graph: nx.DiGraph,
    party_name: str,
) -> list[dict]:
    """
    Get all obligations for a specific party from the graph.

    Args:
        graph:      document graph
        party_name: party name (e.g. "Acme Corp")

    Returns:
        list of obligation dicts with name, description, related nodes
    """
    obligations = []

    # Find party node(s)
    party_ids = [
        nid for nid, attrs in graph.nodes(data=True)
        if (attrs.get("entity_type") == "PARTY" and
            party_name.lower() in attrs.get("name", "").lower())
    ]

    for party_id in party_ids:
        # Get all HAS_OBLIGATION edges from this party
        for _, tgt, edge_attrs in graph.out_edges(party_id, data=True):
            if edge_attrs.get("relation_type") == "HAS_OBLIGATION":
                tgt_attrs = graph.nodes[tgt]
                obligations.append({
                    "party":       graph.nodes[party_id].get("name"),
                    "obligation":  tgt_attrs.get("name", ""),
                    "description": tgt_attrs.get("description", ""),
                    "label":       edge_attrs.get("label", ""),
                })

    return obligations


def find_connected_clauses(
    graph: nx.DiGraph,
    clause_type: str,
) -> list[dict]:
    """
    Find all clauses of a given type and their connected entities.

    Args:
        graph:       document graph
        clause_type: e.g. "payment", "termination", "confidentiality"

    Returns:
        list of clause dicts with connected entities
    """
    results = []

    for nid, attrs in graph.nodes(data=True):
        if (attrs.get("entity_type") == "CLAUSE" and
                clause_type.lower() in attrs.get("name", "").lower()):

            connected = []
            for _, tgt, edge_attrs in graph.out_edges(nid, data=True):
                connected.append({
                    "entity":   graph.nodes[tgt].get("name"),
                    "type":     graph.nodes[tgt].get("entity_type"),
                    "relation": edge_attrs.get("label") or edge_attrs.get("relation_type"),
                })

            results.append({
                "clause":    attrs.get("name"),
                "connected": connected,
            })

    return results
