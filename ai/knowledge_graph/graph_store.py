"""
ai/knowledge_graph/graph_store.py

Persists NetworkX graphs to disk as JSON files.
One graph file per document: data/knowledge_graphs/{document_id}.json

Also maintains a global graph (all_documents.json) that merges
entities across all contracts — used for cross-document queries.

Location: ai/knowledge_graph/graph_store.py
"""

import json
import logging
from pathlib import Path
from typing import Optional

import networkx as nx
from networkx.readwrite import json_graph

from ai.config import settings

logger = logging.getLogger(__name__)


def _graph_dir() -> Path:
    p = Path(settings.knowledge_graph_dir)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _doc_path(document_id: str) -> Path:
    return _graph_dir() / f"{document_id}.json"


def _global_path() -> Path:
    return _graph_dir() / "global_graph.json"


def save_graph(document_id: str, graph: nx.DiGraph) -> None:
    """
    Save a document's NetworkX graph to disk as JSON.

    Args:
        document_id: MongoDB document ID
        graph:       NetworkX directed graph
    """
    path = _doc_path(document_id)
    data = json_graph.node_link_data(graph)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    logger.info(
        f"Graph saved: doc={document_id}  "
        f"nodes={graph.number_of_nodes()}  "
        f"edges={graph.number_of_edges()}"
    )
    _update_global_graph(document_id, graph)


def load_graph(document_id: str) -> Optional[nx.DiGraph]:
    """
    Load a document's graph from disk.

    Args:
        document_id: MongoDB document ID

    Returns:
        NetworkX DiGraph, or None if not found
    """
    path = _doc_path(document_id)
    if not path.exists():
        logger.info(f"No graph found for doc={document_id}")
        return None

    with open(path, "r") as f:
        data = json.load(f)

    graph = json_graph.node_link_graph(data, directed=True)
    logger.info(
        f"Graph loaded: doc={document_id}  "
        f"nodes={graph.number_of_nodes()}  "
        f"edges={graph.number_of_edges()}"
    )
    return graph


def delete_graph(document_id: str) -> None:
    """
    Delete a document's graph from disk and remove from global graph.

    Args:
        document_id: MongoDB document ID
    """
    path = _doc_path(document_id)
    if path.exists():
        path.unlink()
        logger.info(f"Graph deleted: doc={document_id}")

    # Rebuild global graph without this document
    _rebuild_global_graph()


def load_global_graph() -> nx.DiGraph:
    """
    Load the merged graph across all documents.
    Returns empty graph if none exists yet.
    """
    path = _global_path()
    if not path.exists():
        return nx.DiGraph()

    with open(path, "r") as f:
        data = json.load(f)

    return json_graph.node_link_graph(data, directed=True)


def _update_global_graph(document_id: str, new_graph: nx.DiGraph) -> None:
    """
    Merge a document's graph into the global graph.
    Prefixes all node IDs with document_id to avoid collisions.
    """
    global_graph = load_global_graph()

    # Add nodes with document_id prefix
    for node_id, attrs in new_graph.nodes(data=True):
        global_id = f"{document_id}:{node_id}"
        node_attrs = {k: v for k, v in attrs.items() if k != "document_id"}
        node_attrs["document_id"] = document_id
        global_graph.add_node(global_id, **node_attrs)

    # Add edges with prefixed IDs
    for src, tgt, attrs in new_graph.edges(data=True):
        g_src = f"{document_id}:{src}"
        g_tgt = f"{document_id}:{tgt}"
        global_graph.add_edge(g_src, g_tgt, **attrs)

    # Save global graph
    data = json_graph.node_link_data(global_graph)
    with open(_global_path(), "w") as f:
        json.dump(data, f, indent=2)

    logger.info(
        f"Global graph updated: "
        f"total_nodes={global_graph.number_of_nodes()}  "
        f"total_edges={global_graph.number_of_edges()}"
    )


def _rebuild_global_graph() -> None:
    """Rebuild global graph from all remaining document graphs."""
    global_graph = nx.DiGraph()

    for path in _graph_dir().glob("*.json"):
        if path.name == "global_graph.json":
            continue

        document_id = path.stem
        try:
            with open(path, "r") as f:
                data = json.load(f)
            g = json_graph.node_link_graph(data, directed=True)

            for node_id, attrs in g.nodes(data=True):
                global_graph.add_node(
                    f"{document_id}:{node_id}",
                    **attrs,
                    document_id=document_id
                )
            for src, tgt, attrs in g.edges(data=True):
                global_graph.add_edge(
                    f"{document_id}:{src}",
                    f"{document_id}:{tgt}",
                    **attrs
                )
        except Exception as e:
            logger.warning(f"Failed to load graph for {document_id}: {e}")

    data = json_graph.node_link_data(global_graph)
    with open(_global_path(), "w") as f:
        json.dump(data, f, indent=2)


def get_graph_stats() -> dict:
    """Return stats about all stored graphs."""
    graph_dir = _graph_dir()
    doc_graphs = [p for p in graph_dir.glob("*.json") if p.name != "global_graph.json"]
    global_graph = load_global_graph()

    return {
        "total_documents": len(doc_graphs),
        "global_nodes":    global_graph.number_of_nodes(),
        "global_edges":    global_graph.number_of_edges(),
        "graph_dir":       str(graph_dir),
    }
