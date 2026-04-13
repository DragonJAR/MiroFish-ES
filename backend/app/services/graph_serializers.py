"""
Serializadores compartidos para datos de grafo.
Evita duplicación entre GraphBuilderService y GraphitiGraphBuilder.
"""

from typing import Dict, Any, List


def serialize_nodes(nodes_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Serializar nodos al formato estándar de la API.

    Args:
        nodes_data: Lista de dicts con keys: uuid, name, labels, summary,
                   attributes (opcional), created_at (opcional)
    """
    result = []
    for n in nodes_data:
        # created_at puede venir como datetime o string
        created = n.get("created_at")
        if created and not isinstance(created, str):
            created = str(created)

        result.append(
            {
                "uuid": n.get("uuid", ""),
                "name": n.get("name", ""),
                "labels": n.get("labels") or [],
                "summary": n.get("summary") or "",
                "attributes": n.get("attributes") or {},
                "created_at": created,
            }
        )
    return result


def serialize_edges(edges_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Serializar bordes al formato estándar de la API.

    Args:
        edges_data: Lista de dicts con keys: uuid, name, fact, fact_type,
                   source_node_uuid, target_node_uuid, source_node_name,
                   target_node_name, attributes, created_at, valid_at,
                   invalid_at, expired_at (opcional), episodes (opcional)
    """
    result = []
    for e in edges_data:
        # Handle temporal fields
        def _str(v):
            return str(v) if v else None

        result.append(
            {
                "uuid": e.get("uuid", ""),
                "name": e.get("name", ""),
                "fact": e.get("fact", ""),
                "fact_type": e.get("fact_type") or e.get("name") or "",
                "source_node_uuid": e.get("source_node_uuid", ""),
                "target_node_uuid": e.get("target_node_uuid", ""),
                "source_node_name": e.get("source_node_name", ""),
                "target_node_name": e.get("target_node_name", ""),
                "attributes": e.get("attributes") or {},
                "created_at": _str(e.get("created_at")),
                "valid_at": _str(e.get("valid_at")),
                "invalid_at": _str(e.get("invalid_at")),
                "expired_at": _str(e.get("expired_at")),
                "episodes": e.get("episodes") or [],
            }
        )
    return result


def build_graph_data_response(
    graph_id: str, nodes_data: List[Dict], edges_data: List[Dict]
) -> Dict[str, Any]:
    """Construir respuesta estándar de datos de grafo."""
    return {
        "graph_id": graph_id,
        "nodes": serialize_nodes(nodes_data),
        "edges": serialize_edges(edges_data),
        "node_count": len(nodes_data),
        "edge_count": len(edges_data),
    }
