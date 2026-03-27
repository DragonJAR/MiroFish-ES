"""
Memory backend implementation using Zep Cloud SDK
"""

from typing import Dict, Any, List, Optional, Set

from zep_cloud.client import Zep
from zep_cloud import EpisodeData, EntityEdgeSourceTarget

from ..config import Config
from ..utils.logger import get_logger
from ..utils.zep_paging import fetch_all_nodes, fetch_all_edges
from .base import (
    EntityNode,
    FilteredEntities,
    SearchResult,
    GraphInfo,
    EpisodeResult,
    MemoryBackend,
)

logger = get_logger("mirofish.memory")


class ZepMemoryBackend(MemoryBackend):
    """Zep Cloud implementation of MemoryBackend"""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or Config.ZEP_API_KEY
        if not self.api_key:
            raise ValueError("ZEP_API_KEY no configurada")

        self.client = Zep(api_key=self.api_key)
        logger.info("ZepMemoryBackend inicializado")

    def search(
        self,
        query: str,
        graph_id: str,
        mode: str = "quick",
        limit: int = 10,
    ) -> SearchResult:
        """Search graph using Zep's hybrid search"""
        try:
            logger.info(
                f"Buscando en grafo: graph_id={graph_id}, query={query[:50]}..."
            )

            search_results = self.client.graph.search(
                graph_id=graph_id,
                query=query,
                limit=limit,
                scope=scope,
                reranker="cross_encoder",
            )

            facts = []
            edges = []
            nodes = []

            # Parse edge results
            if hasattr(search_results, "edges") and search_results.edges:
                for edge in search_results.edges:
                    if hasattr(edge, "fact") and edge.fact:
                        facts.append(edge.fact)
                    edges.append(
                        {
                            "uuid": getattr(edge, "uuid_", None)
                            or getattr(edge, "uuid", ""),
                            "name": getattr(edge, "name", ""),
                            "fact": getattr(edge, "fact", ""),
                            "source_node_uuid": getattr(edge, "source_node_uuid", ""),
                            "target_node_uuid": getattr(edge, "target_node_uuid", ""),
                        }
                    )

            # Parse node results
            if hasattr(search_results, "nodes") and search_results.nodes:
                for node in search_results.nodes:
                    nodes.append(
                        {
                            "uuid": getattr(node, "uuid_", None)
                            or getattr(node, "uuid", ""),
                            "name": getattr(node, "name", ""),
                            "labels": getattr(node, "labels", []),
                            "summary": getattr(node, "summary", ""),
                        }
                    )
                    if hasattr(node, "summary") and node.summary:
                        facts.append(f"[{node.name}]: {node.summary}")

            logger.info(f"Búsqueda completada: {len(facts)} hechos relacionados")

            return SearchResult(
                facts=facts,
                edges=edges,
                nodes=nodes,
                query=query,
                total_count=len(facts),
            )

        except Exception as e:
            logger.error(f"Búsqueda falló: {str(e)}")
            raise

    def get_entities(
        self, graph_id: str, entity_types: Optional[List[str]] = None
    ) -> List[EntityNode]:
        """Get all entities, optionally filtered by type"""
        try:
            logger.info(f"Obteniendo entidades del grafo {graph_id}...")

            all_nodes = fetch_all_nodes(self.client, graph_id)
            all_edges = fetch_all_edges(self.client, graph_id)

            # Create node map for edge resolution
            node_map = {n.uuid_: n.name for n in all_nodes}

            # Filter entities by type if specified
            filtered_entities = []

            for node in all_nodes:
                labels = node.labels or []

                # Filter out default labels
                custom_labels = [l for l in labels if l not in ["Entity", "Node"]]

                if not custom_labels:
                    continue

                # Filter by entity types if specified
                if entity_types:
                    if not any(t in custom_labels for t in entity_types):
                        continue
                    entity_type = next(t for t in custom_labels if t in entity_types)
                else:
                    entity_type = custom_labels[0]

                # Get related edges
                related_edges = []
                related_node_uuids = set()

                for edge in all_edges:
                    if edge.source_node_uuid == node.uuid_:
                        related_edges.append(
                            {
                                "direction": "outgoing",
                                "edge_name": edge.name,
                                "fact": edge.fact,
                                "target_node_uuid": edge.target_node_uuid,
                            }
                        )
                        related_node_uuids.add(edge.target_node_uuid)
                    elif edge.target_node_uuid == node.uuid_:
                        related_edges.append(
                            {
                                "direction": "incoming",
                                "edge_name": edge.name,
                                "fact": edge.fact,
                                "source_node_uuid": edge.source_node_uuid,
                            }
                        )
                        related_node_uuids.add(edge.source_node_uuid)

                # Get related nodes
                related_nodes = []
                for related_uuid in related_node_uuids:
                    related_name = node_map.get(related_uuid, "")
                    if related_name:
                        related_nodes.append(
                            {
                                "uuid": related_uuid,
                                "name": related_name,
                            }
                        )

                entity = EntityNode(
                    uuid=str(node.uuid_),
                    name=node.name or "",
                    labels=labels,
                    summary=node.summary or "",
                    attributes=node.attributes or {},
                    related_edges=related_edges,
                    related_nodes=related_nodes,
                )

                filtered_entities.append(entity)

            logger.info(f"Obtenidas {len(filtered_entities)} entidades")
            return filtered_entities

        except Exception as e:
            logger.error(f"Error al obtener entidades: {str(e)}")
            raise

    def get_entity_by_uuid(
        self, graph_id: str, entity_uuid: str
    ) -> Optional[EntityNode]:
        """Get a single entity by UUID"""
        try:
            node = self.client.graph.node.get(uuid_=entity_uuid)

            if not node:
                return None

            # Get edges
            edges_result = self.client.graph.node.get_entity_edges(
                node_uuid=entity_uuid
            )
            related_edges = []
            related_nodes = []

            if edges_result:
                for edge in edges_result:
                    related_edges.append(
                        {
                            "uuid": getattr(edge, "uuid_", None)
                            or getattr(edge, "uuid", ""),
                            "name": edge.name or "",
                            "fact": edge.fact or "",
                            "source_node_uuid": edge.source_node_uuid,
                            "target_node_uuid": edge.target_node_uuid,
                        }
                    )

            return EntityNode(
                uuid=str(getattr(node, "uuid_", None) or getattr(node, "uuid", "")),
                name=node.name or "",
                labels=node.labels or [],
                summary=node.summary or "",
                attributes=node.attributes or {},
                related_edges=related_edges,
                related_nodes=related_nodes,
            )

        except Exception as e:
            logger.error(f"Error al obtener entidad {entity_uuid}: {str(e)}")
            return None

    def get_edges(
        self, graph_id: str, entity_uuid: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get edges, optionally filtered by entity"""
        try:
            all_edges = fetch_all_edges(self.client, graph_id)

            if entity_uuid:
                # Filter by entity
                filtered = [
                    {
                        "uuid": edge.uuid_,
                        "name": edge.name,
                        "fact": edge.fact,
                        "source_node_uuid": edge.source_node_uuid,
                        "target_node_uuid": edge.target_node_uuid,
                        "attributes": edge.attributes or {},
                    }
                    for edge in all_edges
                    if edge.source_node_uuid == entity_uuid
                    or edge.target_node_uuid == entity_uuid
                ]
                return filtered
            else:
                return [
                    {
                        "uuid": edge.uuid_,
                        "name": edge.name,
                        "fact": edge.fact,
                        "source_node_uuid": edge.source_node_uuid,
                        "target_node_uuid": edge.target_node_uuid,
                        "attributes": edge.attributes or {},
                    }
                    for edge in all_edges
                ]

        except Exception as e:
            logger.error(f"Error al obtener bordes: {str(e)}")
            raise

    def add_episode(
        self, graph_id: str, data: str, episode_type: str = "text"
    ) -> EpisodeResult:
        """Add an episode to the graph"""
        try:
            episode = EpisodeData(data=data, type=episode_type)
            result = self.client.graph.add(graph_id=graph_id, episodes=[episode])

            return EpisodeResult(
                episode_uuid=str(result[0].uuid_) if result else "",
                status="created",
            )

        except Exception as e:
            logger.error(f"Error al agregar episode: {str(e)}")
            raise

    def create_graph(self, name: str, description: str = "") -> str:
        """Create a new graph"""
        try:
            import uuid

            graph_id = f"mirofish_{uuid.uuid4().hex[:16]}"

            self.client.graph.create(
                graph_id=graph_id,
                name=name,
                description=description,
            )

            logger.info(f"Grafo creado: {graph_id}")
            return graph_id

        except Exception as e:
            logger.error(f"Error al crear grafo: {str(e)}")
            raise

    def delete_graph(self, graph_id: str) -> bool:
        """Delete a graph"""
        try:
            self.client.graph.delete(graph_id=graph_id)
            logger.info(f"Grafo eliminado: {graph_id}")
            return True

        except Exception as e:
            logger.error(f"Error al eliminar grafo: {str(e)}")
            return False

    def get_graph_info(self, graph_id: str) -> Optional[GraphInfo]:
        """Get information about a graph"""
        try:
            nodes = fetch_all_nodes(self.client, graph_id)
            edges = fetch_all_edges(self.client, graph_id)

            # Get entity types
            entity_types = set()
            for node in nodes:
                if node.labels:
                    for label in node.labels:
                        if label not in ["Entity", "Node"]:
                            entity_types.add(label)

            return GraphInfo(
                graph_id=graph_id,
                node_count=len(nodes),
                edge_count=len(edges),
                entity_types=list(entity_types),
            )

        except Exception as e:
            logger.error(f"Error al obtener info del grafo: {str(e)}")
            return None
