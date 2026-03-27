"""
Wrapper para el backend Zep Cloud
Implementación thin de MemoryBackend usando Zep SDK existente
"""

import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from .base import MemoryBackend, SearchResult, EntityNode, GraphInfo, EpisodeResult
from ..config import Config
from ..utils.logger import get_logger
from ..utils.zep_paging import fetch_all_nodes, fetch_all_edges

logger = get_logger("mirofish.memory.zep_backend")


class ZepBackend(MemoryBackend):
    """
    Backend Zep Cloud

    Wrapper thin alrededor del SDK zep_cloud existente
    Reutiliza código de servicios existentes donde sea posible
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or Config.ZEP_API_KEY
        if not self.api_key:
            raise ValueError("ZEP_API_KEY no configurada")

        from zep_cloud.client import Zep

        self.client = Zep(api_key=self.api_key)
        logger.info("ZepBackend inicializado")

    def search(
        self,
        query: str,
        graph_id: str,
        mode: str = "quick",
        limit: int = 10,
    ) -> SearchResult:
        """
        Búsqueda en el grafo Zep

        Utiliza client.graph.search() con diferentes parámetros según el modo
        """
        logger.info(
            f"Búsqueda Zep: graph_id={graph_id}, query={query[:50]}..., mode={mode}"
        )

        try:
            scope = "edges" if mode in ["quick", "insight_forge"] else "both"
            reranker = "cross_encoder" if mode == "quick" else "rrf"

            search_results = self.client.graph.search(
                graph_id=graph_id,
                query=query,
                limit=limit,
                scope=scope,
                reranker=reranker,
            )

            facts = []
            edges = []
            nodes = []

            # Parsear resultados de aristas
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

            # Parsear resultados de nodos
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
                    # El resumen del nodo también cuenta como hecho
                    if hasattr(node, "summary") and node.summary:
                        facts.append(f"[{node.name}]: {node.summary}")

            logger.info(f"Búsqueda completada: {len(facts)} hechos encontrados")

            return SearchResult(
                facts=facts,
                edges=edges,
                nodes=nodes,
                query=query,
                total_count=len(facts),
            )

        except Exception as e:
            logger.error(f"Búsqueda Zep falló: {str(e)}")
            return SearchResult(query=query, total_count=0)

    def get_entities(
        self,
        graph_id: str,
        entity_types: Optional[List[str]] = None,
        enrich_with_edges: bool = True,
    ) -> List[EntityNode]:
        """
        Obtener entidades del grafo Zep

        Reutiliza fetch_all_nodes() para paginación automática
        Filtra por entity_types si se proporciona
        """
        logger.info(f"Obteniendo entidades de grafo {graph_id}...")

        nodes = fetch_all_nodes(self.client, graph_id)
        all_edges = fetch_all_edges(self.client, graph_id) if enrich_with_edges else []

        # Crear mapa de UUID de nodo a datos de nodo
        node_map = {n.uuid_: n for n in nodes}

        # Filtrar entidades
        entities = []
        for node in nodes:
            labels = node.labels or []
            custom_labels = [l for l in labels if l not in ["Entity", "Node"]]

            if not custom_labels:
                continue

            # Filtrar por tipos de entidad específicos si se proporciona
            if entity_types:
                matching_labels = [l for l in custom_labels if l in entity_types]
                if not matching_labels:
                    continue
                entity_type = matching_labels[0]
            else:
                entity_type = custom_labels[0]

            entity = EntityNode(
                uuid=node.uuid_,
                name=node.name or "",
                labels=labels,
                summary=node.summary or "",
                attributes=node.attributes or {},
            )

            # Enriquecer con bordes y nodos relacionados
            if enrich_with_edges:
                related_edges = []
                related_node_uuids = set()

                for edge in all_edges:
                    if edge.source_node_uuid == node.uuid_:
                        related_edges.append(
                            {
                                "direction": "outgoing",
                                "edge_name": edge.name or "",
                                "fact": edge.fact or "",
                                "target_node_uuid": edge.target_node_uuid,
                            }
                        )
                        related_node_uuids.add(edge.target_node_uuid)
                    elif edge.target_node_uuid == node.uuid_:
                        related_edges.append(
                            {
                                "direction": "incoming",
                                "edge_name": edge.name or "",
                                "fact": edge.fact or "",
                                "source_node_uuid": edge.source_node_uuid,
                            }
                        )
                        related_node_uuids.add(edge.source_node_uuid)

                entity.related_edges = related_edges

                # Obtener información de nodos relacionados
                related_nodes = []
                for uuid in related_node_uuids:
                    if uuid in node_map:
                        related_node = node_map[uuid]
                        related_nodes.append(
                            {
                                "uuid": related_node.uuid_,
                                "name": related_node.name or "",
                                "labels": related_node.labels or [],
                                "summary": related_node.summary or "",
                            }
                        )

                entity.related_nodes = related_nodes

            entities.append(entity)

        logger.info(f"Obtenidas {len(entities)} entidades")
        return entities

    def get_entity_by_uuid(
        self,
        graph_id: str,
        uuid: str,
    ) -> Optional[EntityNode]:
        """
        Obtener una entidad por UUID

        Utiliza client.graph.node.get() y enriquece con bordes
        """
        try:
            node = self.client.graph.node.get(uuid_=uuid)
            if not node:
                return None

            # Obtener bordes del nodo
            edges = self.client.graph.node.get_entity_edges(node_uuid=uuid)

            # Obtener todos los nodos para búsqueda de relaciones
            all_nodes = fetch_all_nodes(self.client, graph_id)
            node_map = {n.uuid_: n for n in all_nodes}

            # Procesar bordes y nodos relacionados
            related_edges = []
            related_node_uuids = set()

            for edge in edges:
                if edge.source_node_uuid == uuid:
                    related_edges.append(
                        {
                            "direction": "outgoing",
                            "edge_name": edge.name or "",
                            "fact": edge.fact or "",
                            "target_node_uuid": edge.target_node_uuid,
                        }
                    )
                    related_node_uuids.add(edge.target_node_uuid)
                else:
                    related_edges.append(
                        {
                            "direction": "incoming",
                            "edge_name": edge.name or "",
                            "fact": edge.fact or "",
                            "source_node_uuid": edge.source_node_uuid,
                        }
                    )
                    related_node_uuids.add(edge.source_node_uuid)

            # Obtener información de nodos relacionados
            related_nodes = []
            for related_uuid in related_node_uuids:
                if related_uuid in node_map:
                    related_node = node_map[related_uuid]
                    related_nodes.append(
                        {
                            "uuid": related_node.uuid_,
                            "name": related_node.name or "",
                            "labels": related_node.labels or [],
                            "summary": related_node.summary or "",
                        }
                    )

            return EntityNode(
                uuid=node.uuid_,
                name=node.name or "",
                labels=node.labels or [],
                summary=node.summary or "",
                attributes=node.attributes or {},
                related_edges=related_edges,
                related_nodes=related_nodes,
            )

        except Exception as e:
            logger.error(f"Error al obtener entidad {uuid}: {str(e)}")
            return None

    def get_edges(
        self,
        graph_id: str,
        entity_uuid: Optional[str] = None,
        include_temporal: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Obtener bordes del grafo

        Si entity_uuid se proporciona, filtra bordes relacionados con esa entidad
        """
        logger.info(f"Obteniendo bordes de grafo {graph_id}...")

        edges = fetch_all_edges(self.client, graph_id)

        edges_data = []
        for edge in edges:
            # Filtrar por entity_uuid si se proporciona
            if (
                entity_uuid
                and edge.source_node_uuid != entity_uuid
                and edge.target_node_uuid != entity_uuid
            ):
                continue

            edge_dict = {
                "uuid": getattr(edge, "uuid_", None) or getattr(edge, "uuid", ""),
                "name": edge.name or "",
                "fact": edge.fact or "",
                "source_node_uuid": edge.source_node_uuid,
                "target_node_uuid": edge.target_node_uuid,
                "attributes": edge.attributes or {},
            }

            # Agregar información temporal si se solicita
            if include_temporal:
                edge_dict["created_at"] = getattr(edge, "created_at", None)
                edge_dict["valid_at"] = getattr(edge, "valid_at", None)
                edge_dict["invalid_at"] = getattr(edge, "invalid_at", None)
                edge_dict["expired_at"] = getattr(edge, "expired_at", None)

            edges_data.append(edge_dict)

        logger.info(f"Obtenidos {len(edges_data)} bordes")
        return edges_data

    def add_episode(
        self,
        graph_id: str,
        content: str,
        reference_time: Optional[str] = None,
        name: Optional[str] = None,
        source_type: str = "text",
    ) -> EpisodeResult:
        """
        Agregar episodio al grafo Zep

        Utiliza client.graph.add() para episodios individuales
        """
        from zep_cloud import EpisodeData

        try:
            episode_type = "text" if source_type == "text" else "json"
            episode_data = EpisodeData(data=content, type=episode_type)

            result = self.client.graph.add(graph_id=graph_id, episodes=[episode_data])

            episode_uuid = getattr(result[0], "uuid_", None) or getattr(
                result[0], "uuid", ""
            )

            logger.info(f"Episodio agregado: {episode_uuid}")

            return EpisodeResult(
                episode_uuid=episode_uuid,
                status="completed",
            )

        except Exception as e:
            logger.error(f"Error al agregar episodio: {str(e)}")
            raise

    def create_graph(
        self,
        name: str,
        ontology: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Crear nuevo grafo Zep

        Utiliza client.graph.create()
        Si ontology se proporciona, configura la ontología
        """
        import uuid

        graph_id = f"mirofish_{uuid.uuid4().hex[:16]}"

        self.client.graph.create(
            graph_id=graph_id, name=name, description="MiroFish Social Simulation Graph"
        )

        # Configurar ontología si se proporciona
        if ontology:
            self.set_ontology(graph_id, ontology)

        logger.info(f"Grafo creado: {graph_id}")
        return graph_id

    def set_ontology(self, graph_id: str, ontology: Dict[str, Any]):
        """
        Configurar ontología del grafo

        Código adaptado de graph_builder.py
        """
        import warnings
        from typing import Optional
        from pydantic import Field
        from zep_cloud.external_clients.ontology import (
            EntityModel,
            EntityText,
            EdgeModel,
        )

        # Suprimir advertencias de Pydantic v2
        warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

        RESERVED_NAMES = {
            "uuid",
            "name",
            "group_id",
            "name_embedding",
            "summary",
            "created_at",
        }

        def safe_attr_name(attr_name: str) -> str:
            if attr_name.lower() in RESERVED_NAMES:
                return f"entity_{attr_name}"
            return attr_name

        # Crear tipos de entidad dinámicamente
        entity_types = {}
        for entity_def in ontology.get("entity_types", []):
            name = entity_def["name"]
            description = entity_def.get("description", f"A {name} entity.")

            attrs = {"__doc__": description}
            annotations = {}

            for attr_def in entity_def.get("attributes", []):
                attr_name = safe_attr_name(attr_def["name"])
                attr_desc = attr_def.get("description", attr_name)
                attrs[attr_name] = Field(description=attr_desc, default=None)
                annotations[attr_name] = Optional[EntityText]

            attrs["__annotations__"] = annotations
            entity_class = type(name, (EntityModel,), attrs)
            entity_class.__doc__ = description
            entity_types[name] = entity_class

        # Crear tipos de borde dinámicamente
        edge_definitions = {}
        for edge_def in ontology.get("edge_types", []):
            name = edge_def["name"]
            description = edge_def.get("description", f"A {name} relationship.")

            attrs = {"__doc__": description}
            annotations = {}

            for attr_def in edge_def.get("attributes", []):
                attr_name = safe_attr_name(attr_def["name"])
                attr_desc = attr_def.get("description", attr_name)
                attrs[attr_name] = Field(description=attr_desc, default=None)
                annotations[attr_name] = Optional[str]

            attrs["__annotations__"] = annotations

            class_name = "".join(word.capitalize() for word in name.split("_"))
            edge_class = type(class_name, (EdgeModel,), attrs)
            edge_class.__doc__ = description

            from zep_cloud.external_clients.ontology import EntityEdgeSourceTarget

            source_targets = []
            for st in edge_def.get("source_targets", []):
                source_targets.append(
                    EntityEdgeSourceTarget(
                        source=st.get("source", "Entity"),
                        target=st.get("target", "Entity"),
                    )
                )

            if source_targets:
                edge_definitions[name] = (edge_class, source_targets)

        # Llamar a Zep API para configurar ontología
        if entity_types or edge_definitions:
            self.client.graph.set_ontology(
                graph_ids=[graph_id],
                entities=entity_types if entity_types else None,
                edges=edge_definitions if edge_definitions else None,
            )

        logger.info(f"Ontología configurada para grafo {graph_id}")

    def delete_graph(self, graph_id: str) -> bool:
        """Eliminar grafo Zep"""
        try:
            self.client.graph.delete(graph_id=graph_id)
            logger.info(f"Grafo eliminado: {graph_id}")
            return True
        except Exception as e:
            logger.error(f"Error al eliminar grafo {graph_id}: {str(e)}")
            return False

    def build_indices(self) -> bool:
        """
        Zep no requiere construcción explícita de índices

        Zep maneja índices internamente, este método es un no-op
        """
        return True
