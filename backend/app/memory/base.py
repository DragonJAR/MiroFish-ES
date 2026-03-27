"""
Interfaz base y estructuras de datos compartidas para backends de memoria
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Set


@dataclass
class EntityNode:
    """Estructura de datos del nodo de entidad"""

    uuid: str
    name: str
    labels: List[str]
    summary: str
    attributes: Dict[str, Any]
    # Información de bordes relacionados
    related_edges: List[Dict[str, Any]] = field(default_factory=list)
    # Información de otros nodos relacionados
    related_nodes: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "uuid": self.uuid,
            "name": self.name,
            "labels": self.labels,
            "summary": self.summary,
            "attributes": self.attributes,
            "related_edges": self.related_edges,
            "related_nodes": self.related_nodes,
        }

    def get_entity_type(self) -> Optional[str]:
        """Obtener tipo de entidad (excluir etiqueta predeterminada Entity)"""
        for label in self.labels:
            if label not in ["Entity", "Node"]:
                return label
        return None


@dataclass
class FilteredEntities:
    """Conjunto de entidades filtradas"""

    entities: List[EntityNode]
    entity_types: Set[str]
    total_count: int
    filtered_count: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entities": [e.to_dict() for e in self.entities],
            "entity_types": list(self.entity_types),
            "total_count": self.total_count,
            "filtered_count": self.filtered_count,
        }


@dataclass
class SearchResult:
    """Resultado de búsqueda"""

    facts: List[str]
    edges: List[Dict[str, Any]]
    nodes: List[Dict[str, Any]]
    query: str
    total_count: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "facts": self.facts,
            "edges": self.edges,
            "nodes": self.nodes,
            "query": self.query,
            "total_count": self.total_count,
        }

    def to_text(self) -> str:
        """Convertir a formato de texto, para comprensión del LLM"""
        text_parts = [
            f"Consulta de búsqueda: {self.query}",
            f"Se encontraron {self.total_count} elementos relacionados",
        ]

        if self.facts:
            text_parts.append("\n### Hechos relacionados:")
            for i, fact in enumerate(self.facts, 1):
                text_parts.append(f"{i}. {fact}")

        return "\n".join(text_parts)


@dataclass
class GraphInfo:
    """Información del grafo"""

    graph_id: str
    node_count: int
    edge_count: int
    entity_types: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "graph_id": self.graph_id,
            "node_count": self.node_count,
            "edge_count": self.edge_count,
            "entity_types": self.entity_types,
        }


@dataclass
class EpisodeResult:
    """Resultado de agregar episodio"""

    episode_uuid: str
    nodes_created: int = 0
    edges_created: int = 0
    status: str = "completed"
    created_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "episode_uuid": self.episode_uuid,
            "nodes_created": self.nodes_created,
            "edges_created": self.edges_created,
            "status": self.status,
            "created_at": self.created_at,
        }


class MemoryBackend(ABC):
    """Interfaz abstracta para backends de memoria de grafo"""

    @abstractmethod
    def search(
        self,
        query: str,
        graph_id: str,
        mode: str = "quick",
        limit: int = 10,
    ) -> SearchResult:
        """
        Búsqueda en el grafo

        Args:
            query: Consulta de búsqueda
            graph_id: ID del grafo
            mode: Modo de búsqueda ("quick", "insight_forge", "panorama")
            limit: Número de resultados a devolver

        Returns:
            SearchResult: Resultado de búsqueda
        """
        pass

    @abstractmethod
    def get_entities(
        self,
        graph_id: str,
        entity_types: Optional[List[str]] = None,
        enrich_with_edges: bool = True,
    ) -> List[EntityNode]:
        """
        Obtener entidades del grafo

        Args:
            graph_id: ID del grafo
            entity_types: Tipos de entidad a filtrar (opcional)
            enrich_with_edges: Si incluir información de bordes relacionados

        Returns:
            Lista de EntityNode
        """
        pass

    @abstractmethod
    def get_entity_by_uuid(
        self,
        graph_id: str,
        uuid: str,
    ) -> Optional[EntityNode]:
        """
        Obtener una entidad por UUID

        Args:
            graph_id: ID del grafo
            uuid: UUID de la entidad

        Returns:
            EntityNode o None
        """
        pass

    @abstractmethod
    def get_edges(
        self,
        graph_id: str,
        entity_uuid: Optional[str] = None,
        include_temporal: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Obtener bordes del grafo

        Args:
            graph_id: ID del grafo
            entity_uuid: UUID de entidad para filtrar (opcional)
            include_temporal: Si incluir información temporal

        Returns:
            Lista de bordes
        """
        pass

    @abstractmethod
    def add_episode(
        self,
        graph_id: str,
        content: str,
        reference_time: Optional[str] = None,
        name: Optional[str] = None,
        source_type: str = "text",
    ) -> EpisodeResult:
        """
        Agregar episodio al grafo

        Args:
            graph_id: ID del grafo
            content: Contenido del episodio (texto o JSON)
            reference_time: Tiempo de referencia (opcional)
            name: Nombre del episodio (opcional)
            source_type: Tipo de fuente ("text" o "json")

        Returns:
            EpisodeResult: Resultado de agregar episodio
        """
        pass

    @abstractmethod
    def create_graph(
        self,
        name: str,
        ontology: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Crear nuevo grafo

        Args:
            name: Nombre del grafo
            ontology: Definición de ontología (opcional)

        Returns:
            ID del grafo creado
        """
        pass

    @abstractmethod
    def delete_graph(self, graph_id: str) -> bool:
        """
        Eliminar grafo

        Args:
            graph_id: ID del grafo

        Returns:
            True si se eliminó correctamente
        """
        pass

    @abstractmethod
    def build_indices(self) -> bool:
        """
        Construir índices de base de datos (solo relevante para Graphiti)

        Returns:
            True si se construyeron correctamente
        """
        pass
