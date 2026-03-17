"""
Servicio de lectura y filtrado de entidades Zep
Lee nodos del grafo Zep, filtra nodos que coincidan con tipos de entidad predefinidos
"""

import time
from typing import Dict, Any, List, Optional, Set, Callable, TypeVar
from dataclasses import dataclass, field

from zep_cloud.client import Zep

from ..config import Config
from ..utils.logger import get_logger
from ..utils.zep_paging import fetch_all_nodes, fetch_all_edges

logger = get_logger('mirofish.zep_entity_reader')

# Para tipo de retorno generico
T = TypeVar('T')


@dataclass
class EntityNode:
    """Estructura de datos del nodo de entidad"""
    uuid: str
    name: str
    labels: List[str]
    summary: str
    attributes: Dict[str, Any]
    # Informacion de bordes relacionados
    related_edges: List[Dict[str, Any]] = field(default_factory=list)
    # Informacion de otros nodos relacionados
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


class ZepEntityReader:
    """
    Servicio de lectura y filtrado de entidades Zep
    
    Funcionalidades principales:
    1. Leer todos los nodos del grafo Zep
    2. Filtrar nodos que coincidan con tipos de entidad predefinidos (Labels que no solo sean Entity)
    3. Obtener informacion de bordes y nodos relacionados de cada entidad
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or Config.ZEP_API_KEY
        if not self.api_key:
            raise ValueError("ZEP_API_KEY no configurada")
        
        self.client = Zep(api_key=self.api_key)
    
    def _call_with_retry(
        self, 
        func: Callable[[], T], 
        operation_name: str,
        max_retries: int = 3,
        initial_delay: float = 2.0
    ) -> T:
        """
        Llamada a API Zep con mecanismo de reintento
        
        Args:
            func: Funcion a ejecutar (lambda o callable sin parametros)
            operation_name: Nombre de la operacion para logs
            max_retries: Numero maximo de reintentos (por defecto 3)
            initial_delay: Delay inicial en segundos
            
        Returns:
            Resultado de la llamada API
        """
        last_exception = None
        delay = initial_delay
        
        for attempt in range(max_retries):
            try:
                return func()
            except Exception as e:
                last_exception = e
                if attempt < max_retries - 1:
                    logger.warning(
                        f"Zep {operation_name} intento {attempt + 1} fallo: {str(e)[:100]}, "
                        f"reintentando en {delay:.1f}s..."
                    )
                    time.sleep(delay)
                    delay *= 2  # Exponential backoff
                else:
                    logger.error(f"Zep {operation_name} fallo despues de {max_retries} intentos: {str(e)}")
        
        raise last_exception
    
    def get_all_nodes(self, graph_id: str) -> List[Dict[str, Any]]:
        """
        Obtener todos los nodos del grafo (con paginacion)

        Args:
            graph_id: ID del grafo

        Returns:
            Lista de nodos
        """
        logger.info(f"Obtener todos los nodos del grafo {graph_id}...")

        nodes = fetch_all_nodes(self.client, graph_id)

        nodes_data = []
        for node in nodes:
            nodes_data.append({
                "uuid": getattr(node, 'uuid_', None) or getattr(node, 'uuid', ''),
                "name": node.name or "",
                "labels": node.labels or [],
                "summary": node.summary or "",
                "attributes": node.attributes or {},
            })

        logger.info(f"Se obtuvieron {len(nodes_data)} nodos")
        return nodes_data

    def get_all_edges(self, graph_id: str) -> List[Dict[str, Any]]:
        """
        Obtener todos los bordes del grafo (con paginacion)

        Args:
            graph_id: ID del grafo

        Returns:
            Lista de bordes
        """
        logger.info(f"Obtener todos los bordes del grafo {graph_id}...")

        edges = fetch_all_edges(self.client, graph_id)

        edges_data = []
        for edge in edges:
            edges_data.append({
                "uuid": getattr(edge, 'uuid_', None) or getattr(edge, 'uuid', ''),
                "name": edge.name or "",
                "fact": edge.fact or "",
                "source_node_uuid": edge.source_node_uuid,
                "target_node_uuid": edge.target_node_uuid,
                "attributes": edge.attributes or {},
            })

        logger.info(f"Se obtuvieron {len(edges_data)} bordes")
        return edges_data
    
    def get_node_edges(self, node_uuid: str) -> List[Dict[str, Any]]:
        """
        Obtener todos los bordes relacionados de un nodo especifico (con mecanismo de reintento)
        
        Args:
            node_uuid: UUID del nodo
            
        Returns:
            Lista de bordes
        """
        try:
            # Usar mecanismo de reintento para llamar a Zep API
            edges = self._call_with_retry(
                func=lambda: self.client.graph.node.get_entity_edges(node_uuid=node_uuid),
                operation_name=f"Obtener bordes de nodo(node={node_uuid[:8]}...)"
            )
            
            edges_data = []
            for edge in edges:
                edges_data.append({
                    "uuid": getattr(edge, 'uuid_', None) or getattr(edge, 'uuid', ''),
                    "name": edge.name or "",
                    "fact": edge.fact or "",
                    "source_node_uuid": edge.source_node_uuid,
                    "target_node_uuid": edge.target_node_uuid,
                    "attributes": edge.attributes or {},
                })
            
            return edges_data
        except Exception as e:
            logger.warning(f"Error al obtener bordes del nodo {node_uuid}: {str(e)}")
            return []
    
    def filter_defined_entities(
        self, 
        graph_id: str,
        defined_entity_types: Optional[List[str]] = None,
        enrich_with_edges: bool = True
    ) -> FilteredEntities:
        """
        Filtrar nodos que coincidan con tipos de entidad predefinidos
        
        Logica de filtrado:
        - Si el nodo solo tiene una etiqueta "Entity", significa que no coincide con nuestros tipos predefinidos, omitir
        - Si el nodo tiene etiquetas adicionales a "Entity" y "Node", coincide con tipos predefinidos, mantener
        
        Args:
            graph_id: ID del grafo
            defined_entity_types: Lista de tipos de entidad predefinidos (opcional, si se proporciona solo mantiene esos tipos)
            enrich_with_edges: Si obtener informacion de bordes relacionados de cada entidad
            
        Returns:
            FilteredEntities: Conjunto de entidades filtradas
        """
        logger.info(f"Iniciar filtrado de entidades del grafo {graph_id}...")
        
        # Obtener todos los nodos
        all_nodes = self.get_all_nodes(graph_id)
        total_count = len(all_nodes)
        
        # Obtener todos los bordes (para busqueda posterior)
        all_edges = self.get_all_edges(graph_id) if enrich_with_edges else []
        
        # Crear mapa de UUID de nodo a datos de nodo
        node_map = {n["uuid"]: n for n in all_nodes}
        
        # Filtrar entidades que cumplan condiciones
        filtered_entities = []
        entity_types_found = set()
        
        for node in all_nodes:
            labels = node.get("labels", [])
            
            # Logica de filtrado: Labels deben contener etiquetas distintas de "Entity" y "Node"
            custom_labels = [l for l in labels if l not in ["Entity", "Node"]]
            
            if not custom_labels:
                # Solo tiene etiquetas predeterminadas, omitir
                continue
            
            # Si se especificaron tipos predefinidos, verificar si coincide
            if defined_entity_types:
                matching_labels = [l for l in custom_labels if l in defined_entity_types]
                if not matching_labels:
                    continue
                entity_type = matching_labels[0]
            else:
                entity_type = custom_labels[0]
            
            entity_types_found.add(entity_type)
            
            # Crear objeto de nodo de entidad
            entity = EntityNode(
                uuid=node["uuid"],
                name=node["name"],
                labels=labels,
                summary=node["summary"],
                attributes=node["attributes"],
            )
            
            # Obtener bordes y nodos relacionados
            if enrich_with_edges:
                related_edges = []
                related_node_uuids = set()
                
                for edge in all_edges:
                    if edge["source_node_uuid"] == node["uuid"]:
                        related_edges.append({
                            "direction": "outgoing",
                            "edge_name": edge["name"],
                            "fact": edge["fact"],
                            "target_node_uuid": edge["target_node_uuid"],
                        })
                        related_node_uuids.add(edge["target_node_uuid"])
                    elif edge["target_node_uuid"] == node["uuid"]:
                        related_edges.append({
                            "direction": "incoming",
                            "edge_name": edge["name"],
                            "fact": edge["fact"],
                            "source_node_uuid": edge["source_node_uuid"],
                        })
                        related_node_uuids.add(edge["source_node_uuid"])
                
                entity.related_edges = related_edges
                
                # Obtener informacion basica de nodos relacionados
                related_nodes = []
                for related_uuid in related_node_uuids:
                    if related_uuid in node_map:
                        related_node = node_map[related_uuid]
                        related_nodes.append({
                            "uuid": related_node["uuid"],
                            "name": related_node["name"],
                            "labels": related_node["labels"],
                            "summary": related_node.get("summary", ""),
                        })
                
                entity.related_nodes = related_nodes
            
            filtered_entities.append(entity)
        
        logger.info(f"Filtrado completado: Total de nodos {total_count}, que cumplen condicion {len(filtered_entities)}, "
                   f"Tipos de entidad: {entity_types_found}")
        
        return FilteredEntities(
            entities=filtered_entities,
            entity_types=entity_types_found,
            total_count=total_count,
            filtered_count=len(filtered_entities),
        )
    
    def get_entity_with_context(
        self, 
        graph_id: str, 
        entity_uuid: str
    ) -> Optional[EntityNode]:
        """
        Obtener una entidad y su contexto completo (bordes y nodos relacionados, con reintento)
        
        Args:
            graph_id: ID del grafo
            entity_uuid: UUID de la entidad
            
        Returns:
            EntityNode o None
        """
        try:
            # Usar mecanismo de reintento para obtener nodo
            node = self._call_with_retry(
                func=lambda: self.client.graph.node.get(uuid_=entity_uuid),
                operation_name=f"Obtener detalle de nodo(uuid={entity_uuid[:8]}...)"
            )
            
            if not node:
                return None
            
            # Obtener bordes del nodo
            edges = self.get_node_edges(entity_uuid)
            
            # Obtener todos los nodos para busqueda de relaciones
            all_nodes = self.get_all_nodes(graph_id)
            node_map = {n["uuid"]: n for n in all_nodes}
            
            # Procesar bordes y nodos relacionados
            related_edges = []
            related_node_uuids = set()
            
            for edge in edges:
                if edge["source_node_uuid"] == entity_uuid:
                    related_edges.append({
                        "direction": "outgoing",
                        "edge_name": edge["name"],
                        "fact": edge["fact"],
                        "target_node_uuid": edge["target_node_uuid"],
                    })
                    related_node_uuids.add(edge["target_node_uuid"])
                else:
                    related_edges.append({
                        "direction": "incoming",
                        "edge_name": edge["name"],
                        "fact": edge["fact"],
                        "source_node_uuid": edge["source_node_uuid"],
                    })
                    related_node_uuids.add(edge["source_node_uuid"])
            
            # Obtener informacion de nodos relacionados
            related_nodes = []
            for related_uuid in related_node_uuids:
                if related_uuid in node_map:
                    related_node = node_map[related_uuid]
                    related_nodes.append({
                        "uuid": related_node["uuid"],
                        "name": related_node["name"],
                        "labels": related_node["labels"],
                        "summary": related_node.get("summary", ""),
                    })
            
            return EntityNode(
                uuid=getattr(node, 'uuid_', None) or getattr(node, 'uuid', ''),
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
    
    def get_entities_by_type(
        self, 
        graph_id: str, 
        entity_type: str,
        enrich_with_edges: bool = True
    ) -> List[EntityNode]:
        """
        Obtener todas las entidades de un tipo especifico
        
        Args:
            graph_id: ID del grafo
            entity_type: Tipo de entidad (ej. "Student", "PublicFigure")
            enrich_with_edges: Si obtener informacion de bordes relacionados
            
        Returns:
            Lista de entidades
        """
        result = self.filter_defined_entities(
            graph_id=graph_id,
            defined_entity_types=[entity_type],
            enrich_with_edges=enrich_with_edges
        )
        return result.entities

