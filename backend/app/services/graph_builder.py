"""
Servicio de construcción de grafos
Interfaz 2: Construcción de Standalone Graph usando Zep API
"""

import os
import uuid
import time
import threading
import warnings
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass

from ..config import Config
from ..models.task import TaskManager, TaskStatus
from ..utils.logger import get_logger
from .text_processor import TextProcessor
from ..memory import get_memory_backend

logger = get_logger(__name__)


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


class GraphBuilderService:
    """
    Servicio de construcción de grafos
    Responsable de llamar a Zep API para construir grafos de conocimiento
    """

    def __init__(self, backend=None):
        self.backend = backend or get_memory_backend()
        self.task_manager = TaskManager()

    def build_graph_async(
        self,
        text: str,
        ontology: Dict[str, Any],
        graph_name: str = "MiroFish Graph",
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        batch_size: int = 3,
    ) -> str:
        """
        Construir grafo de forma asíncrona

        Args:
            text: Texto de entrada
            ontology: Definición de ontología (proveniente del interfaz 1)
            graph_name: Nombre del grafo
            chunk_size: Tamaño del chunk de texto
            chunk_overlap: Tamaño de superposición de chunks
            batch_size: Cantidad de chunks a enviar por lote

        Returns:
            ID de tarea
        """
        # Crear tarea
        task_id = self.task_manager.create_task(
            task_type="graph_build",
            metadata={
                "graph_name": graph_name,
                "chunk_size": chunk_size,
                "text_length": len(text),
            },
        )

        # Ejecutar construcción en thread en background
        thread = threading.Thread(
            target=self._build_graph_worker,
            args=(
                task_id,
                text,
                ontology,
                graph_name,
                chunk_size,
                chunk_overlap,
                batch_size,
            ),
        )
        thread.daemon = True
        thread.start()

        return task_id

    def _build_graph_worker(
        self,
        task_id: str,
        text: str,
        ontology: Dict[str, Any],
        graph_name: str,
        chunk_size: int,
        chunk_overlap: int,
        batch_size: int,
    ):
        """Worker thread para construcción del grafo"""
        try:
            self.task_manager.update_task(
                task_id,
                status=TaskStatus.PROCESSING,
                progress=5,
                message="Iniciando construcción del grafo...",
            )

            # 1. Crear grafo
            graph_id = self.create_graph(graph_name)
            self.task_manager.update_task(
                task_id, progress=10, message=f"Grafo creado: {graph_id}"
            )

            # 2. Configurar ontología
            self.set_ontology(graph_id, ontology)
            self.task_manager.update_task(
                task_id, progress=15, message="Ontología configurada"
            )

            # 3. Dividir texto en chunks
            chunks = TextProcessor.split_text(text, chunk_size, chunk_overlap)
            total_chunks = len(chunks)
            self.task_manager.update_task(
                task_id, progress=20, message=f"Texto dividido en {total_chunks} chunks"
            )

            # 4. Enviar datos en lotes
            episode_uuids = self.add_text_batches(
                graph_id,
                chunks,
                batch_size,
                lambda msg, prog: self.task_manager.update_task(
                    task_id,
                    progress=20 + int(prog * 0.4),  # 20-60%
                    message=msg,
                ),
            )

            # 5. Esperar a que Zep termine el procesamiento
            self.task_manager.update_task(
                task_id,
                progress=60,
                message="Esperando procesamiento de datos por Zep...",
            )

            self._wait_for_episodes(
                episode_uuids,
                lambda msg, prog: self.task_manager.update_task(
                    task_id,
                    progress=60 + int(prog * 0.3),  # 60-90%
                    message=msg,
                ),
            )

            # 6. Obtener información del grafo
            self.task_manager.update_task(
                task_id, progress=90, message="Obteniendo información del grafo..."
            )

            graph_info = self._get_graph_info(graph_id)

            # Completar
            self.task_manager.complete_task(
                task_id,
                {
                    "graph_id": graph_id,
                    "graph_info": graph_info.to_dict(),
                    "chunks_processed": total_chunks,
                },
            )

        except Exception as e:
            import traceback

            error_msg = f"{str(e)}\n{traceback.format_exc()}"
            self.task_manager.fail_task(task_id, error_msg)

    def create_graph(self, name: str) -> str:
        """Crear grafo Zep (método público)"""
        graph_id = self.backend.create_graph(name=name, ontology=None)

        return graph_id

    def set_ontology(self, graph_id: str, ontology: Dict[str, Any]):
        """Configurar ontología del grafo (método público)"""
        if hasattr(self.backend, "set_ontology"):
            self.backend.set_ontology(graph_id, ontology)
        else:
            logger.warning(
                f"Backend {type(self.backend).__name__} no soporta set_ontology, omitiendo"
            )

        # Solo continuar si el backend tiene un cliente Zep (para set_ontology avanzado)
        if not hasattr(self.backend, "client"):
            logger.warning(
                f"Backend {type(self.backend).__name__} no soporta configuración de ontología avanzada de Zep, omitiendo"
            )
            return

        try:
            from zep_cloud import EntityEdgeSourceTarget
            from zep_cloud.model.entity_model import EntityModel
            from zep_cloud.model.edge_model import EdgeModel
            from zep_cloud.model.entity_text import EntityText
            from pydantic import Field
        except ImportError:
            logger.warning(
                "Zep Cloud SDK no disponible, omitiendo configuración de ontología avanzada"
            )
            return

        # Suprimir advertencias de Pydantic v2 sobre Field(default=None)
        warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

        # Nombres reservados por Zep, no pueden usarse como nombres de atributos
        RESERVED_NAMES = {
            "uuid",
            "name",
            "group_id",
            "name_embedding",
            "summary",
            "created_at",
        }

        def safe_attr_name(attr_name: str) -> str:
            """Convierte nombres reservados a nombres seguros"""
            if attr_name.lower() in RESERVED_NAMES:
                return f"entity_{attr_name}"
            return attr_name

        # Crear tipos de entidad dinámicamente
        entity_types = {}
        for entity_def in ontology.get("entity_types", []):
            name = entity_def["name"]
            description = entity_def.get("description", f"A {name} entity.")

            # Crear diccionario de atributos y anotaciones de tipo (Pydantic v2 necesita esto)
            attrs = {"__doc__": description}
            annotations = {}

            for attr_def in entity_def.get("attributes", []):
                attr_name = safe_attr_name(attr_def["name"])
                attr_desc = attr_def.get("description", attr_name)
                # Zep API necesita Field con description, esto es obligatorio
                attrs[attr_name] = Field(description=attr_desc, default=None)
                annotations[attr_name] = Optional[EntityText]

            attrs["__annotations__"] = annotations

            # Crear clase dinámicamente
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

        if entity_types or edge_definitions:
            self.backend.client.graph.set_ontology(
                graph_ids=[graph_id],
                entities=entity_types if entity_types else None,
                edges=edge_definitions if edge_definitions else None,
            )

    def add_text_batches(
        self,
        graph_id: str,
        chunks: List[str],
        batch_size: int = 3,
        progress_callback: Optional[Callable] = None,
    ) -> List[str]:
        """Agregar texto al grafo en lotes, devuelve lista de todos los UUIDs de episodes"""
        episode_uuids = []
        total_chunks = len(chunks)

        for i in range(0, total_chunks, batch_size):
            batch_chunks = chunks[i : i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (total_chunks + batch_size - 1) // batch_size

            if progress_callback:
                progress = (i + len(batch_chunks)) / total_chunks
                progress_callback(
                    f"Enviando lote {batch_num}/{total_batches} ({len(batch_chunks)} chunks)...",
                    progress,
                )

            try:
                if hasattr(self.backend, "client") and hasattr(
                    self.backend.client, "graph"
                ):
                    try:
                        from zep_cloud import EpisodeData

                        episodes = [
                            EpisodeData(data=chunk, type="text")
                            for chunk in batch_chunks
                        ]
                        batch_result = self.backend.client.graph.add_batch(
                            graph_id=graph_id, episodes=episodes
                        )
                    except ImportError:
                        batch_result = None
                elif hasattr(self.backend, "add_batch"):
                    batch_result = self.backend.add_batch(
                        graph_id=graph_id, episodes=batch_chunks
                    )
                else:
                    batch_result = []
                    for idx, chunk in enumerate(batch_chunks):
                        result = self.backend.add_episode(
                            graph_id=graph_id,
                            content=chunk,
                            reference_time=datetime.now(timezone.utc),
                            name=f"chunk-{i + idx}",
                            source_type="text",
                        )
                        batch_result.append(result)

                if batch_result and isinstance(batch_result, list):
                    for ep in batch_result:
                        ep_uuid = getattr(ep, "uuid_", None) or getattr(
                            ep, "uuid", None
                        )
                        if ep_uuid:
                            episode_uuids.append(ep_uuid)

                time.sleep(1)

            except Exception as e:
                if progress_callback:
                    progress_callback(f"Error al enviar lote {batch_num}: {str(e)}", 0)
                raise

        return episode_uuids

    def _wait_for_episodes(
        self,
        episode_uuids: List[str],
        progress_callback: Optional[Callable] = None,
        timeout: int = 600,
    ):
        """Esperar a que todos los episodes se procesen (consultando el estado 'processed' de cada episode)"""
        if not episode_uuids:
            if progress_callback:
                progress_callback("No hay episodes que esperar", 1.0)
            return

        if not hasattr(self.backend, "client"):
            if progress_callback:
                progress_callback(
                    "Backend no soporta verificación de estado de episodes, asumiendo procesamiento completado",
                    1.0,
                )
            return

        start_time = time.time()
        pending_episodes = set(episode_uuids)
        completed_count = 0
        total_episodes = len(episode_uuids)

        if progress_callback:
            progress_callback(
                f"Esperando procesamiento de {total_episodes} chunks de texto...", 0
            )

        while pending_episodes:
            if time.time() - start_time > timeout:
                if progress_callback:
                    progress_callback(
                        f"Timeout en algunos chunks, completados {completed_count}/{total_episodes}",
                        completed_count / total_episodes,
                    )
                break

            for ep_uuid in list(pending_episodes):
                try:
                    episode = self.backend.client.graph.episode.get(uuid_=ep_uuid)
                    is_processed = getattr(episode, "processed", False)

                    if is_processed:
                        pending_episodes.remove(ep_uuid)
                        completed_count += 1

                except Exception:
                    pass

            elapsed = int(time.time() - start_time)
            if progress_callback:
                progress_callback(
                    f"Procesando... {completed_count}/{total_episodes} completados, {len(pending_episodes)} pendientes ({elapsed}seg)",
                    completed_count / total_episodes if total_episodes > 0 else 0,
                )

            if pending_episodes:
                time.sleep(3)

        if progress_callback:
            progress_callback(
                f"Procesamiento completado: {completed_count}/{total_episodes}", 1.0
            )

    def _get_graph_info(self, graph_id: str) -> GraphInfo:
        """Obtener información del grafo"""
        nodes = self.backend.get_entities(graph_id=graph_id)
        edges = self.backend.get_edges(graph_id=graph_id)

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

    def get_graph_data(self, graph_id: str) -> Dict[str, Any]:
        """
        Obtener datos completos del grafo (con información detallada)

        Args:
            graph_id: ID del grafo

        Returns:
            Diccionario con nodes y edges, incluyendo información temporal, atributos y otros datos detallados
        """
        nodes = self.backend.get_entities(graph_id=graph_id)
        edges = self.backend.get_edges(graph_id=graph_id)

        node_map = {}
        for node in nodes:
            node_map[node.uuid_] = node.name or ""

        nodes_data = []
        for node in nodes:
            created_at = getattr(node, "created_at", None)
            if created_at:
                created_at = str(created_at)

            nodes_data.append(
                {
                    "uuid": node.uuid_,
                    "name": node.name,
                    "labels": node.labels or [],
                    "summary": node.summary or "",
                    "attributes": node.attributes or {},
                    "created_at": created_at,
                }
            )

        edges_data = []
        for edge in edges:
            created_at = getattr(edge, "created_at", None)
            valid_at = getattr(edge, "valid_at", None)
            invalid_at = getattr(edge, "invalid_at", None)
            expired_at = getattr(edge, "expired_at", None)

            episodes = getattr(edge, "episodes", None) or getattr(
                edge, "episode_ids", None
            )
            if episodes and not isinstance(episodes, list):
                episodes = [str(episodes)]
            elif episodes:
                episodes = [str(e) for e in episodes]

            fact_type = getattr(edge, "fact_type", None) or edge.name or ""

            edges_data.append(
                {
                    "uuid": edge.uuid_,
                    "name": edge.name or "",
                    "fact": edge.fact or "",
                    "fact_type": fact_type,
                    "source_node_uuid": edge.source_node_uuid,
                    "target_node_uuid": edge.target_node_uuid,
                    "source_node_name": node_map.get(edge.source_node_uuid, ""),
                    "target_node_name": node_map.get(edge.target_node_uuid, ""),
                    "attributes": edge.attributes or {},
                    "created_at": str(created_at) if created_at else None,
                    "valid_at": str(valid_at) if valid_at else None,
                    "invalid_at": str(invalid_at) if invalid_at else None,
                    "expired_at": str(expired_at) if expired_at else None,
                    "episodes": episodes or [],
                }
            )

        return {
            "graph_id": graph_id,
            "nodes": nodes_data,
            "edges": edges_data,
            "node_count": len(nodes_data),
            "edge_count": len(edges_data),
        }

    def delete_graph(self, graph_id: str):
        """Eliminar grafo"""
        self.backend.delete_graph(graph_id)
