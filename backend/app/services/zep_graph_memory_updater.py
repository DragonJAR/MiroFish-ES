"""
Servicio de actualización de memoria del grafo
Actualiza dinámicamente las actividades de los Agents en el grafo durante la simulación
"""

import os
import time
import threading
import json
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
from datetime import datetime
from queue import Queue, Empty

from ..config import Config
from ..utils.logger import get_logger
from ..memory import get_memory_backend

logger = get_logger("mirofish.zep_graph_memory_updater")


@dataclass
class AgentActivity:
    """Registro de actividad de Agent"""

    platform: str  # twitter / reddit
    agent_id: int
    agent_name: str
    action_type: str  # CREATE_POST, LIKE_POST, etc.
    action_args: Dict[str, Any]
    round_num: int
    timestamp: str

    def to_episode_text(self) -> str:
        """
        Convertir actividad a descripción de texto que se puede enviar a Zep

        Usar formato de descripción en lenguaje natural para que Zep pueda extraer entidades y relaciones
        No agregar prefijo relacionado con simulación para evitar mislead la actualización del grafo
        """
        # Generar diferentes descripciones según el tipo de acción
        action_descriptions = {
            "CREATE_POST": self._describe_create_post,
            "LIKE_POST": self._describe_like_post,
            "DISLIKE_POST": self._describe_dislike_post,
            "REPOST": self._describe_repost,
            "QUOTE_POST": self._describe_quote_post,
            "FOLLOW": self._describe_follow,
            "CREATE_COMMENT": self._describe_create_comment,
            "LIKE_COMMENT": self._describe_like_comment,
            "DISLIKE_COMMENT": self._describe_dislike_comment,
            "SEARCH_POSTS": self._describe_search,
            "SEARCH_USER": self._describe_search_user,
            "MUTE": self._describe_mute,
        }

        describe_func = action_descriptions.get(
            self.action_type, self._describe_generic
        )
        description = describe_func()

        # Retornar formato "nombre del agent: descripción de actividad" directamente, sin prefijo de simulación
        return f"{self.agent_name}: {description}"

    def _describe_create_post(self) -> str:
        content = self.action_args.get("content", "")
        if content:
            return f"publicó un post: 「{content}」"
        return "publicó un post"

    def _describe_like_post(self) -> str:
        """Like al post - incluye contenido original del post y autor"""
        post_content = self.action_args.get("post_content", "")
        post_author = self.action_args.get("post_author_name", "")

        if post_content and post_author:
            return f"dio like al post de {post_author}: 「{post_content}」"
        elif post_content:
            return f"dio like a un post: 「{post_content}」"
        elif post_author:
            return f"dio like a un post de {post_author}"
        return "dio like a un post"

    def _describe_dislike_post(self) -> str:
        """Dislike al post - incluye contenido original del post y autor"""
        post_content = self.action_args.get("post_content", "")
        post_author = self.action_args.get("post_author_name", "")

        if post_content and post_author:
            return f"dio dislike al post de {post_author}: 「{post_content}」"
        elif post_content:
            return f"dio dislike a un post: 「{post_content}」"
        elif post_author:
            return f"dio dislike a un post de {post_author}"
        return "dio dislike a un post"

    def _describe_repost(self) -> str:
        """Republicar post - incluye contenido original y autor"""
        original_content = self.action_args.get("original_content", "")
        original_author = self.action_args.get("original_author_name", "")

        if original_content and original_author:
            return f"republicó el post de {original_author}: 「{original_content}」"
        elif original_content:
            return f"republicó un post: 「{original_content}」"
        elif original_author:
            return f"republicó un post de {original_author}"
        return "republicó un post"

    def _describe_quote_post(self) -> str:
        """Citar post - incluye contenido original, autor y comentario de cita"""
        original_content = self.action_args.get("original_content", "")
        original_author = self.action_args.get("original_author_name", "")
        quote_content = self.action_args.get(
            "quote_content", ""
        ) or self.action_args.get("content", "")

        base = ""
        if original_content and original_author:
            base = f"citó el post de {original_author}「{original_content}」"
        elif original_content:
            base = f"citó un post「{original_content}」"
        elif original_author:
            base = f"citó un post de {original_author}"
        else:
            base = "citó un post"

        if quote_content:
            base += f", y comentó: 「{quote_content}」"
        return base

    def _describe_follow(self) -> str:
        """Seguir usuario - incluye nombre del usuario seguido"""
        target_user_name = self.action_args.get("target_user_name", "")

        if target_user_name:
            return f"siguió al usuario「{target_user_name}」"
        return "siguió a un usuario"

    def _describe_create_comment(self) -> str:
        """Publicar comentario - incluye contenido y información del post comentado"""
        content = self.action_args.get("content", "")
        post_content = self.action_args.get("post_content", "")
        post_author = self.action_args.get("post_author_name", "")

        if content:
            if post_content and post_author:
                return f"comentó en el post de {post_author}「{post_content}」: 「{content}」"
            elif post_content:
                return f"comentó en el post「{post_content}」: 「{content}」"
            elif post_author:
                return f"comentó en el post de {post_author}: 「{content}」"
            return f"comentó: 「{content}」"
        return "publicó un comentario"

    def _describe_like_comment(self) -> str:
        """Like a comentario - incluye contenido y autor"""
        comment_content = self.action_args.get("comment_content", "")
        comment_author = self.action_args.get("comment_author_name", "")

        if comment_content and comment_author:
            return f"dio like al comentario de {comment_author}: 「{comment_content}」"
        elif comment_content:
            return f"dio like a un comentario: 「{comment_content}」"
        elif comment_author:
            return f"dio like a un comentario de {comment_author}"
        return "dio like a un comentario"

    def _describe_dislike_comment(self) -> str:
        """Dislike a comentario - incluye contenido y autor"""
        comment_content = self.action_args.get("comment_content", "")
        comment_author = self.action_args.get("comment_author_name", "")

        if comment_content and comment_author:
            return (
                f"dio dislike al comentario de {comment_author}: 「{comment_content}」"
            )
        elif comment_content:
            return f"dio dislike a un comentario: 「{comment_content}」"
        elif comment_author:
            return f"dio dislike a un comentario de {comment_author}"
        return "dio dislike a un comentario"

    def _describe_search(self) -> str:
        """Buscar posts - incluye palabras clave de búsqueda"""
        query = self.action_args.get("query", "") or self.action_args.get("keyword", "")
        return f"buscó「{query}」" if query else "realizó una búsqueda"

    def _describe_search_user(self) -> str:
        """Buscar usuario - incluye palabras clave de búsqueda"""
        query = self.action_args.get("query", "") or self.action_args.get(
            "username", ""
        )
        return f"buscó usuario「{query}」" if query else "buscó usuario"

    def _describe_mute(self) -> str:
        """Silenciar usuario - incluye nombre del usuario silenciado"""
        target_user_name = self.action_args.get("target_user_name", "")

        if target_user_name:
            return f"silenció al usuario「{target_user_name}」"
        return "silenció a un usuario"

    def _describe_generic(self) -> str:
        # Para tipos de acción desconocidos, generar descripción genérica
        return f"ejecutó operación {self.action_type}"


class ZepGraphMemoryUpdater:
    """
    Actualizador de memoria del grafo

    Monitorea los archivos de logs de acciones de la simulación, y actualiza las nuevas actividades de agentes al grafo en tiempo real.
    Agrupa por plataforma, cada BATCH_SIZE actividades se envían en lote.

    Todos los comportamientos significativos se actualizarán al grafo, action_args incluirá información de contexto completa:
    - Like/dislike del post original
    - República/cita del post original
    - Usuario seguido/silenciado
    - Like/dislike del comentario original
    """

    # Tamaño del lote (cuántas actividades acumular por plataforma antes de enviar)
    BATCH_SIZE = 5

    # Mapeo de nombres de plataformas (para mostrar en consola)
    PLATFORM_DISPLAY_NAMES = {
        "twitter": "mundo1",
        "reddit": "mundo2",
    }

    # Intervalo de envío (segundos), evitar demasiadas solicitudes rápidas
    SEND_INTERVAL = 0.5

    # Configuración de reintentos
    MAX_RETRIES = 3
    RETRY_DELAY = 2  # segundos

    def __init__(self, graph_id: str, api_key: Optional[str] = None, backend=None):
        """
        Inicializar actualizador

        Args:
            graph_id: ID del grafo
            api_key: API Key (opcional, por defecto leer de configuración)
            backend: Memory backend (opcional, por defecto usa get_memory_backend())
        """
        self.graph_id = graph_id
        self.api_key = api_key or Config.ZEP_API_KEY

        if not self.api_key:
            raise ValueError("API_KEY no configurada")

        self.backend = backend or get_memory_backend()

        # Cola de actividades
        self._activity_queue: Queue = Queue()

        # Buffer de actividades por plataforma (cada plataforma acumula hasta BATCH_SIZE antes de enviar en lote)
        self._platform_buffers: Dict[str, List[AgentActivity]] = {
            "twitter": [],
            "reddit": [],
        }
        self._buffer_lock = threading.Lock()

        # Bandera de control
        self._running = False
        self._worker_thread: Optional[threading.Thread] = None

        # Estadísticas
        self._total_activities = 0  # Actividades realmente añadidas a la cola
        self._total_sent = 0  # Lotes enviados exitosamente
        self._total_items_sent = 0  # Actividades enviadas exitosamente
        self._failed_count = 0  # Lotes de envío fallidos
        self._skipped_count = 0  # Actividades filtradas/saltadas (DO_NOTHING)

        logger.info(
            f"GraphMemoryUpdater inicializado: graph_id={graph_id}, batch_size={self.BATCH_SIZE}"
        )

    def _get_platform_display_name(self, platform: str) -> str:
        """Obtener nombre para mostrar de la plataforma"""
        return self.PLATFORM_DISPLAY_NAMES.get(platform.lower(), platform)

    def start(self):
        """Iniciar hilo de trabajo en segundo plano"""
        if self._running:
            return

        self._running = True
        self._worker_thread = threading.Thread(
            target=self._worker_loop,
            daemon=True,
            name=f"GraphMemoryUpdater-{self.graph_id[:8]}",
        )
        self._worker_thread.start()
        logger.info(f"GraphMemoryUpdater iniciado: graph_id={self.graph_id}")

    def stop(self):
        """Detener hilo de trabajo en segundo plano"""
        self._running = False

        # Enviar actividades restantes
        self._flush_remaining()

        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=10)

        logger.info(
            f"GraphMemoryUpdater detenido: graph_id={self.graph_id}, "
            f"total_activities={self._total_activities}, "
            f"batches_sent={self._total_sent}, "
            f"items_sent={self._total_items_sent}, "
            f"failed={self._failed_count}, "
            f"skipped={self._skipped_count}"
        )

    def add_activity(self, activity: AgentActivity):
        """
        Añadir una actividad de agent a la cola

        Todos los comportamientos significativos se añadirán a la cola, incluyendo:
        - CREATE_POST (publicar post)
        - CREATE_COMMENT (comentar)
        - QUOTE_POST (citr post)
        - SEARCH_POSTS (buscar posts)
        - SEARCH_USER (buscar usuario)
        - LIKE_POST/DISLIKE_POST (like/dislike a post)
        - REPOST (republicar)
        - FOLLOW (seguir)
        - MUTE (silenciar)
        - LIKE_COMMENT/DISLIKE_COMMENT (like/dislike a comentario)

        action_args incluirá información de contexto completa (como contenido del post, nombre de usuario, etc.).

        Args:
            activity: Registro de actividad de Agent
        """
        # Saltar actividades de tipo DO_NOTHING
        if activity.action_type == "DO_NOTHING":
            self._skipped_count += 1
            return

        self._activity_queue.put(activity)
        self._total_activities += 1
        logger.debug(
            f"Añadir actividad a cola: {activity.agent_name} - {activity.action_type}"
        )

    def add_activity_from_dict(self, data: Dict[str, Any], platform: str):
        """
        Añadir actividad desde datos de diccionario

        Args:
            data: Datos de diccionario parseados de actions.jsonl
            platform: Nombre de plataforma (twitter/reddit)
        """
        # Saltar entradas de tipo evento
        if "event_type" in data:
            return

        activity = AgentActivity(
            platform=platform,
            agent_id=data.get("agent_id", 0),
            agent_name=data.get("agent_name", ""),
            action_type=data.get("action_type", ""),
            action_args=data.get("action_args", {}),
            round_num=data.get("round", 0),
            timestamp=data.get("timestamp", datetime.now().isoformat()),
        )

        self.add_activity(activity)

    def _worker_loop(self):
        """Hilo de trabajo en segundo plano - enviar actividades a Zep por lotes según plataforma"""
        while self._running or not self._activity_queue.empty():
            try:
                # Intentar obtener actividad de la cola (tiempo máximo 1 segundo)
                try:
                    activity = self._activity_queue.get(timeout=1)

                    # Añadir actividad al buffer de la plataforma correspondiente
                    platform = activity.platform.lower()
                    with self._buffer_lock:
                        if platform not in self._platform_buffers:
                            self._platform_buffers[platform] = []
                        self._platform_buffers[platform].append(activity)

                        # Verificar si esa plataforma alcanzó el tamaño del lote
                        if len(self._platform_buffers[platform]) >= self.BATCH_SIZE:
                            batch = self._platform_buffers[platform][: self.BATCH_SIZE]
                            self._platform_buffers[platform] = self._platform_buffers[
                                platform
                            ][self.BATCH_SIZE :]
                            # Liberar candado antes de enviar
                            self._send_batch_activities(batch, platform)
                            # Intervalo de envío, evitar demasiadas solicitudes rápidas
                            time.sleep(self.SEND_INTERVAL)

                except Empty:
                    pass

            except Exception as e:
                logger.error(f"Excepción en hilo de trabajo: {e}")
                time.sleep(1)

    def _send_batch_activities(self, activities: List[AgentActivity], platform: str):
        """
        Enviar lote de actividades al grafo (fusionadas en un texto)

        Args:
            activities: Lista de actividades de Agent
            platform: Nombre de plataforma
        """
        if not activities:
            return

        # Fusionar múltiples actividades en un texto, separadas por saltos de línea
        episode_texts = [activity.to_episode_text() for activity in activities]
        combined_text = "\n".join(episode_texts)

        # Envío con reintentos
        for attempt in range(self.MAX_RETRIES):
            try:
                self.backend.add_episode(
                    graph_id=self.graph_id, content=combined_text, source_type="text"
                )

                self._total_sent += 1
                self._total_items_sent += len(activities)
                display_name = self._get_platform_display_name(platform)
                logger.info(
                    f"Envío en lote exitoso de {len(activities)} actividades de {display_name} al grafo {self.graph_id}"
                )
                logger.debug(
                    f"Vista previa del contenido del lote: {combined_text[:200]}..."
                )
                return

            except Exception as e:
                if attempt < self.MAX_RETRIES - 1:
                    logger.warning(
                        f"Envío en lote fallido (intento {attempt + 1}/{self.MAX_RETRIES}): {e}"
                    )
                    time.sleep(self.RETRY_DELAY * (attempt + 1))
                else:
                    logger.error(
                        f"Envío en lote fallido, reintentado {self.MAX_RETRIES} veces: {e}"
                    )
                    self._failed_count += 1

    def _flush_remaining(self):
        """Enviar actividades restantes en cola y buffer"""
        # Primero procesar actividades restantes en la cola, añadirlas al buffer
        while not self._activity_queue.empty():
            try:
                activity = self._activity_queue.get_nowait()
                platform = activity.platform.lower()
                with self._buffer_lock:
                    if platform not in self._platform_buffers:
                        self._platform_buffers[platform] = []
                    self._platform_buffers[platform].append(activity)
            except Empty:
                break

        # Luego enviar actividades restantes en buffers de cada plataforma (aunque no alcancen BATCH_SIZE)
        with self._buffer_lock:
            for platform, buffer in self._platform_buffers.items():
                if buffer:
                    display_name = self._get_platform_display_name(platform)
                    logger.info(
                        f"Enviando {len(buffer)} actividades restantes de plataforma {display_name}"
                    )
                    self._send_batch_activities(buffer, platform)
            # Limpiar todos los buffers
            for platform in self._platform_buffers:
                self._platform_buffers[platform] = []

    def get_stats(self) -> Dict[str, Any]:
        """Obtener información de estadísticas"""
        with self._buffer_lock:
            buffer_sizes = {p: len(b) for p, b in self._platform_buffers.items()}

        return {
            "graph_id": self.graph_id,
            "batch_size": self.BATCH_SIZE,
            "total_activities": self._total_activities,  # Total de actividades añadidas a la cola
            "batches_sent": self._total_sent,  # Lotes enviados exitosamente
            "items_sent": self._total_items_sent,  # Actividades enviadas exitosamente
            "failed_count": self._failed_count,  # Lotes de envío fallidos
            "skipped_count": self._skipped_count,  # Actividades filtradas/saltadas (DO_NOTHING)
            "queue_size": self._activity_queue.qsize(),
            "buffer_sizes": buffer_sizes,  # Tamaño de buffers por plataforma
            "running": self._running,
        }


class ZepGraphMemoryManager:
    """
    Gestor de actualizadores de memoria del grafo para múltiples simulaciones

    Cada simulación puede tener su propia instancia de actualizador
    """

    _updaters: Dict[str, ZepGraphMemoryUpdater] = {}
    _lock = threading.Lock()

    @classmethod
    def create_updater(
        cls, simulation_id: str, graph_id: str, backend=None
    ) -> ZepGraphMemoryUpdater:
        """
        Crear actualizador de memoria de grafo para simulación

        Args:
            simulation_id: ID de simulación
            graph_id: ID del grafo
            backend: Memory backend (opcional)

        Returns:
            Instancia de ZepGraphMemoryUpdater
        """
        with cls._lock:
            # Si ya existe, detener el anterior primero
            if simulation_id in cls._updaters:
                cls._updaters[simulation_id].stop()

            updater = ZepGraphMemoryUpdater(graph_id, backend=backend)
            updater.start()
            cls._updaters[simulation_id] = updater

            logger.info(
                f"Creado actualizador de memoria de grafo: simulation_id={simulation_id}, graph_id={graph_id}"
            )
            return updater

    @classmethod
    def get_updater(cls, simulation_id: str) -> Optional[ZepGraphMemoryUpdater]:
        """Obtener actualizador de simulación"""
        return cls._updaters.get(simulation_id)

    @classmethod
    def stop_updater(cls, simulation_id: str):
        """Detener y eliminar actualizador de simulación"""
        with cls._lock:
            if simulation_id in cls._updaters:
                cls._updaters[simulation_id].stop()
                del cls._updaters[simulation_id]
                logger.info(
                    f"Detenido actualizador de memoria de grafo: simulation_id={simulation_id}"
                )

    # Bandera para prevenir llamadas repetidas a stop_all
    _stop_all_done = False

    @classmethod
    def stop_all(cls):
        """Detener todos los actualizadores"""
        # Prevenir llamada repetida
        if cls._stop_all_done:
            return
        cls._stop_all_done = True

        with cls._lock:
            if cls._updaters:
                for simulation_id, updater in list(cls._updaters.items()):
                    try:
                        updater.stop()
                    except Exception as e:
                        logger.error(
                            f"Error al detener actualizador: simulation_id={simulation_id}, error={e}"
                        )
                cls._updaters.clear()
            logger.info("Detenidos todos los actualizadores de memoria de grafo")

    @classmethod
    def get_all_stats(cls) -> Dict[str, Dict[str, Any]]:
        """Obtener estadísticas de todos los actualizadores"""
        return {
            sim_id: updater.get_stats() for sim_id, updater in cls._updaters.items()
        }
