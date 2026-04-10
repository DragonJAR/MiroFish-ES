"""
Servicio de actualización de memoria de grafo Zep
Actualiza dinámicamente actividades de Agente en simulación al grafo Zep
"""

import os
import time
import threading
import json
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
from datetime import datetime
from queue import Queue, Empty

from zep_cloud.client import Zep

from ..config import Config
from ..utils.logger import get_logger
from ..utils.locale import get_locale, set_locale

logger = get_logger("mirofish.zep_graph_memory_updater")


@dataclass
class AgentActivity:
    """Registro de actividad de Agente"""

    platform: str  # twitter / reddit
    agent_id: int
    agent_name: str
    action_type: str  # CREATE_POST, LIKE_POST, etc.
    action_args: Dict[str, Any]
    round_num: int
    timestamp: str

    def to_episode_text(self) -> str:
        """
        Convertir actividad a descripción de texto que puede enviarse a Zep

        Adopta formato de descripción en lenguaje natural, permitiendo a Zep extraer entidades y relaciones
        No agregar prefijos relacionados con simulación, evitar actualizar grafo con información engañosa
        """
        # Generar diferentes descripciones según diferentes tipos de acción
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

        # Devolver directamente en formato "nombre del agent: descripción de actividad", sin agregar prefijo de simulación
        return f"{self.agent_name}: {description}"

    def _describe_create_post(self) -> str:
        content = self.action_args.get("content", "")
        if content:
            return f"publicó un mensaje: 「{content}」"
        return "publicó un mensaje"

    def _describe_like_post(self) -> str:
        """Dar like a un post - contiene contenido original del post y información del autor"""
        post_content = self.action_args.get("post_content", "")
        post_author = self.action_args.get("post_author_name", "")

        if post_content and post_author:
            return f"dio like a {post_author} del post: 「{post_content}」"
        elif post_content:
            return f"dio like a un post: 「{post_content}」"
        elif post_author:
            return f"dio like a un post de {post_author}"
        return "dio like a un post"

    def _describe_dislike_post(self) -> str:
        """Hacer downvote a un post - contiene contenido original del post y información del autor"""
        post_content = self.action_args.get("post_content", "")
        post_author = self.action_args.get("post_author_name", "")

        if post_content and post_author:
            return f"hizo downvote a {post_author} del post: 「{post_content}」"
        elif post_content:
            return f"hizo downvote a un post: 「{post_content}」"
        elif post_author:
            return f"hizo downvote a un post de {post_author}"
        return "hizo downvote a un post"

    def _describe_repost(self) -> str:
        """Retuitear post - contiene contenido del post original e información del autor"""
        original_content = self.action_args.get("original_content", "")
        original_author = self.action_args.get("original_author_name", "")

        if original_content and original_author:
            return f"retuiteó un post de {original_author}: 「{original_content}」"
        elif original_content:
            return f"retuiteó un post: 「{original_content}」"
        elif original_author:
            return f"retuiteó un post de {original_author}"
        return "retuiteó un post"

    def _describe_quote_post(self) -> str:
        """Citar post - contiene contenido del post original, información del autor y comentario citado"""
        original_content = self.action_args.get("original_content", "")
        original_author = self.action_args.get("original_author_name", "")
        quote_content = self.action_args.get(
            "quote_content", ""
        ) or self.action_args.get("content", "")

        base = ""
        if original_content and original_author:
            base = f"citó un post de {original_author}「{original_content}」"
        elif original_content:
            base = f"citó un post「{original_content}」"
        elif original_author:
            base = f"citó un post de {original_author}"
        else:
            base = "citó un post"

        if quote_content:
            base += f" y comentó: 「{quote_content}」"
        return base

    def _describe_follow(self) -> str:
        """Seguir usuario - contiene nombre del usuario seguido"""
        target_user_name = self.action_args.get("target_user_name", "")

        if target_user_name:
            return f"siguió al usuario「{target_user_name}」"
        return "siguió a un usuario"

    def _describe_create_comment(self) -> str:
        """Publicar comentario - contiene contenido del comentario y información del post comentado"""
        content = self.action_args.get("content", "")
        post_content = self.action_args.get("post_content", "")
        post_author = self.action_args.get("post_author_name", "")

        if content:
            if post_content and post_author:
                return f"comentó en el post「{post_content}」de {post_author}: 「{content}」"
            elif post_content:
                return f"comentó en el post「{post_content}」: 「{content}」"
            elif post_author:
                return f"comentó en un post de {post_author}: 「{content}」"
            return f"comentó: 「{content}」"
        return "publicó un comentario"

    def _describe_like_comment(self) -> str:
        """Dar like a un comentario - contiene contenido del comentario y información del autor"""
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
        """Hacer downvote a un comentario - contiene contenido del comentario y información del autor"""
        comment_content = self.action_args.get("comment_content", "")
        comment_author = self.action_args.get("comment_author_name", "")

        if comment_content and comment_author:
            return f"hizo downvote al comentario de {comment_author}: 「{comment_content}」"
        elif comment_content:
            return f"hizo downvote a un comentario: 「{comment_content}」"
        elif comment_author:
            return f"hizo downvote a un comentario de {comment_author}"
        return "hizo downvote a un comentario"

    def _describe_search(self) -> str:
        """搜索帖子 - 包含搜索关键词"""
        query = self.action_args.get("query", "") or self.action_args.get("keyword", "")
        return f"搜索了「{query}」" if query else "进行了搜索"

    def _describe_search_user(self) -> str:
        """搜索用户 - 包含搜索关键词"""
        query = self.action_args.get("query", "") or self.action_args.get(
            "username", ""
        )
        return f"搜索了用户「{query}」" if query else "搜索了用户"

    def _describe_mute(self) -> str:
        """Bloquear usuario - contiene nombre del usuario bloqueado"""
        target_user_name = self.action_args.get("target_user_name", "")

        if target_user_name:
            return f"bloqueó al usuario「{target_user_name}」"
        return "bloqueó a un usuario"

    def _describe_generic(self) -> str:
        # Para tipos de acción desconocidos, generar descripción genérica
        return f"ejecutó operación {self.action_type}"


class ZepGraphMemoryUpdater:
    """
    Actualizador de memoria de Grafo Zep

    Monitorea el archivo de log actions de simulación, actualiza dinámicamente actividades de Agente en el Grafo Zep.
    Agrupa por plataforma, envía por lotes a Zep después de acumular BATCH_SIZE actividades.

    Todos los comportamientos significativos se actualizarán a Zep, action_args contendrá información contextual completa:
    - Contenido original de posts con likes/dislikes
    - Contenido original de posts retuiteados/citados
    - Nombres de usuarios seguidos/bloqueados
    - Contenido original de comentarios con likes/dislikes
    """

    # Tamaño de envío por lotes (cuántos acumular por plataforma antes de enviar)
    BATCH_SIZE = 5

    # Mapeo de nombres de plataformas (para mostrar en consola)
    PLATFORM_DISPLAY_NAMES = {
        "twitter": "Mundo 1",
        "reddit": "Mundo 2",
    }

    # Intervalo de envío (segundos), evitar solicitudes demasiado rápidas
    SEND_INTERVAL = 0.5

    # Reintentar配置
    MAX_RETRIES = 3
    RETRY_DELAY = 2  # 秒

    def __init__(self, graph_id: str, api_key: Optional[str] = None):
        """
        Inicializar actualizador

        Args:
            graph_id: ZepGrafoID
            api_key: Zep API Key (opcional, por defecto leer desde configuración)
        """
        self.graph_id = graph_id
        self.api_key = api_key or Config.ZEP_API_KEY

        if not self.api_key:
            raise ValueError("ZEP_API_KEY no configurada")

        self.client = Zep(api_key=self.api_key)

        # Cola de actividades
        self._activity_queue: Queue = Queue()

        # Buffer de actividades agrupado por plataforma (cada plataforma acumula hasta BATCH_SIZE antes de enviar por lote)
        self._platform_buffers: Dict[str, List[AgentActivity]] = {
            "twitter": [],
            "reddit": [],
        }
        self._buffer_lock = threading.Lock()

        # Bandera de control
        self._running = False
        self._worker_thread: Optional[threading.Thread] = None

        # Estadísticas
        self._total_activities = 0  # Número de actividades agregadas a la cola
        self._total_sent = 0  # Número de lotes enviados con éxito a Zep
        self._total_items_sent = 0  # Número de actividades enviadas con éxito a Zep
        self._failed_count = 0  # Número de lotes fallidos al enviar
        self._skipped_count = 0  # Número de actividades filtradas/saltadas (DO_NOTHING)

        logger.info(
            f"ZepGraphMemoryUpdater Inicialización completada: graph_id={graph_id}, batch_size={self.BATCH_SIZE}"
        )

    def _get_platform_display_name(self, platform: str) -> str:
        """Obtener nombre de visualización de plataforma"""
        return self.PLATFORM_DISPLAY_NAMES.get(platform.lower(), platform)

    def start(self):
        """Iniciar hilo de trabajo en segundo plano"""
        if self._running:
            return

        # Capturar locale antes de lanzar hilo en segundo plano
        current_locale = get_locale()

        self._running = True
        self._worker_thread = threading.Thread(
            target=self._worker_loop,
            args=(current_locale,),
            daemon=True,
            name=f"ZepMemoryUpdater-{self.graph_id[:8]}",
        )
        self._worker_thread.start()
        logger.info(f"ZepGraphMemoryUpdater Iniciado: graph_id={self.graph_id}")

    def stop(self):
        """Detener hilo de trabajo en segundo plano"""
        self._running = False

        # Enviar actividades restantes
        self._flush_remaining()

        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=10)

        logger.info(
            f"ZepGraphMemoryUpdater Detenido: graph_id={self.graph_id}, "
            f"total_activities={self._total_activities}, "
            f"batches_sent={self._total_sent}, "
            f"items_sent={self._total_items_sent}, "
            f"failed={self._failed_count}, "
            f"skipped={self._skipped_count}"
        )

    def add_activity(self, activity: AgentActivity):
        """
        Agregar una actividad de agente a la cola

        Todos los comportamientos significativos se agregarán a la cola, incluyendo：
        - CREATE_POST (publicar post)
        - CREATE_COMMENT (comentar)
        - QUOTE_POST (citar post)
        - SEARCH_POSTS (buscar posts)
        - SEARCH_USER (buscar usuario)
        - LIKE_POST/DISLIKE_POST (dar like/dislike a post)
        - REPOST (retuitear)
        - FOLLOW (seguir)
        - MUTE (bloquear)
        - LIKE_COMMENT/DISLIKE_COMMENT (dar like/dislike a comentario)

        action_args contendrá información contextual completa (como contenido de post, nombre de usuario, etc.).

        Args:
            activity: Registro de actividad de Agente
        """
        # Saltar actividades de tipo DO_NOTHING
        if activity.action_type == "DO_NOTHING":
            self._skipped_count += 1
            return

        self._activity_queue.put(activity)
        self._total_activities += 1
        logger.debug(
            f"Agregar actividad a cola Zep: {activity.agent_name} - {activity.action_type}"
        )

    def add_activity_from_dict(self, data: Dict[str, Any], platform: str):
        """
        Agregar actividad desde datos de diccionario

        Args:
            data: Datos de diccionario parseados desde actions.jsonl
            platform: Nombre de plataforma (twitter/reddit)
        """
        # Saltar事件类型的条目
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

    def _worker_loop(self, locale: str = "zh"):
        """Ciclo de trabajo en segundo plano - enviar actividades por lotes por plataforma a Zep"""
        set_locale(locale)
        while self._running or not self._activity_queue.empty():
            try:
                # Intentar obtener actividad desde la cola (tiempo de espera 1 segundo)
                try:
                    activity = self._activity_queue.get(timeout=1)

                    # Agregar actividad al buffer de plataforma correspondiente
                    platform = activity.platform.lower()
                    with self._buffer_lock:
                        if platform not in self._platform_buffers:
                            self._platform_buffers[platform] = []
                        self._platform_buffers[platform].append(activity)

                        # Verificar si la plataforma ha alcanzado tamaño de lote
                        if len(self._platform_buffers[platform]) >= self.BATCH_SIZE:
                            batch = self._platform_buffers[platform][: self.BATCH_SIZE]
                            self._platform_buffers[platform] = self._platform_buffers[
                                platform
                            ][self.BATCH_SIZE :]
                            # Liberación锁后再发送
                            self._send_batch_activities(batch, platform)
                            # Intervalo de envío, evitar solicitudes demasiado rápidas
                            time.sleep(self.SEND_INTERVAL)

                except Empty:
                    pass

            except Exception as e:
                logger.error(f"Excepción en ciclo de trabajo: {e}")
                time.sleep(1)

    def _send_batch_activities(self, activities: List[AgentActivity], platform: str):
        """
        Enviar actividades por lote a ZepGrafo (combinar como un texto)

        Args:
            activities: Lista de actividades de Agente
            platform: Nombre de plataforma
        """
        if not activities:
            return

        # Combinar múltiples actividades como un texto, separar por nueva línea
        episode_texts = [activity.to_episode_text() for activity in activities]
        combined_text = "\n".join(episode_texts)

        # Envío con reintento
        for attempt in range(self.MAX_RETRIES):
            try:
                self.client.graph.add(
                    graph_id=self.graph_id, type="text", data=combined_text
                )

                self._total_sent += 1
                self._total_items_sent += len(activities)
                display_name = self._get_platform_display_name(platform)
                logger.info(
                    f"Éxito批量发送 {len(activities)} 条{display_name}活动到Grafo {self.graph_id}"
                )
                logger.debug(f"批量内容预览: {combined_text[:200]}...")
                return

            except Exception as e:
                if attempt < self.MAX_RETRIES - 1:
                    logger.warning(
                        f"批量发送到ZepFallido (尝试 {attempt + 1}/{self.MAX_RETRIES}): {e}"
                    )
                    time.sleep(self.RETRY_DELAY * (attempt + 1))
                else:
                    logger.error(
                        f"批量发送到ZepFallido，已Reintentar{self.MAX_RETRIES}次: {e}"
                    )
                    self._failed_count += 1

    def _flush_remaining(self):
        """Enviar actividades restantes en cola y buffer"""
        # Primero procesar actividades restantes en cola, agregar al buffer
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

        # Luego enviar actividades restantes en buffers de cada plataforma (incluso si hay menos de BATCH_SIZE)
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
        """Obtener información estadística"""
        with self._buffer_lock:
            buffer_sizes = {p: len(b) for p, b in self._platform_buffers.items()}

        return {
            "graph_id": self.graph_id,
            "batch_size": self.BATCH_SIZE,
            "total_activities": self._total_activities,  # Número total de actividades agregadas a la cola
            "batches_sent": self._total_sent,  # Número de lotes enviados con éxito
            "items_sent": self._total_items_sent,  # Número de actividades enviadas con éxito
            "failed_count": self._failed_count,  # Número de lotes fallidos al enviar
            "skipped_count": self._skipped_count,  # Número de actividades filtradas/saltadas (DO_NOTHING)
            "queue_size": self._activity_queue.qsize(),
            "buffer_sizes": buffer_sizes,  # Tamaño de buffers por plataforma
            "running": self._running,
        }


class ZepGraphMemoryManager:
    """
    Gestionar actualizadores de memoria de Grafo Zep para múltiples simulaciones

    Cada simulación puede tener su propia instancia de actualizador
    """

    _updaters: Dict[str, ZepGraphMemoryUpdater] = {}
    _lock = threading.Lock()

    @classmethod
    def create_updater(cls, simulation_id: str, graph_id: str) -> ZepGraphMemoryUpdater:
        """
        Crear actualizador de memoria de Grafo para simulación

        Args:
            simulation_id: ID de simulación
            graph_id: ZepGrafoID

        Returns:
            Instancia de ZepGraphMemoryUpdater
        """
        with cls._lock:
            # 如果已存在，先停止旧的
            if simulation_id in cls._updaters:
                cls._updaters[simulation_id].stop()

            updater = ZepGraphMemoryUpdater(graph_id)
            updater.start()
            cls._updaters[simulation_id] = updater

            logger.info(
                f"Creando actualizador de memoria de Grafo: simulation_id={simulation_id}, graph_id={graph_id}"
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
                    f"Detenida actualización de memoria de Grafo: simulation_id={simulation_id}"
                )

    # Bandera para prevenir llamadas repetidas de stop_all
    _stop_all_done = False

    @classmethod
    def stop_all(cls):
        """Detener todos los actualizadores"""
        # Prevenir llamadas repetidas
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
                            f"Falló al detener actualizador: simulation_id={simulation_id}, error={e}"
                        )
                cls._updaters.clear()
            logger.info("Detenidos todos los actualizadores de memoria de Grafo")

    @classmethod
    def get_all_stats(cls) -> Dict[str, Dict[str, Any]]:
        """Obtener información estadística de todos los actualizadores"""
        return {
            sim_id: updater.get_stats() for sim_id, updater in cls._updaters.items()
        }
