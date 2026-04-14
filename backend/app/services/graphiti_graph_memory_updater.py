"""
Servicio de actualización de memoria de grafo Graphiti
Implementa GraphMemoryUpdaterInterface usando Graphiti + Neo4j
"""

import threading
from typing import Dict, Any, List, Optional

from ..memory.updaters import (
    GraphMemoryUpdaterInterface,
    GraphMemoryUpdateResult,
)
from ..memory.graphiti_backend import GraphitiBackend
from ..utils.logger import get_logger

logger = get_logger("mirofish.graphiti_graph_memory_updater")


class GraphitiGraphMemoryUpdater(GraphMemoryUpdaterInterface):
    """
    Actualizador de memoria de Grafo Graphiti

    Implementa GraphMemoryUpdaterInterface para DRY con Zep.
    Usa GraphitiBackend.add_episode() para enviar episodios al grafo.
    """

    _updaters: Dict[str, "GraphitiGraphMemoryUpdater"] = {}
    _lock = threading.Lock()

    # Bandera para prevenir llamadas repetidas de stop_all
    _stop_all_done = False

    def __init__(self, graph_id: str = None):
        """
        Inicializar actualizador Graphiti

        Args:
            graph_id: ID del grafo en Graphiti/Neo4j
        """
        self.graph_id = graph_id
        self._backend = GraphitiBackend()

        # Estadísticas
        self._total_episodes = 0
        self._total_entities = 0
        self._failed_count = 0

        logger.info(f"GraphitiGraphMemoryUpdater inicializado: graph_id={graph_id}")

    def create_updater(
        self, session_id: str, graph_id: str
    ) -> "GraphitiGraphMemoryUpdater":
        """
        Crear actualizador de memoria de Grafo para simulación

        Args:
            session_id: ID de simulación
            graph_id: ID del grafo en Graphiti

        Returns:
            Instancia de GraphitiGraphMemoryUpdater
        """
        with self._lock:
            if session_id in self._updaters:
                logger.warning(
                    f"Ya existe un updater para session_id={session_id}, reemplazando"
                )

            updater = GraphitiGraphMemoryUpdater(graph_id)
            self._updaters[session_id] = updater

            logger.info(
                f"Creando actualizador de memoria de Grafo Graphiti: session_id={session_id}, graph_id={graph_id}"
            )
            return updater

    def update_batch(
        self, entities: List[dict], relations: List[dict], session_id: str
    ) -> GraphMemoryUpdateResult:
        """
        Enviar lote de entidades y relaciones al grafo Graphiti.

        Cada entidad se convierte en un episodio de texto que Graphiti
        procesará para extraer entidades y relaciones automáticamente.

        Args:
            entities: Lista de diccionarios con datos de entidades
            relations: Lista de diccionarios con datos de relaciones
            session_id: ID de la sesión

        Returns:
            GraphMemoryUpdateResult con resultados
        """
        try:
            updater = self._updaters.get(session_id)
            if not updater:
                return GraphMemoryUpdateResult(
                    success=False,
                    updated_count=0,
                    failed_count=len(entities),
                    error_message=f"No existe updater para session_id={session_id}",
                )

            updated_count = 0
            failed_count = 0

            for entity in entities:
                try:
                    # Convertir entidad a texto descriptivo
                    episode_text = self._entity_to_episode_text(entity)
                    if episode_text:
                        result = updater._backend.add_episode(
                            graph_id=updater.graph_id,
                            content=episode_text,
                            reference_time=entity.get("timestamp"),
                            name=f"Episode_{entity.get('agent_name', 'unknown')}_{entity.get('round_num', 0)}",
                        )
                        if result and result.status == "completed":
                            updated_count += 1
                        else:
                            failed_count += 1
                    else:
                        failed_count += 1
                except Exception as e:
                    logger.error(f"Error al procesar entidad: {e}")
                    failed_count += 1

            updater._total_episodes += updated_count
            updater._total_entities += len(entities)
            updater._failed_count += failed_count

            return GraphMemoryUpdateResult(
                success=(failed_count == 0),
                updated_count=updated_count,
                failed_count=failed_count,
            )

        except Exception as e:
            logger.error(f"Error en update_batch Graphiti: {e}")
            return GraphMemoryUpdateResult(
                success=False,
                updated_count=0,
                failed_count=len(entities),
                error_message=str(e),
            )

    def _entity_to_episode_text(self, entity: dict) -> str:
        """
        Convertir diccionario de entidad a texto descriptivo para Graphiti.

        Args:
            entity: Diccionario con datos de entidad

        Returns:
            Texto descriptivo o None si no hay suficiente información
        """
        agent_name = entity.get("agent_name", "")
        action_type = entity.get("action_type", "")
        action_args = entity.get("action_args", {})

        if not agent_name:
            return None

        # Generar descripción según tipo de acción
        descriptions = {
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

        describe_func = descriptions.get(action_type, self._describe_generic)
        description = describe_func(action_args)

        return f"{agent_name}: {description}"

    def _describe_create_post(self, args: dict) -> str:
        content = args.get("content", "")
        if content:
            return f"publicó un mensaje: 「{content}」"
        return "publicó un mensaje"

    def _describe_like_post(self, args: dict) -> str:
        post_content = args.get("post_content", "")
        post_author = args.get("post_author_name", "")
        if post_content and post_author:
            return f"dio like a {post_author} del post: 「{post_content}」"
        elif post_content:
            return f"dio like a un post: 「{post_content}」"
        elif post_author:
            return f"dio like a un post de {post_author}"
        return "dio like a un post"

    def _describe_dislike_post(self, args: dict) -> str:
        post_content = args.get("post_content", "")
        post_author = args.get("post_author_name", "")
        if post_content and post_author:
            return f"hizo downvote a {post_author} del post: 「{post_content}」"
        elif post_content:
            return f"hizo downvote a un post: 「{post_content}」"
        elif post_author:
            return f"hizo downvote a un post de {post_author}"
        return "hizo downvote a un post"

    def _describe_repost(self, args: dict) -> str:
        original_content = args.get("original_content", "")
        original_author = args.get("original_author_name", "")
        if original_content and original_author:
            return f"retuiteó un post de {original_author}: 「{original_content}」"
        elif original_content:
            return f"retuiteó un post: 「{original_content}」"
        elif original_author:
            return f"retuiteó un post de {original_author}"
        return "retuiteó un post"

    def _describe_quote_post(self, args: dict) -> str:
        original_content = args.get("original_content", "")
        original_author = args.get("original_author_name", "")
        quote_content = args.get("quote_content", "") or args.get("content", "")

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

    def _describe_follow(self, args: dict) -> str:
        target_user_name = args.get("target_user_name", "")
        if target_user_name:
            return f"siguió al usuario「{target_user_name}」"
        return "siguió a un usuario"

    def _describe_create_comment(self, args: dict) -> str:
        content = args.get("content", "")
        post_content = args.get("post_content", "")
        post_author = args.get("post_author_name", "")

        if content:
            if post_content and post_author:
                return f"comentó en el post「{post_content}」de {post_author}: 「{content}」"
            elif post_content:
                return f"comentó en el post「{post_content}」: 「{content}」"
            elif post_author:
                return f"comentó en un post de {post_author}: 「{content}」"
            return f"comentó: 「{content}」"
        return "publicó un comentario"

    def _describe_like_comment(self, args: dict) -> str:
        comment_content = args.get("comment_content", "")
        comment_author = args.get("comment_author_name", "")
        if comment_content and comment_author:
            return f"dio like al comentario de {comment_author}: 「{comment_content}」"
        elif comment_content:
            return f"dio like a un comentario: 「{comment_content}」"
        elif comment_author:
            return f"dio like a un comentario de {comment_author}"
        return "dio like a un comentario"

    def _describe_dislike_comment(self, args: dict) -> str:
        comment_content = args.get("comment_content", "")
        comment_author = args.get("comment_author_name", "")
        if comment_content and comment_author:
            return f"hizo downvote al comentario de {comment_author}: 「{comment_content}」"
        elif comment_content:
            return f"hizo downvote a un comentario: 「{comment_content}」"
        elif comment_author:
            return f"hizo downvote a un comentario de {comment_author}"
        return "hizo downvote a un comentario"

    def _describe_search(self, args: dict) -> str:
        query = args.get("query", "") or args.get("keyword", "")
        return f"buscó「{query}」" if query else "realizó una búsqueda"

    def _describe_search_user(self, args: dict) -> str:
        query = args.get("query", "") or args.get("username", "")
        return f"buscó al usuario「{query}」" if query else "buscó usuarios"

    def _describe_mute(self, args: dict) -> str:
        target_user_name = args.get("target_user_name", "")
        if target_user_name:
            return f"bloqueó al usuario「{target_user_name}」"
        return "bloqueó a un usuario"

    def _describe_generic(self, args: dict) -> str:
        action_type = args.get("action_type", "UNKNOWN")
        return f"ejecutó operación {action_type}"

    def finalize(self, session_id: str) -> GraphMemoryUpdateResult:
        """
        Flush final de datos pendientes.

        Para Graphiti, add_episode() ya procesa inmediatamente,
        así que este método solo retorna las estadísticas finales.
        """
        try:
            updater = self._updaters.get(session_id)
            if not updater:
                return GraphMemoryUpdateResult(
                    success=False,
                    updated_count=0,
                    failed_count=0,
                    error_message=f"No existe updater para session_id={session_id}",
                )

            return GraphMemoryUpdateResult(
                success=True,
                updated_count=updater._total_episodes,
                failed_count=updater._failed_count,
            )
        except Exception as e:
            logger.error(f"Error en finalize Graphiti: {e}")
            return GraphMemoryUpdateResult(
                success=False,
                updated_count=0,
                failed_count=0,
                error_message=str(e),
            )

    def stop_updater(self, session_id: str):
        """Detener y eliminar actualizador de simulación"""
        with self._lock:
            if session_id in self._updaters:
                del self._updaters[session_id]
                logger.info(
                    f"Detenida actualización de memoria de Grafo Graphiti: session_id={session_id}"
                )

    def get_current_updater(
        self, session_id: str
    ) -> Optional["GraphitiGraphMemoryUpdater"]:
        """Obtener el actualizador actual para una sesión"""
        return self._updaters.get(session_id)

    def get_stats(self, session_id: str) -> dict:
        """Obtener estadísticas del actualizador"""
        updater = self._updaters.get(session_id)
        if updater:
            return {
                "graph_id": updater.graph_id,
                "total_episodes": updater._total_episodes,
                "total_entities": updater._total_entities,
                "failed_count": updater._failed_count,
            }
        return {}

    def stop_all(self):
        """Detener todos los actualizadores"""
        if self._stop_all_done:
            return
        self._stop_all_done = True

        with self._lock:
            if self._updaters:
                self._updaters.clear()
            logger.info(
                "Detenidos todos los actualizadores de memoria de Grafo Graphiti"
            )
