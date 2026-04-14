"""
Interfaz abstracta para actualizadores de memoria de grafo (Zep y Graphiti)
Patrón DRY: ambos backends implementan la misma interfaz
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class GraphMemoryUpdateResult:
    """Resultado de operación de actualización de memoria de grafo"""

    success: bool
    updated_count: int
    failed_count: int
    error_message: Optional[str] = None


class GraphMemoryUpdaterInterface(ABC):
    """
     Interfaz abstracta para actualizadores de memoria de grafo.

    both Zep and Graphiti backends implement this interface,
     permitiendo que simulation_runner.py use el mismo código
     independientemente del backend configurado.
    """

    @abstractmethod
    def create_updater(
        self, session_id: str, graph_id: str
    ) -> "GraphMemoryUpdaterInterface":
        """
        Crear una nueva instancia de actualizador para una sesión.

        Args:
            session_id: ID de la simulación
            graph_id: ID del grafo en el backend

        Returns:
            Instancia del actualizador
        """
        pass

    @abstractmethod
    def update_batch(
        self, entities: List[dict], relations: List[dict], session_id: str
    ) -> GraphMemoryUpdateResult:
        """
        Enviar un lote de entidades y relaciones al grafo.

        Args:
            entities: Lista de diccionarios con datos de entidades
            relations: Lista de diccionarios con datos de relaciones
            session_id: ID de la sesión

        Returns:
            GraphMemoryUpdateResult con resultados de la operación
        """
        pass

    @abstractmethod
    def finalize(self, session_id: str) -> GraphMemoryUpdateResult:
        """
        Finalizar y flush final de datos pendientes.

        Args:
            session_id: ID de la sesión

        Returns:
            GraphMemoryUpdateResult con resultados de la operación
        """
        pass

    @abstractmethod
    def stop_updater(self, session_id: str):
        """
        Detener y limpiar el actualizador de una sesión.

        Args:
            session_id: ID de la sesión
        """
        pass

    @abstractmethod
    def get_stats(self, session_id: str) -> dict:
        """
        Obtener estadísticas del actualizador.

        Args:
            session_id: ID de la sesión

        Returns:
            Diccionario con estadísticas
        """
        pass

    @abstractmethod
    def get_current_updater(
        self, session_id: str
    ) -> Optional["GraphMemoryUpdaterInterface"]:
        """
        Obtener el actualizador actual para una sesión.

        Útil para usar métodos específicos del backend después de crear el actualizador.

        Args:
            session_id: ID de la sesión

        Returns:
            Instancia del actualizador o None si no existe
        """
        pass
