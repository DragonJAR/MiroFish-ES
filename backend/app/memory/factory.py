"""
Factory para backends de memoria
Patrón Singleton para manejar la instancia del backend
"""

from typing import Optional
from .base import MemoryBackend
from ..config import Config
from ..utils.logger import get_logger

logger = get_logger("mirofish.memory.factory")

# Backend instance (singleton)
_backend_instance: Optional[MemoryBackend] = None


def get_memory_backend() -> MemoryBackend:
    """
    Obtener instancia singleton del backend de memoria

    El tipo de backend se determina por la configuración MEMORY_BACKEND
    Valores soportados: "zep" (por defecto), "graphiti"
    """
    global _backend_instance

    if _backend_instance is None:
        backend_type = getattr(Config, "MEMORY_BACKEND", "zep").lower()

        logger.info(f"Inicializando backend de memoria: {backend_type}")

        if backend_type == "graphiti":
            from .graphiti_backend import GraphitiBackend

            _backend_instance = GraphitiBackend()
        else:
            from .zep_backend import ZepBackend

            _backend_instance = ZepBackend()

        logger.info(f"Backend de memoria inicializado: {backend_type}")

    return _backend_instance


def reset_memory_backend():
    """
    Reiniciar singleton del backend de memoria

    Útil para testing o cambios de configuración
    """
    global _backend_instance
    _backend_instance = None
    logger.info("Backend de memoria reiniciado")
