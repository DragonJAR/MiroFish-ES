"""
Módulo de abstracción de memoria de grafo
Interfaz unificada para múltiples backends (Zep, Graphiti)
"""

from typing import Optional
from .base import (
    MemoryBackend,
    SearchResult,
    EntityNode,
    GraphInfo,
    EpisodeResult,
    FilteredEntities,
)
from .factory import get_memory_backend, get_memory_updater, reset_memory_backend
from .updaters import GraphMemoryUpdaterInterface, GraphMemoryUpdateResult

__all__ = [
    "MemoryBackend",
    "SearchResult",
    "EntityNode",
    "GraphInfo",
    "EpisodeResult",
    "FilteredEntities",
    "get_memory_backend",
    "get_memory_updater",
    "reset_memory_backend",
    "GraphMemoryUpdaterInterface",
    "GraphMemoryUpdateResult",
]
