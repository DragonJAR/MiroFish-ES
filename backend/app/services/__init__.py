"""
Módulo de servicios de negocio

Imports son lazy cuando es posible para evitar cargar backends innecesarios.
El patrón es: código que necesita un backend específico debe usar
`get_memory_backend()` o `get_memory_updater()` del módulo `app.memory`.
"""

from ..memory.base import EntityNode, FilteredEntities
from ..memory import get_memory_backend, get_memory_updater

# ── Non-backend-specific ──────────────────────────────────────────────────────
from .ontology_generator import OntologyGenerator
from .text_processor import TextProcessor
from .oasis_profile_generator import OasisProfileGenerator, OasisAgentProfile
from .simulation_manager import SimulationManager, SimulationState, SimulationStatus
from .simulation_config_generator import (
    SimulationConfigGenerator,
    SimulationParameters,
    AgentActivityConfig,
    TimeSimulationConfig,
    EventConfig,
    PlatformConfig,
)
from .simulation_runner import (
    SimulationRunner,
    SimulationRunState,
    RunnerStatus,
    AgentAction,
    RoundSummary,
)
from .simulation_ipc import (
    SimulationIPCClient,
    SimulationIPCServer,
    IPCCommand,
    IPCResponse,
    CommandType,
    CommandStatus,
)

# ── Backend-conditional (solo se carga el activo) ────────────────────────────
# Si MEMORY_BACKEND cambia, se reinicia el proceso y se carga el otro.
# Pattern: lazy via factory (app.memory.factory), no imports top-level.
_BACKEND = None


def _get_backend():
    global _BACKEND
    if _BACKEND is None:
        from ..config import Config

        _BACKEND = getattr(Config, "MEMORY_BACKEND", "zep").lower()
    return _BACKEND


def __getattr__(name: str):
    backend = _get_backend()

    # Zep-only
    if name in (
        "GraphBuilderService",
        "ZepEntityReader",
        "ZepGraphMemoryUpdater",
        "ZepGraphMemoryManager",
        "AgentActivity",
    ):
        if backend == "zep":
            import importlib

            if name == "GraphBuilderService":
                return importlib.import_module(
                    ".graph_builder", __package__
                ).GraphBuilderService
            if name == "ZepEntityReader":
                return importlib.import_module(
                    ".zep_entity_reader", __package__
                ).ZepEntityReader
            if name == "ZepGraphMemoryUpdater":
                m = importlib.import_module(".zep_graph_memory_updater", __package__)
                return m.ZepGraphMemoryUpdater
            if name == "ZepGraphMemoryManager":
                return importlib.import_module(
                    ".zep_graph_memory_updater", __package__
                ).ZepGraphMemoryManager
            if name == "AgentActivity":
                return importlib.import_module(
                    ".zep_graph_memory_updater", __package__
                ).AgentActivity
        raise AttributeError(f"{name} requires MEMORY_BACKEND=zep (current={backend})")

    # Graphiti-only
    if name == "GraphitiGraphMemoryUpdater":
        if backend == "graphiti":
            import importlib

            return importlib.import_module(
                ".graphiti_graph_memory_updater", __package__
            ).GraphitiGraphMemoryUpdater
        raise AttributeError(
            f"{name} requires MEMORY_BACKEND=graphiti (current={backend})"
        )

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Non-backend-specific
    "OntologyGenerator",
    "GraphBuilderService",
    "TextProcessor",
    "ZepEntityReader",
    "EntityNode",
    "FilteredEntities",
    "OasisProfileGenerator",
    "OasisAgentProfile",
    "SimulationManager",
    "SimulationState",
    "SimulationStatus",
    "SimulationConfigGenerator",
    "SimulationParameters",
    "AgentActivityConfig",
    "TimeSimulationConfig",
    "EventConfig",
    "PlatformConfig",
    "SimulationRunner",
    "SimulationRunState",
    "RunnerStatus",
    "AgentAction",
    "RoundSummary",
    "ZepGraphMemoryUpdater",
    "ZepGraphMemoryManager",
    "AgentActivity",
    "SimulationIPCClient",
    "SimulationIPCServer",
    "IPCCommand",
    "IPCResponse",
    "CommandType",
    "CommandStatus",
    "get_memory_backend",
    "get_memory_updater",
    "GraphitiGraphMemoryUpdater",
]
