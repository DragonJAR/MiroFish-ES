"""
Módulo de servicios de negocio
"""

from .ontology_generator import OntologyGenerator
from .graph_builder import GraphBuilderService
from .text_processor import TextProcessor
from .oasis_profile_generator import OasisProfileGenerator, OasisAgentProfile
from ..memory.base import EntityNode, FilteredEntities

try:
    from .zep_entity_reader import ZepEntityReader
except ImportError:
    ZepEntityReader = None
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
from .zep_graph_memory_updater import (
    ZepGraphMemoryUpdater,
    ZepGraphMemoryManager,
    AgentActivity,
)
from .simulation_ipc import (
    SimulationIPCClient,
    SimulationIPCServer,
    IPCCommand,
    IPCResponse,
    CommandType,
    CommandStatus,
)

# Memory backend
from ..memory import get_memory_backend

__all__ = [
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
]
