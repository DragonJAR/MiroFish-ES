"""
Ejecutor de simulación OASIS
Ejecuta simulaciones en segundo plano y registra las acciones de cada Agente, con monitoreo de estado en tiempo real
"""

import os
import sys
import json
import time
import asyncio
import threading
import subprocess
import signal
import atexit
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from queue import Queue

from ..config import Config
from ..utils.logger import get_logger
from ..utils.locale import get_locale, set_locale
from .zep_graph_memory_updater import ZepGraphMemoryManager
from .simulation_ipc import SimulationIPCClient, CommandType, IPCResponse

logger = get_logger("mirofish.simulation_runner")

# Marca si se ha registrado la función de limpieza
_cleanup_registered = False

# Detección de plataforma
IS_WINDOWS = sys.platform == "win32"


class RunnerStatus(str, Enum):
    """Estado del ejecutor"""

    IDLE = "idle"
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    STOPPED = "stopped"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class AgentAction:
    """Registro de acción de Agente"""

    round_num: int
    timestamp: str
    platform: str  # twitter / reddit
    agent_id: int
    agent_name: str
    action_type: str  # CREATE_POST, LIKE_POST, etc.
    action_args: Dict[str, Any] = field(default_factory=dict)
    result: Optional[str] = None
    success: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "round_num": self.round_num,
            "timestamp": self.timestamp,
            "platform": self.platform,
            "agent_id": self.agent_id,
            "agent_name": self.agent_name,
            "action_type": self.action_type,
            "action_args": self.action_args,
            "result": self.result,
            "success": self.success,
        }


@dataclass
class RoundSummary:
    """Resumen por ronda"""

    round_num: int
    start_time: str
    end_time: Optional[str] = None
    simulated_hour: int = 0
    twitter_actions: int = 0
    reddit_actions: int = 0
    active_agents: List[int] = field(default_factory=list)
    actions: List[AgentAction] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "round_num": self.round_num,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "simulated_hour": self.simulated_hour,
            "twitter_actions": self.twitter_actions,
            "reddit_actions": self.reddit_actions,
            "active_agents": self.active_agents,
            "actions_count": len(self.actions),
            "actions": [a.to_dict() for a in self.actions],
        }


@dataclass
class SimulationRunState:
    """Estado de ejecución de simulación (tiempo real)"""

    simulation_id: str
    runner_status: RunnerStatus = RunnerStatus.IDLE

    # Información de progreso
    current_round: int = 0
    total_rounds: int = 0
    simulated_hours: int = 0
    total_simulation_hours: int = 0

    # Rondas y tiempo de simulación independientes por plataforma (para visualización en paralelo de dos plataformas)
    twitter_current_round: int = 0
    reddit_current_round: int = 0
    twitter_simulated_hours: int = 0
    reddit_simulated_hours: int = 0

    # Estado de plataformas
    twitter_running: bool = False
    reddit_running: bool = False
    twitter_actions_count: int = 0
    reddit_actions_count: int = 0

    # Estado de finalización de plataformas (detectando eventos simulation_end en actions.jsonl)
    twitter_completed: bool = False
    reddit_completed: bool = False

    # Resumen por ronda
    rounds: List[RoundSummary] = field(default_factory=list)

    # Acciones recientes (para visualización en tiempo real en el frontend)
    recent_actions: List[AgentAction] = field(default_factory=list)
    max_recent_actions: int = 50

    # Marcas de tiempo
    started_at: Optional[str] = None
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    completed_at: Optional[str] = None

    # Información de error
    error: Optional[str] = None

    # ID de proceso (para detención)
    process_pid: Optional[int] = None

    def add_action(self, action: AgentAction):
        """Agregar acción a la lista de acciones recientes"""
        self.recent_actions.insert(0, action)
        if len(self.recent_actions) > self.max_recent_actions:
            self.recent_actions = self.recent_actions[: self.max_recent_actions]

        if action.platform == "twitter":
            self.twitter_actions_count += 1
        else:
            self.reddit_actions_count += 1

        self.updated_at = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "simulation_id": self.simulation_id,
            "runner_status": self.runner_status.value,
            "current_round": self.current_round,
            "total_rounds": self.total_rounds,
            "simulated_hours": self.simulated_hours,
            "total_simulation_hours": self.total_simulation_hours,
            "progress_percent": round(
                self.current_round / max(self.total_rounds, 1) * 100, 1
            ),
            # Rondas y tiempos independientes por plataforma
            "twitter_current_round": self.twitter_current_round,
            "reddit_current_round": self.reddit_current_round,
            "twitter_simulated_hours": self.twitter_simulated_hours,
            "reddit_simulated_hours": self.reddit_simulated_hours,
            "twitter_running": self.twitter_running,
            "reddit_running": self.reddit_running,
            "twitter_completed": self.twitter_completed,
            "reddit_completed": self.reddit_completed,
            "twitter_actions_count": self.twitter_actions_count,
            "reddit_actions_count": self.reddit_actions_count,
            "total_actions_count": self.twitter_actions_count
            + self.reddit_actions_count,
            "started_at": self.started_at,
            "updated_at": self.updated_at,
            "completed_at": self.completed_at,
            "error": self.error,
            "process_pid": self.process_pid,
        }

    def to_detail_dict(self) -> Dict[str, Any]:
        """Detalles completos con acciones recientes"""
        result = self.to_dict()
        result["recent_actions"] = [a.to_dict() for a in self.recent_actions]
        result["rounds_count"] = len(self.rounds)
        return result


class SimulationRunner:
    """
    Ejecutor de simulación

    Responsabilidades:
    1. Ejecutar simulación OASIS en proceso de segundo plano
    2. Analizar registros de ejecución, registrar acciones de cada Agente
    3. Proporcionar interfaz de consulta de estado en tiempo real
    4. Soportar operaciones de pausa/detención/reanudación
    """

    # Directorio de almacenamiento de estados de ejecución
    RUN_STATE_DIR = os.path.join(os.path.dirname(__file__), "../../uploads/simulations")

    # Directorio de scripts
    SCRIPTS_DIR = os.path.join(os.path.dirname(__file__), "../../scripts")

    # Estado de ejecución en memoria
    _run_states: Dict[str, SimulationRunState] = {}
    _processes: Dict[str, subprocess.Popen] = {}
    _action_queues: Dict[str, Queue] = {}
    _monitor_threads: Dict[str, threading.Thread] = {}
    _stdout_files: Dict[str, Any] = {}  # Almacenar manejador de archivo stdout
    _stderr_files: Dict[str, Any] = {}  # Almacenar manejador de archivo stderr

    # Configuración de actualización de memoria de grafo
    _graph_memory_enabled: Dict[str, bool] = {}  # simulation_id -> enabled

    @classmethod
    def get_run_state(cls, simulation_id: str) -> Optional[SimulationRunState]:
        """Obtener estado de ejecución"""
        if simulation_id in cls._run_states:
            return cls._run_states[simulation_id]

        # Intentar cargar desde archivo
        state = cls._load_run_state(simulation_id)
        if state:
            cls._run_states[simulation_id] = state
        return state

    @classmethod
    def _load_run_state(cls, simulation_id: str) -> Optional[SimulationRunState]:
        """Cargar estado de ejecución desde archivo"""
        state_file = os.path.join(cls.RUN_STATE_DIR, simulation_id, "run_state.json")
        if not os.path.exists(state_file):
            return None

        try:
            with open(state_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            state = SimulationRunState(
                simulation_id=simulation_id,
                runner_status=RunnerStatus(data.get("runner_status", "idle")),
                current_round=data.get("current_round", 0),
                total_rounds=data.get("total_rounds", 0),
                simulated_hours=data.get("simulated_hours", 0),
                total_simulation_hours=data.get("total_simulation_hours", 0),
                # Rondas y tiempo independientes por plataforma
                twitter_current_round=data.get("twitter_current_round", 0),
                reddit_current_round=data.get("reddit_current_round", 0),
                twitter_simulated_hours=data.get("twitter_simulated_hours", 0),
                reddit_simulated_hours=data.get("reddit_simulated_hours", 0),
                twitter_running=data.get("twitter_running", False),
                reddit_running=data.get("reddit_running", False),
                twitter_completed=data.get("twitter_completed", False),
                reddit_completed=data.get("reddit_completed", False),
                twitter_actions_count=data.get("twitter_actions_count", 0),
                reddit_actions_count=data.get("reddit_actions_count", 0),
                started_at=data.get("started_at"),
                updated_at=data.get("updated_at", datetime.now().isoformat()),
                completed_at=data.get("completed_at"),
                error=data.get("error"),
                process_pid=data.get("process_pid"),
            )

            # Cargar acciones recientes
            actions_data = data.get("recent_actions", [])
            for a in actions_data:
                state.recent_actions.append(
                    AgentAction(
                        round_num=a.get("round_num", 0),
                        timestamp=a.get("timestamp", ""),
                        platform=a.get("platform", ""),
                        agent_id=a.get("agent_id", 0),
                        agent_name=a.get("agent_name", ""),
                        action_type=a.get("action_type", ""),
                        action_args=a.get("action_args", {}),
                        result=a.get("result"),
                        success=a.get("success", True),
                    )
                )

            return state
        except Exception as e:
            logger.error(f"Fallo al cargar estado de ejecución: {str(e)}")
            return None

    @classmethod
    def _save_run_state(cls, state: SimulationRunState):
        """Guardar estado de ejecución en archivo"""
        sim_dir = os.path.join(cls.RUN_STATE_DIR, state.simulation_id)
        os.makedirs(sim_dir, exist_ok=True)
        state_file = os.path.join(sim_dir, "run_state.json")

        data = state.to_detail_dict()

        with open(state_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        cls._run_states[state.simulation_id] = state

    @classmethod
    def start_simulation(
        cls,
        simulation_id: str,
        platform: str = "parallel",  # twitter / reddit / parallel
        max_rounds: int = None,  # Máximo número de rondas de simulación (opcional, para truncar simulaciones muy largas)
        enable_graph_memory_update: bool = False,  # Si actualizar actividades al grafo de Zep
        graph_id: str = None,  # ID del grafo de Zep (requerido cuando se habilita la actualización del grafo)
    ) -> SimulationRunState:
        """
        Iniciar simulación

        Args:
            simulation_id: ID de simulación
            platform: Plataforma a ejecutar (twitter/reddit/parallel)
            max_rounds: Máximo número de rondas de simulación (opcional, para truncar simulaciones muy largas)
            enable_graph_memory_update: Si actualizar dinámicamente las actividades del Agente al grafo de Zep
            graph_id: ID del grafo de Zep (requerido cuando se habilita la actualización del grafo)

        Returns:
            SimulationRunState
        """
        # Verificar si ya está en ejecución
        existing = cls.get_run_state(simulation_id)
        if existing and existing.runner_status in [
            RunnerStatus.RUNNING,
            RunnerStatus.STARTING,
        ]:
            raise ValueError(f"Simulación ya está en ejecución: {simulation_id}")

        # Cargar configuración de simulación
        sim_dir = os.path.join(cls.RUN_STATE_DIR, simulation_id)
        config_path = os.path.join(sim_dir, "simulation_config.json")

        if not os.path.exists(config_path):
            raise ValueError(
                f"Configuración de simulación no existe, llame primero al endpoint /prepare"
            )

        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        # Inicializar estado de ejecución
        time_config = config.get("time_config", {})
        total_hours = time_config.get("total_simulation_hours", 72)
        minutes_per_round = time_config.get("minutes_per_round", 30)
        total_rounds = int(total_hours * 60 / minutes_per_round)

        # Si se especificó un máximo de rondas, truncar
        if max_rounds is not None and max_rounds > 0:
            original_rounds = total_rounds
            total_rounds = min(total_rounds, max_rounds)
            if total_rounds < original_rounds:
                logger.info(
                    f"Rondas truncadas: {original_rounds} -> {total_rounds} (max_rounds={max_rounds})"
                )

        state = SimulationRunState(
            simulation_id=simulation_id,
            runner_status=RunnerStatus.STARTING,
            total_rounds=total_rounds,
            total_simulation_hours=total_hours,
            started_at=datetime.now().isoformat(),
        )

        cls._save_run_state(state)

        # Si se habilita la actualización de memoria de grafo, crear actualizador
        if enable_graph_memory_update:
            if not graph_id:
                raise ValueError(
                    "Debe proporcionar graph_id al habilitar la actualización de memoria de grafo"
                )

            try:
                ZepGraphMemoryManager.create_updater(simulation_id, graph_id)
                cls._graph_memory_enabled[simulation_id] = True
                logger.info(
                    f"Actualización de memoria de grafo habilitada: simulation_id={simulation_id}, graph_id={graph_id}"
                )
            except Exception as e:
                logger.error(f"Fallo al crear actualizador de memoria de grafo: {e}")
                cls._graph_memory_enabled[simulation_id] = False
        else:
            cls._graph_memory_enabled[simulation_id] = False

        # Determinar qué script ejecutar (los scripts están en backend/scripts/ directorio)
        if platform == "twitter":
            script_name = "run_twitter_simulation.py"
            state.twitter_running = True
        elif platform == "reddit":
            script_name = "run_reddit_simulation.py"
            state.reddit_running = True
        else:
            script_name = "run_parallel_simulation.py"
            state.twitter_running = True
            state.reddit_running = True

        script_path = os.path.join(cls.SCRIPTS_DIR, script_name)

        if not os.path.exists(script_path):
            raise ValueError(f"Script no existe: {script_path}")

        # Crear cola de acciones
        action_queue = Queue()
        cls._action_queues[simulation_id] = action_queue

        # Iniciar proceso de simulación
        try:
            # Construir comando de ejecución, usar ruta completa
            # Nueva estructura de registro:
            #   twitter/actions.jsonl - Registro de acciones de Twitter
            #   reddit/actions.jsonl  - Registro de acciones de Reddit
            #   simulation.log        - Registro de proceso principal

            cmd = [
                sys.executable,  # Intérprete de Python
                script_path,
                "--config",
                config_path,  # Usar ruta completa de archivo de configuración
            ]

            # Si se especificó un máximo de rondas, agregar a parámetros de línea de comandos
            if max_rounds is not None and max_rounds > 0:
                cmd.extend(["--max-rounds", str(max_rounds)])

            # Crear archivo de registro principal, evitar que el proceso se bloquee por tubería stdout/stderr llena
            main_log_path = os.path.join(sim_dir, "simulation.log")
            main_log_file = open(main_log_path, "w", encoding="utf-8")

            # Establecer variables de entorno del subproceso, asegurar uso de codificación UTF-8 en Windows
            # Esto puede corregir problemas con bibliotecas de terceros (como OASIS) que no especifican codificación al leer archivos
            env = os.environ.copy()
            env["PYTHONUTF8"] = (
                "1"  # Python 3.7+ soporta, hacer que todos los open() usen UTF-8 por defecto
            )
            env["PYTHONIOENCODING"] = "utf-8"  # Asegurar que stdout/stderr usen UTF-8

            # Establecer directorio de trabajo como directorio de simulación (bases de datos y otros archivos se generarán aquí)
            # Usar start_new_session=True para crear nuevo grupo de procesos, asegurar que se puedan terminar todos los subprocesos a través de os.killpg
            process = subprocess.Popen(
                cmd,
                cwd=sim_dir,
                stdout=main_log_file,
                stderr=subprocess.STDOUT,  # stderr también escribe al mismo archivo
                text=True,
                encoding="utf-8",  # Especificar codificación explícitamente
                bufsize=1,
                env=env,  # Pasar variables de entorno con configuración UTF-8
                start_new_session=True,  # Crear nuevo grupo de procesos, asegurar que se puedan terminar todos los procesos relacionados cuando se cierre el servidor
            )

            # Guardar manejador de archivo para cierre posterior
            cls._stdout_files[simulation_id] = main_log_file
            cls._stderr_files[simulation_id] = None  # Ya no se necesita stderr separado

            state.process_pid = process.pid
            state.runner_status = RunnerStatus.RUNNING
            cls._processes[simulation_id] = process
            cls._save_run_state(state)

            # Capture locale before spawning monitor thread
            current_locale = get_locale()

            # Iniciar hilo de monitoreo
            monitor_thread = threading.Thread(
                target=cls._monitor_simulation,
                args=(simulation_id, current_locale),
                daemon=True,
            )
            monitor_thread.start()
            cls._monitor_threads[simulation_id] = monitor_thread

            logger.info(
                f"Simulación iniciada exitosamente: {simulation_id}, pid={process.pid}, platform={platform}"
            )

        except Exception as e:
            state.runner_status = RunnerStatus.FAILED
            state.error = str(e)
            cls._save_run_state(state)
            raise

        return state

    @classmethod
    def _monitor_simulation(cls, simulation_id: str, locale: str = "es"):
        """Monitorear proceso de simulación, analizar registro de acciones"""
        set_locale(locale)
        sim_dir = os.path.join(cls.RUN_STATE_DIR, simulation_id)

        # Nueva estructura de registro: registros de acciones por plataforma
        twitter_actions_log = os.path.join(sim_dir, "twitter", "actions.jsonl")
        reddit_actions_log = os.path.join(sim_dir, "reddit", "actions.jsonl")

        process = cls._processes.get(simulation_id)
        state = cls.get_run_state(simulation_id)

        if not process or not state:
            return

        twitter_position = 0
        reddit_position = 0

        try:
            while process.poll() is None:  # Proceso aún en ejecución
                # Leer registro de acciones de Twitter
                if os.path.exists(twitter_actions_log):
                    twitter_position = cls._read_action_log(
                        twitter_actions_log, twitter_position, state, "twitter"
                    )

                # Leer registro de acciones de Reddit
                if os.path.exists(reddit_actions_log):
                    reddit_position = cls._read_action_log(
                        reddit_actions_log, reddit_position, state, "reddit"
                    )

                # Actualizar estado
                cls._save_run_state(state)
                time.sleep(2)

            # Después de que finaliza el proceso, leer registro por última vez
            if os.path.exists(twitter_actions_log):
                cls._read_action_log(
                    twitter_actions_log, twitter_position, state, "twitter"
                )
            if os.path.exists(reddit_actions_log):
                cls._read_action_log(
                    reddit_actions_log, reddit_position, state, "reddit"
                )

            # Finalización del proceso
            exit_code = process.returncode

            if exit_code == 0:
                state.runner_status = RunnerStatus.COMPLETED
                state.completed_at = datetime.now().isoformat()
                logger.info(f"Simulación completada: {simulation_id}")
            else:
                state.runner_status = RunnerStatus.FAILED
                # Leer información de error desde el archivo de registro principal
                main_log_path = os.path.join(sim_dir, "simulation.log")
                error_info = ""
                try:
                    if os.path.exists(main_log_path):
                        with open(main_log_path, "r", encoding="utf-8") as f:
                            error_info = f.read()[
                                -2000:
                            ]  # Tomar los últimos 2000 caracteres
                except Exception:
                    pass
                state.error = (
                    f"Código de salida del proceso: {exit_code}, error: {error_info}"
                )
                logger.error(
                    f"Simulación fallida: {simulation_id}, error={state.error}"
                )

            state.twitter_running = False
            state.reddit_running = False
            cls._save_run_state(state)

        except Exception as e:
            logger.error(
                f"Excepción en hilo de monitoreo: {simulation_id}, error={str(e)}"
            )
            state.runner_status = RunnerStatus.FAILED
            state.error = str(e)
            cls._save_run_state(state)

        finally:
            # Detener actualizador de memoria de grafo
            if cls._graph_memory_enabled.get(simulation_id, False):
                try:
                    ZepGraphMemoryManager.stop_updater(simulation_id)
                    logger.info(
                        f"Actualización de memoria de grafo detenida: simulation_id={simulation_id}"
                    )
                except Exception as e:
                    logger.error(
                        f"Fallo al detener actualizador de memoria de grafo: {e}"
                    )
                cls._graph_memory_enabled.pop(simulation_id, None)

            # Limpiar recursos del proceso
            cls._processes.pop(simulation_id, None)
            cls._action_queues.pop(simulation_id, None)

            # Cerrar manejadores de archivos de registro
            if simulation_id in cls._stdout_files:
                try:
                    cls._stdout_files[simulation_id].close()
                except Exception:
                    pass
                cls._stdout_files.pop(simulation_id, None)
            if simulation_id in cls._stderr_files and cls._stderr_files[simulation_id]:
                try:
                    cls._stderr_files[simulation_id].close()
                except Exception:
                    pass
                cls._stderr_files.pop(simulation_id, None)

    @classmethod
    def _read_action_log(
        cls, log_path: str, position: int, state: SimulationRunState, platform: str
    ) -> int:
        """
        Leer archivo de registro de acciones

        Args:
            log_path: Ruta del archivo de registro
            position: Ultima posicion leida
            state: Objeto de estado de ejecucion
            platform: Nombre de plataforma (twitter/reddit)

        Returns:
            Nueva posicion de lectura
        """
        # Verificar si la actualizacion de memoria de grafo esta habilitada
        graph_memory_enabled = cls._graph_memory_enabled.get(state.simulation_id, False)
        graph_updater = None
        if graph_memory_enabled:
            graph_updater = ZepGraphMemoryManager.get_updater(state.simulation_id)

        try:
            with open(log_path, "r", encoding="utf-8") as f:
                f.seek(position)
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            action_data = json.loads(line)

                            # Procesar entradas del tipo de evento
                            if "event_type" in action_data:
                                event_type = action_data.get("event_type")

                                # Detectar evento simulation_end y marcar plataforma como completada
                                if event_type == "simulation_end":
                                    if platform == "twitter":
                                        state.twitter_completed = True
                                        state.twitter_running = False
                                        logger.info(
                                            f"Twitter simulacion completada: {state.simulation_id}, total_rounds={action_data.get('total_rounds')}, total_actions={action_data.get('total_actions')}"
                                        )
                                    elif platform == "reddit":
                                        state.reddit_completed = True
                                        state.reddit_running = False
                                        logger.info(
                                            f"Reddit simulacion completada: {state.simulation_id}, total_rounds={action_data.get('total_rounds')}, total_actions={action_data.get('total_actions')}"
                                        )

                                    # Verificar si todas las plataformas habilitadas completaron
                                    # Si solo se ejecuto una plataforma, verificar solo esa plataforma
                                    # Si se ejecutaron dos plataformas, ambas deben completar
                                    all_completed = cls._check_all_platforms_completed(
                                        state
                                    )
                                    if all_completed:
                                        state.runner_status = RunnerStatus.COMPLETED
                                        state.completed_at = datetime.now().isoformat()
                                        logger.info(
                                            f"Todas las plataformas de simulacion completadas: {state.simulation_id}"
                                        )

                                # Actualizar informacion de ronda (desde evento round_end)
                                elif event_type == "round_end":
                                    round_num = action_data.get("round", 0)
                                    simulated_hours = action_data.get(
                                        "simulated_hours", 0
                                    )

                                    # Actualizar rondas y tiempos independientes por plataforma
                                    if platform == "twitter":
                                        if round_num > state.twitter_current_round:
                                            state.twitter_current_round = round_num
                                        state.twitter_simulated_hours = simulated_hours
                                    elif platform == "reddit":
                                        if round_num > state.reddit_current_round:
                                            state.reddit_current_round = round_num
                                        state.reddit_simulated_hours = simulated_hours

                                    # Ronda total: maximo de las dos plataformas
                                    if round_num > state.current_round:
                                        state.current_round = round_num
                                    # Tiempo total: maximo de las dos plataformas
                                    state.simulated_hours = max(
                                        state.twitter_simulated_hours,
                                        state.reddit_simulated_hours,
                                    )

                                continue

                            action = AgentAction(
                                round_num=action_data.get("round", 0),
                                timestamp=action_data.get(
                                    "timestamp", datetime.now().isoformat()
                                ),
                                platform=platform,
                                agent_id=action_data.get("agent_id", 0),
                                agent_name=action_data.get("agent_name", ""),
                                action_type=action_data.get("action_type", ""),
                                action_args=action_data.get("action_args", {}),
                                result=action_data.get("result"),
                                success=action_data.get("success", True),
                            )
                            state.add_action(action)

                            # Actualizar ronda
                            if (
                                action.round_num
                                and action.round_num > state.current_round
                            ):
                                state.current_round = action.round_num

                            # Si la actualizacion de memoria de grafo esta habilitada, enviar actividad a Zep
                            if graph_updater:
                                graph_updater.add_activity_from_dict(
                                    action_data, platform
                                )

                        except json.JSONDecodeError:
                            pass
                return f.tell()
        except Exception as e:
            logger.warning(f"Fall al leer registro de acciones: {log_path}, error={e}")
            return position

    @classmethod
    def _check_all_platforms_completed(cls, state: SimulationRunState) -> bool:
        """
        Verificar si todas las plataformas habilitadas completaron la simulacion

        Determinar si la plataforma esta habilitada verificando si existe el archivo actions.jsonl correspondiente

        Returns:
            True si todas las plataformas habilitadas completaron
        """
        sim_dir = os.path.join(cls.RUN_STATE_DIR, state.simulation_id)
        twitter_log = os.path.join(sim_dir, "twitter", "actions.jsonl")
        reddit_log = os.path.join(sim_dir, "reddit", "actions.jsonl")

        # Verificar cuales plataformas estan habilitadas (verificando si el archivo existe)
        twitter_enabled = os.path.exists(twitter_log)
        reddit_enabled = os.path.exists(reddit_log)

        # Si la plataforma esta habilitada pero no completada, devolver False
        if twitter_enabled and not state.twitter_completed:
            return False
        if reddit_enabled and not state.reddit_completed:
            return False

        # Al menos una plataforma esta habilitada y completada
        return twitter_enabled or reddit_enabled

    @classmethod
    def _terminate_process(
        cls, process: subprocess.Popen, simulation_id: str, timeout: int = 10
    ):
        """
        Terminar proceso y sus subprocesos de forma multiplataforma

        Args:
            process: Proceso a terminar
            simulation_id: ID de simulacion (para registro)
            timeout: Tiempo de espera para que el proceso termine (segundos)
        """
        if IS_WINDOWS:
            # Windows: usar taskkill para terminar proceso y subprocesos
            # /F = forzar terminacion, /T = terminar proceso y subprocesos
            logger.info(
                f"Terminar proceso (Windows): simulation={simulation_id}, pid={process.pid}"
            )
            try:
                # Primero intentar terminacion elegante
                subprocess.run(
                    ["taskkill", "/PID", str(process.pid), "/T"],
                    capture_output=True,
                    timeout=5,
                )
                try:
                    process.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    # Forzar terminacion
                    logger.warning(
                        f"Proceso sin respuesta, forzar terminacion: {simulation_id}"
                    )
                    subprocess.run(
                        ["taskkill", "/F", "/PID", str(process.pid), "/T"],
                        capture_output=True,
                        timeout=5,
                    )
                    process.wait(timeout=5)
            except Exception as e:
                logger.warning(f"taskkill fallido, intentar terminate: {e}")
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
        else:
            # Unix: usar grupo de proceso para terminacion
            # debido a que se uso start_new_session=True, el ID del grupo de proceso es igual al PID del proceso principal
            pgid = os.getpgid(process.pid)
            logger.info(
                f"Terminar grupo de proceso (Unix): simulation={simulation_id}, pgid={pgid}"
            )

            # Primero enviar SIGTERM a todo el grupo de proceso
            os.killpg(pgid, signal.SIGTERM)

            try:
                process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                # Si no termina despues del tiempo de espera, forzar envio de SIGKILL
                logger.warning(
                    f"Grupo de proceso sin respuesta a SIGTERM, forzar terminacion: {simulation_id}"
                )
                os.killpg(pgid, signal.SIGKILL)
                process.wait(timeout=5)

    @classmethod
    def stop_simulation(cls, simulation_id: str) -> SimulationRunState:
        """Detener simulacion"""
        state = cls.get_run_state(simulation_id)
        if not state:
            raise ValueError(f"La simulacion no existe: {simulation_id}")

        if state.runner_status not in [RunnerStatus.RUNNING, RunnerStatus.PAUSED]:
            raise ValueError(
                f"La simulacion no esta en ejecucion: {simulation_id}, status={state.runner_status}"
            )

        state.runner_status = RunnerStatus.STOPPING
        cls._save_run_state(state)

        # Terminar proceso
        process = cls._processes.get(simulation_id)
        if process and process.poll() is None:
            try:
                cls._terminate_process(process, simulation_id)
            except ProcessLookupError:
                # El proceso ya no existe
                pass
            except Exception as e:
                logger.error(
                    f"Fall al terminar grupo de proceso: {simulation_id}, error={e}"
                )
                # Regresar a terminacion directa del proceso
                try:
                    process.terminate()
                    process.wait(timeout=5)
                except Exception:
                    process.kill()

        state.runner_status = RunnerStatus.STOPPED
        state.twitter_running = False
        state.reddit_running = False
        state.completed_at = datetime.now().isoformat()
        cls._save_run_state(state)

        # Detener actualizador de memoria de grafo
        if cls._graph_memory_enabled.get(simulation_id, False):
            try:
                ZepGraphMemoryManager.stop_updater(simulation_id)
                logger.info(
                    f"Actualizador de memoria de grafo detenido: simulation_id={simulation_id}"
                )
            except Exception as e:
                logger.error(f"Fall al detener actualizador de memoria de grafo: {e}")
            cls._graph_memory_enabled.pop(simulation_id, None)

        logger.info(f"Simulacion detenida: {simulation_id}")
        return state

    @classmethod
    def _read_actions_from_file(
        cls,
        file_path: str,
        default_platform: Optional[str] = None,
        platform_filter: Optional[str] = None,
        agent_id: Optional[int] = None,
        round_num: Optional[int] = None,
    ) -> List[AgentAction]:
        """
        Leer acciones desde archivo de acciones individual

        Args:
            file_path: Ruta del archivo de registro de acciones
            default_platform: Plataforma por defecto (cuando la accion no tiene campo platform)
            platform_filter: Filtrar plataforma
            agent_id: Filtrar ID de Agente
            round_num: Filtrar ronda
        """
        if not os.path.exists(file_path):
            return []

        actions = []

        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                try:
                    data = json.loads(line)

                    # Saltar registros no de acciones (como simulation_start, round_start, round_end, etc.)
                    if "event_type" in data:
                        continue

                    # Saltar registros sin agent_id (no son acciones de Agente)
                    if "agent_id" not in data:
                        continue

                    # Obtener plataforma: usar platform del registro, o la plataforma por defecto
                    record_platform = data.get("platform") or default_platform or ""

                    # Filtrar
                    if platform_filter and record_platform != platform_filter:
                        continue
                    if agent_id is not None and data.get("agent_id") != agent_id:
                        continue
                    if round_num is not None and data.get("round") != round_num:
                        continue

                    actions.append(
                        AgentAction(
                            round_num=data.get("round", 0),
                            timestamp=data.get("timestamp", ""),
                            platform=record_platform,
                            agent_id=data.get("agent_id", 0),
                            agent_name=data.get("agent_name", ""),
                            action_type=data.get("action_type", ""),
                            action_args=data.get("action_args", {}),
                            result=data.get("result"),
                            success=data.get("success", True),
                        )
                    )

                except json.JSONDecodeError:
                    continue

        return actions

    @classmethod
    def get_all_actions(
        cls,
        simulation_id: str,
        platform: Optional[str] = None,
        agent_id: Optional[int] = None,
        round_num: Optional[int] = None,
    ) -> List[AgentAction]:
        """
        Obtener historial completo de acciones de todas las plataformas (sin limite de paginacion)

        Args:
            simulation_id: ID de simulacion
            platform: Filtrar plataforma (twitter/reddit)
            agent_id: Filtrar Agente
            round_num: Filtrar ronda

        Returns:
            Lista completa de acciones ordenada por timestamp (nuevas primero)
        """
        sim_dir = os.path.join(cls.RUN_STATE_DIR, simulation_id)
        actions = []

        # Leer archivo de acciones de Twitter (se establece platform automaticamente como twitter segun ruta)
        twitter_actions_log = os.path.join(sim_dir, "twitter", "actions.jsonl")
        if not platform or platform == "twitter":
            actions.extend(
                cls._read_actions_from_file(
                    twitter_actions_log,
                    default_platform="twitter",  # Auto completar campo platform
                    platform_filter=platform,
                    agent_id=agent_id,
                    round_num=round_num,
                )
            )

        # Leer archivo de acciones de Reddit (se establece platform automaticamente como reddit segun ruta)
        reddit_actions_log = os.path.join(sim_dir, "reddit", "actions.jsonl")
        if not platform or platform == "reddit":
            actions.extend(
                cls._read_actions_from_file(
                    reddit_actions_log,
                    default_platform="reddit",  # Auto completar campo platform
                    platform_filter=platform,
                    agent_id=agent_id,
                    round_num=round_num,
                )
            )

        # Si no hay archivo de plataforma, intentar leer formato antiguo individual
        if not actions:
            actions_log = os.path.join(sim_dir, "actions.jsonl")
            actions = cls._read_actions_from_file(
                actions_log,
                default_platform=None,  # El archivo de formato antiguo debe tener campo platform
                platform_filter=platform,
                agent_id=agent_id,
                round_num=round_num,
            )

        # Ordenar por timestamp (nuevas primero)
        actions.sort(key=lambda x: x.timestamp, reverse=True)

        return actions

    @classmethod
    def get_actions(
        cls,
        simulation_id: str,
        limit: int = 100,
        offset: int = 0,
        platform: Optional[str] = None,
        agent_id: Optional[int] = None,
        round_num: Optional[int] = None,
    ) -> List[AgentAction]:
        """
        Obtener historial de acciones (con paginacion)

        Args:
            simulation_id: ID de simulacion
            limit: Limite de cantidad devuelta
            offset: Offset
            platform: Filtrar plataforma
            agent_id: Filtrar Agente
            round_num: Filtrar ronda

        Returns:
            Lista de acciones
        """
        actions = cls.get_all_actions(
            simulation_id=simulation_id,
            platform=platform,
            agent_id=agent_id,
            round_num=round_num,
        )

        # Paginacion
        return actions[offset : offset + limit]

    @classmethod
    def get_timeline(
        cls, simulation_id: str, start_round: int = 0, end_round: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Obtener linea de tiempo de simulacion (resumido por ronda)

        Args:
            simulation_id: ID de simulacion
            start_round: Ronda inicial
            end_round: Ronda final

        Returns:
            Informacion resumida de cada ronda
        """
        actions = cls.get_actions(simulation_id, limit=10000)

        # Agrupar por ronda
        rounds: Dict[int, Dict[str, Any]] = {}

        for action in actions:
            round_num = action.round_num

            if round_num < start_round:
                continue
            if end_round is not None and round_num > end_round:
                continue

            if round_num not in rounds:
                rounds[round_num] = {
                    "round_num": round_num,
                    "twitter_actions": 0,
                    "reddit_actions": 0,
                    "active_agents": set(),
                    "action_types": {},
                    "first_action_time": action.timestamp,
                    "last_action_time": action.timestamp,
                }

            r = rounds[round_num]

            if action.platform == "twitter":
                r["twitter_actions"] += 1
            else:
                r["reddit_actions"] += 1

            r["active_agents"].add(action.agent_id)
            r["action_types"][action.action_type] = (
                r["action_types"].get(action.action_type, 0) + 1
            )
            r["last_action_time"] = action.timestamp

        # Convertir a lista
        result = []
        for round_num in sorted(rounds.keys()):
            r = rounds[round_num]
            result.append(
                {
                    "round_num": round_num,
                    "twitter_actions": r["twitter_actions"],
                    "reddit_actions": r["reddit_actions"],
                    "total_actions": r["twitter_actions"] + r["reddit_actions"],
                    "active_agents_count": len(r["active_agents"]),
                    "active_agents": list(r["active_agents"]),
                    "action_types": r["action_types"],
                    "first_action_time": r["first_action_time"],
                    "last_action_time": r["last_action_time"],
                }
            )

        return result

    @classmethod
    def get_agent_stats(cls, simulation_id: str) -> List[Dict[str, Any]]:
        """
        Obtener informacion de estadisticas de cada Agente

        Returns:
            Lista de estadisticas de Agente
        """
        actions = cls.get_actions(simulation_id, limit=10000)

        agent_stats: Dict[int, Dict[str, Any]] = {}

        for action in actions:
            agent_id = action.agent_id

            if agent_id not in agent_stats:
                agent_stats[agent_id] = {
                    "agent_id": agent_id,
                    "agent_name": action.agent_name,
                    "total_actions": 0,
                    "twitter_actions": 0,
                    "reddit_actions": 0,
                    "action_types": {},
                    "first_action_time": action.timestamp,
                    "last_action_time": action.timestamp,
                }

            stats = agent_stats[agent_id]
            stats["total_actions"] += 1

            if action.platform == "twitter":
                stats["twitter_actions"] += 1
            else:
                stats["reddit_actions"] += 1

            stats["action_types"][action.action_type] = (
                stats["action_types"].get(action.action_type, 0) + 1
            )
            stats["last_action_time"] = action.timestamp

        # Ordenar por cantidad total de acciones
        result = sorted(
            agent_stats.values(), key=lambda x: x["total_actions"], reverse=True
        )

        return result

    @classmethod
    def cleanup_simulation_logs(cls, simulation_id: str) -> Dict[str, Any]:
        """
        Limpiar registros de ejecucion de simulacion (para reiniciar forzosamente)

        Eliminara los siguientes archivos:
        - run_state.json
        - twitter/actions.jsonl
        - reddit/actions.jsonl
        - simulation.log
        - stdout.log / stderr.log
        - twitter_simulation.db (base de datos de plataforma Twitter)
        - reddit_simulation.db (base de datos de plataforma Reddit)
        - env_status.json (archivo de estado del entorno)

        Nota: no eliminara archivos de configuracion (simulation_config.json) ni archivos de perfil

        Args:
            simulation_id: ID de simulacion

        Returns:
            Informacion del resultado de limpieza
        """
        import shutil

        sim_dir = os.path.join(cls.RUN_STATE_DIR, simulation_id)

        if not os.path.exists(sim_dir):
            return {
                "success": True,
                "message": "Directorio de simulacion no existe, no necesita limpieza",
            }

        cleaned_files = []
        errors = []

        # Lista de archivos a eliminar (incluyendo archivos de base de datos)
        files_to_delete = [
            "run_state.json",
            "simulation.log",
            "stdout.log",
            "stderr.log",
            "twitter_simulation.db",  # Base de datos de plataforma Twitter
            "reddit_simulation.db",  # Base de datos de plataforma Reddit
            "env_status.json",  # Archivo de estado del entorno
        ]

        # Lista de directorios a eliminar (contiene registros de acciones)
        dirs_to_clean = ["twitter", "reddit"]

        # Eliminar archivos
        for filename in files_to_delete:
            file_path = os.path.join(sim_dir, filename)
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    cleaned_files.append(filename)
                except Exception as e:
                    errors.append(f"Eliminar {filename} fallido: {str(e)}")

        # Limpiar registros de acciones en directorio de plataforma
        for dir_name in dirs_to_clean:
            dir_path = os.path.join(sim_dir, dir_name)
            if os.path.exists(dir_path):
                actions_file = os.path.join(dir_path, "actions.jsonl")
                if os.path.exists(actions_file):
                    try:
                        os.remove(actions_file)
                        cleaned_files.append(f"{dir_name}/actions.jsonl")
                    except Exception as e:
                        errors.append(
                            f"Eliminar {dir_name}/actions.jsonl fallido: {str(e)}"
                        )

        # Limpiar estado de ejecucion de memoria
        if simulation_id in cls._run_states:
            del cls._run_states[simulation_id]

        logger.info(
            f"Limpieza de registros de simulacion completada: {simulation_id}, archivos eliminados: {cleaned_files}"
        )

        return {
            "success": len(errors) == 0,
            "cleaned_files": cleaned_files,
            "errors": errors if errors else None,
        }

    # Bandera para prevenir limpieza duplicada
    _cleanup_done = False

    @classmethod
    def cleanup_all_simulations(cls):
        """
        Limpiar todos los procesos de simulacion en ejecucion

        Llamar al cerrar el servidor para asegurar que todos los subprocesos sean terminados
        """
        # Prevenir limpieza duplicada
        if cls._cleanup_done:
            return
        cls._cleanup_done = True

        # Verificar si hay contenido que limpiar (evitar que procesos vacios impriman registros innecesarios)
        has_processes = bool(cls._processes)
        has_updaters = bool(cls._graph_memory_enabled)

        if not has_processes and not has_updaters:
            return  # No hay contenido que limpiar, devolver silenciosamente

        logger.info("Limpiando todos los procesos de simulacion...")

        # Primero detener todos los actualizadores de memoria de grafo (stop_all imprimira registros internamente)
        try:
            ZepGraphMemoryManager.stop_all()
        except Exception as e:
            logger.error(f"Fall al detener el actualizador de memoria de grafo: {e}")
        cls._graph_memory_enabled.clear()

        # Copiar diccionario para evitar modificacion durante iteracion
        processes = list(cls._processes.items())

        for simulation_id, process in processes:
            try:
                if process.poll() is None:  # Proceso aun en ejecucion
                    logger.info(
                        f"Terminar proceso de simulacion: {simulation_id}, pid={process.pid}"
                    )

                    try:
                        # Usar metodo de terminacion de proceso multiplataforma
                        cls._terminate_process(process, simulation_id, timeout=5)
                    except (ProcessLookupError, OSError):
                        # Proceso posiblemente ya no existe, intentar terminacion directa
                        try:
                            process.terminate()
                            process.wait(timeout=3)
                        except Exception:
                            process.kill()

                    # Actualizar run_state.json
                    state = cls.get_run_state(simulation_id)
                    if state:
                        state.runner_status = RunnerStatus.STOPPED
                        state.twitter_running = False
                        state.reddit_running = False
                        state.completed_at = datetime.now().isoformat()
                        state.error = "Servidor cerrado, simulacion fue terminada"
                        cls._save_run_state(state)

                    # Tambien actualizar state.json, establecer estado como stopped
                    try:
                        sim_dir = os.path.join(cls.RUN_STATE_DIR, simulation_id)
                        state_file = os.path.join(sim_dir, "state.json")
                        logger.info(f"Intentar actualizar state.json: {state_file}")
                        if os.path.exists(state_file):
                            with open(state_file, "r", encoding="utf-8") as f:
                                state_data = json.load(f)
                            state_data["status"] = "stopped"
                            state_data["updated_at"] = datetime.now().isoformat()
                            with open(state_file, "w", encoding="utf-8") as f:
                                json.dump(state_data, f, indent=2, ensure_ascii=False)
                            logger.info(
                                f"Ya se actualizo state.json con estado stopped: {simulation_id}"
                            )
                        else:
                            logger.warning(f"state.json no existe: {state_file}")
                    except Exception as state_err:
                        logger.warning(
                            f"Fall al actualizar state.json: {simulation_id}, error={state_err}"
                        )

            except Exception as e:
                logger.error(f"Fall al limpiar proceso: {simulation_id}, error={e}")

        # Limpiar manejadores de archivos
        for simulation_id, file_handle in list(cls._stdout_files.items()):
            try:
                if file_handle:
                    file_handle.close()
            except Exception:
                pass
        cls._stdout_files.clear()

        for simulation_id, file_handle in list(cls._stderr_files.items()):
            try:
                if file_handle:
                    file_handle.close()
            except Exception:
                pass
        cls._stderr_files.clear()

        # Limpiar estado de memoria
        cls._processes.clear()
        cls._action_queues.clear()

        logger.info("Limpieza de procesos de simulacion completada")

    @classmethod
    def register_cleanup(cls):
        """
        Registrar funcion de limpieza

        Llamar al iniciar aplicacion Flask para asegurar limpieza al cerrar servidor
        """
        global _cleanup_registered

        if _cleanup_registered:
            return

        # En modo debug de Flask, solo registrar en subproceso reloader (el proceso que ejecuta la aplicacion real)
        # WERKZEUG_RUN_MAIN=true indica que es el subproceso reloader
        # Si no es modo debug, no existe esta variable de entorno, y tambien necesita registrar
        is_reloader_process = os.environ.get("WERKZEUG_RUN_MAIN") == "true"
        is_debug_mode = (
            os.environ.get("FLASK_DEBUG") == "1"
            or os.environ.get("WERKZEUG_RUN_MAIN") is not None
        )

        # En modo debug, solo registrar en subproceso reloader; en modo no-debug siempre registrar
        if is_debug_mode and not is_reloader_process:
            _cleanup_registered = True  # Marcar como registrado, prevenir que el subproceso intente de nuevo
            return

        # Guardar el procesador de seal original
        original_sigint = signal.getsignal(signal.SIGINT)
        original_sigterm = signal.getsignal(signal.SIGTERM)
        # SIGHUP solo existe en sistema Unix (macOS/Linux), Windows no tiene
        original_sighup = None
        has_sighup = hasattr(signal, "SIGHUP")
        if has_sighup:
            original_sighup = signal.getsignal(signal.SIGHUP)

        def cleanup_handler(signum=None, frame=None):
            """Procesador de seal: primero limpiar proceso de simulacion, luego llamar al procesador original"""
            # Solo imprimir registro cuando hay procesos que limpiar
            if cls._processes or cls._graph_memory_enabled:
                logger.info(f"Seal recibida {signum}, iniciar limpieza...")
            cls.cleanup_all_simulations()

            # Llamar al procesador de seal original, dejar que Flask salga normalmente
            if signum == signal.SIGINT and callable(original_sigint):
                original_sigint(signum, frame)
            elif signum == signal.SIGTERM and callable(original_sigterm):
                original_sigterm(signum, frame)
            elif has_sighup and signum == signal.SIGHUP:
                # SIGHUP: enviado al cerrar terminal
                if callable(original_sighup):
                    original_sighup(signum, frame)
                else:
                    # Comportamiento por defecto: salida normal
                    sys.exit(0)
            else:
                # Si el procesador original no es invocable (como SIG_DFL), usar comportamiento por defecto
                raise KeyboardInterrupt

        # Registrar procesador atexit como respaldo
        atexit.register(cls.cleanup_all_simulations)

        # Registrar procesador de seal (solo en hilo principal)
        try:
            # SIGTERM: seal por defecto de comando kill
            signal.signal(signal.SIGTERM, cleanup_handler)
            # SIGINT: Ctrl+C
            signal.signal(signal.SIGINT, cleanup_handler)
            # SIGHUP: cierre de terminal (solo sistema Unix)
            if has_sighup:
                signal.signal(signal.SIGHUP, cleanup_handler)
        except ValueError:
            # No esta en hilo principal, solo puede usar atexit
            logger.warning(
                "No se puede registrar procesador de seal (no esta en hilo principal), usando atexit"
            )

        _cleanup_registered = True

    @classmethod
    def get_running_simulations(cls) -> List[str]:
        """
        Obtener lista de IDs de simulacion en ejecucion
        """
        running = []
        for sim_id, process in cls._processes.items():
            if process.poll() is None:
                running.append(sim_id)
        return running

    # ============== Funcion de Interview ==============

    @classmethod
    def check_env_alive(cls, simulation_id: str) -> bool:
        """
        Verificar si el entorno de simulacion esta activo (puede recibir comandos de Interview)

        Args:
            simulation_id: ID de simulacion

        Returns:
            True si el entorno esta activo, False si el entorno ya cerro
        """
        sim_dir = os.path.join(cls.RUN_STATE_DIR, simulation_id)
        if not os.path.exists(sim_dir):
            return False

        ipc_client = SimulationIPCClient(sim_dir)
        return ipc_client.check_env_alive()

    @classmethod
    def get_env_status_detail(cls, simulation_id: str) -> Dict[str, Any]:
        """
        Obtener informacion detallada del estado del entorno de simulacion

        Args:
            simulation_id: ID de simulacion

        Returns:
            Diccionario de detalle de estado, contiene status, twitter_available, reddit_available, timestamp
        """
        sim_dir = os.path.join(cls.RUN_STATE_DIR, simulation_id)
        status_file = os.path.join(sim_dir, "env_status.json")

        default_status = {
            "status": "stopped",
            "twitter_available": False,
            "reddit_available": False,
            "timestamp": None,
        }

        if not os.path.exists(status_file):
            return default_status

        try:
            with open(status_file, "r", encoding="utf-8") as f:
                status = json.load(f)
            return {
                "status": status.get("status", "stopped"),
                "twitter_available": status.get("twitter_available", False),
                "reddit_available": status.get("reddit_available", False),
                "timestamp": status.get("timestamp"),
            }
        except (json.JSONDecodeError, OSError):
            return default_status

    @classmethod
    def interview_agent(
        cls,
        simulation_id: str,
        agent_id: int,
        prompt: str,
        platform: str = None,
        timeout: float = 60.0,
    ) -> Dict[str, Any]:
        """
        Entrevistar un solo Agente

        Args:
            simulation_id: ID de simulacion
            agent_id: ID de Agente
            prompt: Pregunta de entrevista
            platform: Especificar plataforma (opcional)
                - "twitter": Solo entrevistar plataforma Twitter
                - "reddit": Solo entrevistar plataforma Reddit
                - None: En simulacion dual, entrevistar ambas plataformas simultaneamente, devolver resultado integrado
            timeout: Tiempo de espera agotado (segundos)

        Returns:
            Diccionario de resultado de entrevista

        Raises:
            ValueError: La simulacion no existe o el entorno no esta en ejecucion
            TimeoutError: Tiempo de espera agotado esperando respuesta
        """
        sim_dir = os.path.join(cls.RUN_STATE_DIR, simulation_id)
        if not os.path.exists(sim_dir):
            raise ValueError(f"La simulacion no existe: {simulation_id}")

        ipc_client = SimulationIPCClient(sim_dir)

        if not ipc_client.check_env_alive():
            raise ValueError(
                f"El entorno de simulacion no esta en ejecucion o ya Cerro, no se puede ejecutar Interview: {simulation_id}"
            )

        logger.info(
            f"Enviar comando de Interview: simulation_id={simulation_id}, agent_id={agent_id}, platform={platform}"
        )

        response = ipc_client.send_interview(
            agent_id=agent_id, prompt=prompt, platform=platform, timeout=timeout
        )

        if response.status.value == "completed":
            return {
                "success": True,
                "agent_id": agent_id,
                "prompt": prompt,
                "result": response.result,
                "timestamp": response.timestamp,
            }
        else:
            return {
                "success": False,
                "agent_id": agent_id,
                "prompt": prompt,
                "error": response.error,
                "timestamp": response.timestamp,
            }

    @classmethod
    def interview_agents_batch(
        cls,
        simulation_id: str,
        interviews: List[Dict[str, Any]],
        platform: str = None,
        timeout: float = 120.0,
    ) -> Dict[str, Any]:
        """
        Entrevistar multiples Agentes en lote

        Args:
            simulation_id: ID de simulacion
            interviews: Lista de entrevistas, cada elemento contiene {"agent_id": int, "prompt": str, "platform": str(opcional)}
            platform: Plataforma por defecto (opcional, sera sobrescrito por platform de cada entrevista)
                - "twitter": Por defecto solo entrevistar plataforma Twitter
                - "reddit": Por defecto solo entrevistar plataforma Reddit
                - None: En simulacion dual, cada Agente es entrevistado en ambas plataformas simultaneamente
            timeout: Tiempo de espera agotado (segundos)

        Returns:
            Diccionario de resultado de lote de entrevistas

        Raises:
            ValueError: La simulacion no existe o el entorno no esta en ejecucion
            TimeoutError: Tiempo de espera agotado esperando respuesta
        """
        sim_dir = os.path.join(cls.RUN_STATE_DIR, simulation_id)
        if not os.path.exists(sim_dir):
            raise ValueError(f"La simulacion no existe: {simulation_id}")

        ipc_client = SimulationIPCClient(sim_dir)

        if not ipc_client.check_env_alive():
            raise ValueError(
                f"El entorno de simulacion no esta en ejecucion o ya Cerro, no se puede ejecutar Interview: {simulation_id}"
            )

        logger.info(
            f"Enviar comando de Interview en lote: simulation_id={simulation_id}, count={len(interviews)}, platform={platform}"
        )

        response = ipc_client.send_batch_interview(
            interviews=interviews, platform=platform, timeout=timeout
        )

        if response.status.value == "completed":
            return {
                "success": True,
                "interviews_count": len(interviews),
                "result": response.result,
                "timestamp": response.timestamp,
            }
        else:
            return {
                "success": False,
                "interviews_count": len(interviews),
                "error": response.error,
                "timestamp": response.timestamp,
            }

    @classmethod
    def interview_all_agents(
        cls,
        simulation_id: str,
        prompt: str,
        platform: str = None,
        timeout: float = 180.0,
    ) -> Dict[str, Any]:
        """
        Entrevistar todos los Agentes (entrevista global)

        Usar la misma pregunta para entrevistar a todos los Agentes en la simulacion

        Args:
            simulation_id: ID de simulacion
            prompt: Pregunta de entrevista (todos los Agentes usan la misma pregunta)
            platform: Especificar plataforma (opcional)
                - "twitter": Solo entrevistar plataforma Twitter
                - "reddit": Solo entrevistar plataforma Reddit
                - None: En simulacion dual, cada Agente es entrevistado en ambas plataformas simultaneamente
            timeout: Tiempo de espera agotado (segundos)

        Returns:
            Diccionario de resultado de entrevista global
        """
        sim_dir = os.path.join(cls.RUN_STATE_DIR, simulation_id)
        if not os.path.exists(sim_dir):
            raise ValueError(f"La simulacion no existe: {simulation_id}")

        # Obtener informacion de todos los Agentes desde archivo de configuracion
        config_path = os.path.join(sim_dir, "simulation_config.json")
        if not os.path.exists(config_path):
            raise ValueError(f"Configuracion de simulacion no existe: {simulation_id}")

        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        agent_configs = config.get("agent_configs", [])
        if not agent_configs:
            raise ValueError(
                f"Configuracion de simulacion no tiene Agentes: {simulation_id}"
            )

        # Construir lista de entrevistas en lote
        interviews = []
        for agent_config in agent_configs:
            agent_id = agent_config.get("agent_id")
            if agent_id is not None:
                interviews.append({"agent_id": agent_id, "prompt": prompt})

        logger.info(
            f"Enviar comando de Interview global: simulation_id={simulation_id}, agent_count={len(interviews)}, platform={platform}"
        )

        return cls.interview_agents_batch(
            simulation_id=simulation_id,
            interviews=interviews,
            platform=platform,
            timeout=timeout,
        )

    @classmethod
    def close_simulation_env(
        cls, simulation_id: str, timeout: float = 30.0
    ) -> Dict[str, Any]:
        """
        Cerrar entorno de simulacion (en lugar de detener proceso de simulacion)

        Enviar comando de cierre de entorno a la simulacion para salida elegante en modo comando

        Args:
            simulation_id: ID de simulacion
            timeout: Tiempo de espera agotado (segundos)

        Returns:
            Diccionario de resultado de operacion
        """
        sim_dir = os.path.join(cls.RUN_STATE_DIR, simulation_id)
        if not os.path.exists(sim_dir):
            raise ValueError(f"La simulacion no existe: {simulation_id}")

        ipc_client = SimulationIPCClient(sim_dir)

        if not ipc_client.check_env_alive():
            return {"success": True, "message": "El entorno ya Cerro"}

        logger.info(
            f"Enviar comando de cierre de entorno: simulation_id={simulation_id}"
        )

        try:
            response = ipc_client.send_close_env(timeout=timeout)

            return {
                "success": response.status.value == "completed",
                "message": "Comando de cierre de entorno ya fue enviado",
                "result": response.result,
                "timestamp": response.timestamp,
            }
        except TimeoutError:
            # Tiempo de espera agotado posiblemente porque el entorno esta cerrando
            return {
                "success": True,
                "message": "Comando de cierre de entorno ya fue enviado (tiempo de espera agotado esperando respuesta, el entorno puede estar cerrando)",
            }

    @classmethod
    def _get_interview_history_from_db(
        cls,
        db_path: str,
        platform_name: str,
        agent_id: Optional[int] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Obtener historial de Interview desde base de datos individual"""
        import sqlite3

        if not os.path.exists(db_path):
            return []

        results = []

        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            if agent_id is not None:
                cursor.execute(
                    """
                    SELECT user_id, info, created_at
                    FROM trace
                    WHERE action = 'interview' AND user_id = ?
                    ORDER BY created_at DESC
                    LIMIT ?
                """,
                    (agent_id, limit),
                )
            else:
                cursor.execute(
                    """
                    SELECT user_id, info, created_at
                    FROM trace
                    WHERE action = 'interview'
                    ORDER BY created_at DESC
                    LIMIT ?
                """,
                    (limit,),
                )

            for user_id, info_json, created_at in cursor.fetchall():
                try:
                    info = json.loads(info_json) if info_json else {}
                except json.JSONDecodeError:
                    info = {"raw": info_json}

                results.append(
                    {
                        "agent_id": user_id,
                        "response": info.get("response", info),
                        "prompt": info.get("prompt", ""),
                        "timestamp": created_at,
                        "platform": platform_name,
                    }
                )

            conn.close()

        except Exception as e:
            logger.error(f"Fall al leer historial de Interview ({platform_name}): {e}")

        return results

    @classmethod
    def get_interview_history(
        cls,
        simulation_id: str,
        platform: str = None,
        agent_id: Optional[int] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Obtener historial de registros de Interview (desde base de datos)

        Args:
            simulation_id: ID de simulacion
            platform: Tipo de plataforma (reddit/twitter/None)
                - "reddit": Solo obtener historial de plataforma Reddit
                - "twitter": Solo obtener historial de plataforma Twitter
                - None: Obtener historial de ambas plataformas
            agent_id: Especificar ID de Agente (opcional, solo obtener historial de ese Agente)
            limit: Limite de cantidad devuelta por plataforma

        Returns:
            Lista de registros de historial de Interview
        """
        sim_dir = os.path.join(cls.RUN_STATE_DIR, simulation_id)

        results = []

        # Determinar plataformas a consultar
        if platform in ("reddit", "twitter"):
            platforms = [platform]
        else:
            # Cuando no se especifica platform, consultar ambas plataformas
            platforms = ["twitter", "reddit"]

        for p in platforms:
            db_path = os.path.join(sim_dir, f"{p}_simulation.db")
            platform_results = cls._get_interview_history_from_db(
                db_path=db_path, platform_name=p, agent_id=agent_id, limit=limit
            )
            results.extend(platform_results)

        # Ordenar por tiempo en orden descendente
        results.sort(key=lambda x: x.get("timestamp", ""), reverse=True)

        # Si se consultaron multiples plataformas, limitar total
        if len(platforms) > 1 and len(results) > limit:
            results = results[:limit]

        return results
