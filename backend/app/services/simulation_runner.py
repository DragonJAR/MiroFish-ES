"""
Ejecutor de simulación OASIS
Ejecuta simulación en segundo plano y registra acciones de cada Agente, soporta monitoreo de estado en tiempo real
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

    # Marcar si se ha registrado función de limpieza
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
    """Resumen de cada ronda"""

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

    # Rondas independientes por plataforma y tiempo de simulación (para visualización paralela de doble plataforma)
    twitter_current_round: int = 0
    reddit_current_round: int = 0
    twitter_simulated_hours: int = 0
    reddit_simulated_hours: int = 0

    # Estado de plataforma
    twitter_running: bool = False
    reddit_running: bool = False
    twitter_actions_count: int = 0
    reddit_actions_count: int = 0

    # Estado de completado de plataforma (detectando evento simulation_end en actions.jsonl)
    twitter_completed: bool = False
    reddit_completed: bool = False

    # Resumen de cada ronda
    rounds: List[RoundSummary] = field(default_factory=list)

    # Acciones recientes (para visualización en tiempo real en frontend)
    recent_actions: List[AgentAction] = field(default_factory=list)
    max_recent_actions: int = 50

    # Marca de tiempo
    started_at: Optional[str] = None
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    completed_at: Optional[str] = None

    # Información de error
    error: Optional[str] = None

    # ID de proceso (para detención)
    process_pid: Optional[int] = None

    def add_action(self, action: AgentAction):
        """Agregar acción a lista de acciones recientes"""
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
                self.current_round / max(self.total_rounds,1) * 100, 1
            ),
            # Rounds y tiempos independientes por plataforma
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
        """Contiene información detallada de las acciones recientes"""
        result = self.to_dict()
        result["recent_actions"] = [a.to_dict() for a in self.recent_actions]
        result["rounds_count"] = len(self.rounds)
        return result


class SimulationRunner:
    """
    Ejecutor de simulación

    Responsable de:
    1. Ejecutar simulación OASIS en proceso de segundo plano
    2. Analizar logs de ejecución, registrar acciones de cada Agente
    3. Proporcionar interfaz de consulta de estado en tiempo real
    4. Soportar pausar/detener/reiniciar operaciones
    """

    # Directorio de almacenamiento de estado de ejecución
    RUN_STATE_DIR = os.path.join(os.path.dirname(__file__), "../../uploads/simulations")

    # Directorio de scripts
    SCRIPTS_DIR = os.path.join(os.path.dirname(__file__), "../../scripts")

    # Estado de ejecución en memoria
    _run_states: Dict[str, SimulationRunState] = {}
    _processes: Dict[str, subprocess.Popen] = {}
    _action_queues: Dict[str, Queue] = {}
    _monitor_threads: Dict[str, threading.Thread] = {}
    _stdout_files: Dict[str, Any] = {}  # Almacenar manejadores de archivos stdout
    _stderr_files: Dict[str, Any] = {}  # Almacenar manejadores de archivos stderr

    # Configuración de actualización de memoria de Grafo
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
                # Rounds y tiempos independientes por plataforma
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
            logger.error(f"Error al cargar estado de ejecución: {str(e)}")
            return None

    @classmethod
    def _save_run_state(cls, state: SimulationRunState):
        """Guardar estado de ejecución a archivo"""
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
        max_rounds: int = None,  # Número máximo de rondas de simulación (opcional, para cortar simulaciones excesivamente largas)
        enable_graph_memory_update: bool = False,  # Si actualizar actividades al ZepGrafo
        graph_id: str = None,  # ZepGrafoID (requerido cuando se habilita actualización de Grafo)
    ) -> SimulationRunState:
        """
        Iniciar simulación

        Args:
            simulation_id: ID de simulación
            platform: Plataforma de ejecución (twitter/reddit/parallel)
            max_rounds: Número máximo de rondas de simulación (opcional, para cortar simulaciones excesivamente largas)
            enable_graph_memory_update: Si actualizar dinámicamente actividades de Agente a ZepGrafo
            graph_id: ZepGrafoID (requerido cuando se habilita actualización de Grafo)

        Returns:
            SimulationRunState
        """
        # Verificar si ya está ejecutándose
        existing = cls.get_run_state(simulation_id)
        if existing and existing.runner_status in [
            RunnerStatus.RUNNING,
            RunnerStatus.STARTING,
        ]:
            raise ValueError(f"La simulación ya está ejecutándose: {simulation_id}")

        # Cargar configuración de simulación
        sim_dir = os.path.join(cls.RUN_STATE_DIR, simulation_id)
        config_path = os.path.join(sim_dir, "simulation_config.json")

        if not os.path.exists(config_path):
            raise ValueError(f"Configuración de simulación no encontrada. Por favor llama /prepare primero.")

        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        # Inicializar estado de ejecución
        time_config = config.get("time_config", {})
        total_hours = time_config.get("total_simulation_hours", 72)
        minutes_per_round = time_config.get("minutes_per_round", 30)
        total_rounds = int(total_hours * 60 / minutes_per_round)

        # Si se especificó número máximo de rondas, entonces truncar
        if max_rounds is not None and max_rounds > 0:
            original_rounds = total_rounds
            total_rounds = min(total_rounds, max_rounds)
            if total_rounds < original_rounds:
                logger.info(
                    f"Número de rondas truncado: {original_rounds} -> {total_rounds} (max_rounds={max_rounds})"
                )

        state = SimulationRunState(
            simulation_id=simulation_id,
            runner_status=RunnerStatus.STARTING,
            total_rounds=total_rounds,
            total_simulation_hours=total_hours,
            started_at=datetime.now().isoformat(),
        )

        cls._save_run_state(state)

        # Si habilitar actualización de memoria de Grafo, crear actualizador
        if enable_graph_memory_update:
            if not graph_id:
                raise ValueError("启用Grafo记忆更新时必须提供 graph_id")

            try:
                ZepGraphMemoryManager.create_updater(simulation_id, graph_id)
                cls._graph_memory_enabled[simulation_id] = True
                logger.info(
                    f"Habilitada actualización de memoria de Grafo: simulation_id={simulation_id}, graph_id={graph_id}"
                )
            except Exception as e:
                logger.error(f"Falló al crear actualizador de memoria de Grafo: {e}")
                cls._graph_memory_enabled[simulation_id] = False
        else:
            cls._graph_memory_enabled[simulation_id] = False

        # Determinar cuál script ejecutar (el script se encuentra en directorio backend/scripts/)
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
            raise ValueError(f"El script no existe: {script_path}")

        # Crear cola de acciones
        action_queue = Queue()
        cls._action_queues[simulation_id] = action_queue

        # Iniciar proceso de simulación
        try:
            # Construir comando de ejecución, usar ruta completa
            # Nueva estructura de logs:
            #   twitter/actions.jsonl - Log de acciones de Twitter
            #   reddit/actions.jsonl  - Log de acciones de Reddit
            #   simulation.log        - Log del proceso principal

             cmd = [
                sys.executable,  # Python intérprete
                script_path,
                "--config",
                config_path,  # Usar ruta completa de archivo de configuración
            ]

            # Si se especificó número máximo de rondas, agregar a argumentos de línea de comandos
            if max_rounds is not None and max_rounds > 0:
                cmd.extend(["--max-rounds", str(max_rounds)])

            # Crear archivo de log principal, evitar que stdout/stderr se bloqueen cuando el buffer de tuberías esté lleno
            main_log_path = os.path.join(sim_dir, "simulation.log")
            main_log_file = open(main_log_path, "w", encoding="utf-8")

            # Configurar variables de entorno de proceso hijo, asegurar uso de codificación UTF-8 en Windows
            # Esto puede corregir problemas cuando bibliotecas de terceros (como OASIS) leen archivos sin especificar codificación
            env = os.environ.copy()
             env["PYTHONUTF8"] = "1"  # Soportado en Python 3.7+, hacer que todos los open() usen UTF-8 por defecto
            env["PYTHONIOENCODING"] = "utf-8"  # Asegurar que stdout/stderr usen UTF-8

            # Configurar directorio de trabajo como directorio de simulación (bases de datos etc. se generarán aquí)
            # Usar start_new_session=True para crear nuevo grupo de procesos, asegurar que se puedan terminar todos los subprocesos con os.killpg
            process = subprocess.Popen(
                cmd,
                cwd=sim_dir,
                stdout=main_log_file,
                stderr=subprocess.STDOUT,  # stderr 也写入同一archivos
                text=True,
                encoding="utf-8",  # Especificar codificación explícitamente
                bufsize=1,
                env=env,  # Pasar variables de entorno con configuración UTF-8
                start_new_session=True,  # Crear nuevo grupo de procesos, asegurar que al cerrar servidor se puedan terminar todos los procesos relacionados
            )

            # Guardar manejo de archivo para cerrar posteriormente
            cls._stdout_files[simulation_id] = main_log_file
            cls._stderr_files[simulation_id] = None  # Ya no se necesita stderr separado

            state.process_pid = process.pid
            state.runner_status = RunnerStatus.RUNNING
            cls._processes[simulation_id] = process
            cls._save_run_state(state)

            # Capturar locale antes de lanzar hilo de monitoreo
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
                f"Simulación iniciada con éxito: {simulation_id}, pid={process.pid}, platform={platform}"
            )

        except Exception as e:
            state.runner_status = RunnerStatus.FAILED
            state.error = str(e)
            cls._save_run_state(state)
            raise

        return state

    @classmethod
    def _monitor_simulation(cls, simulation_id: str, locale: str = "zh"):
        """Monitorear proceso de simulación, analizar logs de acciones"""
        set_locale(locale)
        sim_dir = os.path.join(cls.RUN_STATE_DIR, simulation_id)

        # Nueva estructura de logs: logs de acciones separados por plataforma
        twitter_actions_log = os.path.join(sim_dir, "twitter", "actions.jsonl")
        reddit_actions_log = os.path.join(sim_dir, "reddit", "actions.jsonl")

        process = cls._processes.get(simulation_id)
        state = cls.get_run_state(simulation_id)

        if not process or not state:
            return

        twitter_position = 0
        reddit_position = 0

        try:
            while process.poll() is None:  # El proceso aún está ejecutándose
                # Leer log de acciones de Twitter
                if os.path.exists(twitter_actions_log):
                    twitter_position = cls._read_action_log(
                        twitter_actions_log, twitter_position, state, "twitter"
                    )

                # Leer log de acciones de Reddit
                if os.path.exists(reddit_actions_log):
                    reddit_position = cls._read_action_log(
                        reddit_actions_log, reddit_position, state, "reddit"
                            )

                            # Actualizar rondas
                            if (
                                action.round_num
                                and                                 action.round_num > state.current_round
                            ):
                                state.current_round = action.round_num

                            # Si se habilitó actualización de memoria de Grafo, enviar actividad a Zep
                            if graph_updater:
                                graph_updater.add_activity_from_dict(
                                    action_data, platform
                                )

                        except json.JSONDecodeError:
                            pass
                return f.tell()
        except Exception as e:
            logger.warning(f"Falló al leer log de acciones: {log_path}, error={e}")
            return position

    @classmethod
    def _check_all_platforms_completed(cls, state: SimulationRunState) -> bool:
        """
        Verificar si todas las plataformas habilitadas han completado la simulación

        Se verifica si la plataforma está habilitada comprobando si existe el archivo actions.jsonl correspondiente

        Returns:
            True si todas las plataformas habilitadas están completadas
        """
        sim_dir = os.path.join(cls.RUN_STATE_DIR, state.simulation_id)
        twitter_log = os.path.join(sim_dir, "twitter", "actions.jsonl")
        reddit_log = os.path.join(sim_dir, "reddit", "actions.jsonl")

        # Verificar cuáles plataformas están habilitadas (juzgando por existencia de archivos)
        twitter_enabled = os.path.exists(twitter_log)
        reddit_enabled = os.path.exists(reddit_log)

        # Si la plataforma está habilitada pero no está completada, volver False
        if twitter_enabled and not state.twitter_completed:
            return False
        if reddit_enabled and not state.reddit_completed:
            return False

        # Al menos una plataforma está habilitada y completada
        return twitter_enabled or reddit_enabled

    @classmethod
    def _terminate_process(
        cls, process: subprocess.Popen, simulation_id: str, timeout: int = 10
    ):
        """
        Terminar proceso y sus subprocesos de forma multiplataforma

        Args:
            process: Proceso a terminar
            simulation_id: ID de simulación (para logs)
            timeout: Tiempo de espera para que el proceso salga (segundos)
        """
        if IS_WINDOWS:
            # Windows: Usar comando taskkill para terminar árbol de procesos
            # /F = Forzar terminación, /T = Terminar árbol de procesos (incluyendo subprocesos)
            logger.info(
                f"Terminar árbol de procesos (Windows): simulation={simulation_id}, pid={process.pid}"
            )
            try:
                # 先尝试优雅终止
                subprocess.run(
                    ["taskkill", "/PID", str(process.pid), "/T"],
                    capture_output=True,
                    timeout=5,
                )
                try:
                    process.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    # Forzar terminación
                    logger.warning(f"Proceso no respondió, forzando terminación: {simulation_id}")
                    subprocess.run(
                        ["taskkill", "/F", "/PID", str(process.pid), "/T"],
                        capture_output=True,
                        timeout=5,
                    )
                    process.wait(timeout=5)
            except Exception as e:
                logger.warning(f"Falló taskkill, intentando terminate: {e}")
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
        else:
            # Unix: Usar terminación de grupo de procesos
            # Ya que se usó start_new_session=True, ID de grupo de proceso es igual a PID del proceso principal
            pgid = os.getpgid(process.pid)
            logger.info(f"Terminar grupo de procesos (Unix): simulation={simulation_id}, pgid={pgid}")

            # Enviar primero SIGTERM a todo el grupo de procesos
            os.killpg(pgid, signal.SIGTERM)

            try:
                process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                # Si después de tiempo de espera aún no terminó, forzar SIGKILL
                logger.warning(f"Grupo de procesos no respondió a SIGTERM, forzando terminación: {simulation_id}")
                os.killpg(pgid, signal.SIGKILL)
                process.wait(timeout=5)

    @classmethod
    def stop_simulation(cls, simulation_id: str) -> SimulationRunState:
        """Detener simulación"""
        state = cls.get_run_state(simulation_id)
        if not state:
            raise ValueError(f"La simulación no existe: {simulation_id}")

        if state.runner_status not in [RunnerStatus.RUNNING, RunnerStatus.PAUSED]:
            raise ValueError(
                f"Simulación no está en ejecución: {simulation_id}, status={state.runner_status}"
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
                logger.error(f"Falló al terminar grupo de procesos: {simulation_id}, error={e}")
                # 回退到直接终止进程
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

        # Detener actualizador de memoria de Grafo
        if cls._graph_memory_enabled.get(simulation_id, False):
            try:
                ZepGraphMemoryManager.stop_updater(simulation_id)
                logger.info(f"Detenida actualización de memoria de Grafo: simulation_id={simulation_id}")
            except Exception as e:
                logger.error(f"Falló al detener actualizador de memoria de Grafo: {e}")
            cls._graph_memory_enabled.pop(simulation_id, None)

        logger.info(f"Simulación detenida: {simulation_id}")
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
        Leer acciones de un archivo de acciones

        Args:
            file_path: Ruta del archivo de log de acciones
            default_platform: Plataforma predeterminada (usar cuando no hay campo platform en registro)
            platform_filter: Filtrar por plataforma
            agent_id: Filtrar por Agent ID
            round_num: Filtrar por número de ronda
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

                    # Saltar registros que no son acciones (como simulation_start, round_start, round_end u otros eventos)
                    if "event_type" in data:
                        continue

                    # Saltar registros sin agent_id (no son acciones de Agente)
                    if "agent_id" not in data:
                        continue

                    # Obtener plataforma: priorizar usar platform del registro, de lo contrario usar plataforma predeterminada
                    record_platform = data.get("platform") or default_platform or ""

                    # Filtrado
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

        # Ordenar por timestamp (más reciente primero)
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
        Obtener historial de acciones (con paginación)

        Args:
            simulation_id: ID de simulación
            limit: Límite de cantidad a devolver
            offset: Desplazamiento
            platform: Filtrar por plataforma
            agent_id: Filtrar por Agente
            round_num: Filtrar por número de ronda

        Returns:
            Lista de acciones
        """
        actions = cls.get_all_actions(
            simulation_id=simulation_id,
            platform=platform,
            agent_id=agent_id,
            round_num=round_num,
        )

        # Paginación
        return actions[offset : offset + limit]

    @classmethod
    def get_timeline(
        cls, simulation_id: str, start_round: int = 0, end_round: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Obtener línea de tiempo de simulación (agrupado por número de rondas)

        Args:
            simulation_id: ID de simulación
            start_round: Ronda inicial
            end_round: Ronda final

        Returns:
            Información resumida por cada ronda
        """
        actions = cls.get_actions(simulation_id, limit=10000)

        # Agrupar por número de ronda
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
                r["action_types"].get(action.action_type, 0) +1
            )
            r["last_action_time"] = action.timestamp

        # Transformar a lista
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
        Obtener información estadística de cada Agente

        Returns:
            Lista de estadísticas de Agentes
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
                stats["action_types"].get(action.action_type, 0) +1
            )
            stats["last_action_time"] = action.timestamp

        # Ordenar por total de acciones
        result = sorted(
            agent_stats.values(), key=lambda x: x["total_actions"], reverse=True
        )

        return result

    @classmethod
    def cleanup_simulation_logs(cls, simulation_id: str) -> Dict[str, Any]:
        """
        Limpiar logs de ejecución de simulación (usado para reiniciar simulación forzadamente)

        Se eliminarán los siguientes archivos:
        - run_state.json
        - twitter/actions.jsonl
        - reddit/actions.jsonl
        - simulation.log
        - stdout.log / stderr.log
        - twitter_simulation.db (base de datos de simulación)
        - reddit_simulation.db (base de datos de simulación)
        - env_status.json (estado de entorno)

        Nota: No se eliminarán archivos de configuración (simulation_config.json) ni archivos profile

        Args:
            simulation_id: ID de simulación

        Returns:
            Información de resultado de limpieza
        """
        import shutil

        sim_dir = os.path.join(cls.RUN_STATE_DIR, simulation_id)

        if not os.path.exists(sim_dir):
            return {"success": True, "message": "Directorio de simulación no existe, nada que limpiar"}

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
            "env_status.json",  # Archivo de estado de entorno
        ]

        # Lista de directorios a eliminar (conteniendo logs de acciones)
        dirs_to_clean = ["twitter", "reddit"]

        # Eliminar文件
        for filename in files_to_delete:
            file_path = os.path.join(sim_dir, filename)
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    cleaned_files.append(filename)
                except Exception as e:
                    errors.append(f"Falló al eliminar {filename}: {str(e)}")

        # Limpiar logs de acciones en directorios de plataforma
        for dir_name in dirs_to_clean:
            dir_path = os.path.join(sim_dir, dir_name)
            if os.path.exists(dir_path):
                actions_file = os.path.join(dir_path, "actions.jsonl")
                if os.path.exists(actions_file):
                    try:
                        os.remove(actions_file)
                        cleaned_files.append(f"{dir_name}/actions.jsonl")
                    except Exception as e:
                        errors.append(f"Falló al eliminar {dir_name}/actions.jsonl: {str(e)}")

        # Limpiar estado de ejecución en memoria
        if simulation_id in cls._run_states:
            del cls._run_states[simulation_id]

        logger.info(f"Limpieza de logs de simulación completada: {simulation_id}, archivos eliminados: {cleaned_files}")

        return {
            "success": len(errors) == 0,
            "cleaned_files": cleaned_files,
            "errors": errors if errors else None,
        }

    # Bandera para prevenir limpieza repetida
    _cleanup_done = False

    @classmethod
    def cleanup_all_simulations(cls):
        """
        Limpiar todos los procesos de simulación en ejecución

        Se llama al cerrar el servidor, asegurar que todos los subprocesos sean terminados
        """
        # Prevenir limpieza repetida
        if cls._cleanup_done:
            return
        cls._cleanup_done = True

        # Verificar si hay contenido que necesita limpieza (evitar imprimir logs cuando no haya procesos)
        has_processes = bool(cls._processes)
        has_updaters = bool(cls._graph_memory_enabled)

        if not has_processes and not has_updaters:
            return # No hay contenido que limpiar, volver silenciosamente

        logger.info("Limpiando todos los procesos de simulación...")

        # Primero detener todos los actualizadores de memoria de Grafo (stop_all imprimirá logs internamente)
        try:
            ZepGraphMemoryManager.stop_all()
        except Exception as e:
            logger.error(f"Falló al detener actualizadores de memoria de Grafo: {e}")
        cls._graph_memory_enabled.clear()

        # Copiar diccionario para evitar modificar durante iteración
        processes = list(cls._processes.items())

        for simulation_id, process in processes:
            try:
                if process.poll() is None:  # El proceso aún está ejecutándose
                    logger.info(f"Terminar proceso de simulación: {simulation_id}, pid={process.pid}")

                    try:
                        # Usar método de terminación de proceso multiplataforma
                        cls._terminate_process(process, simulation_id, timeout=5)
                    except (ProcessLookupError, OSError):
                        # El proceso ya puede no existir, intentar terminar directamente
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
                        state.error = "Servidor cerrado, simulación terminada"
                        cls._save_run_state(state)

                    # Simultáneamente actualizar state.json, establecer estado como stopped
                    try:</think><tool_call>bash<arg_key>command</arg_key><arg_value>python3 -c "import re; print('Chinese characters remaining:', len(re.findall(r'[\u4e00-\u9fff]', open('/Users/jaimearestrepo/Proyectos/MiroFish-ES/backend/app/services/simulation_runner.py').read())))"
                        sim_dir = os.path.join(cls.RUN_STATE_DIR, simulation_id)
                        state_file = os.path.join(sim_dir, "state.json")
                        logger.info(f"尝试更新 state.json: {state_file}")
                        if os.path.exists(state_file):
                            with open(state_file, "r", encoding="utf-8") as f:
                                state_data = json.load(f)
                            state_data["status"] = "stopped"
                            state_data["updated_at"] = datetime.now().isoformat()
                            with open(state_file, "w", encoding="utf-8") as f:
                                json.dump(state_data, f, indent=2, ensure_ascii=False)
                            logger.info(
                                f"已更新 state.json 状态为 stopped: {simulation_id}"
                            )
                        else:
                            logger.warning(f"state.json 不存在: {state_file}")
                    except Exception as state_err:
                        logger.warning(
                            f"更新 state.json Fallido: {simulation_id}, error={state_err}"
                        )

            except Exception as e:
                logger.error(f"清理进程Fallido: {simulation_id}, error={e}")

        # 清理文件句柄
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

        # 清理内存中的状态
        cls._processes.clear()
        cls._action_queues.clear()

        logger.info("模拟进程清理完成")

    @classmethod
    def register_cleanup(cls):
        """
        注册清理函数

        在 Flask 应用启动时调用，确保服务器Cerrar时清理所有模拟进程
        """
        global _cleanup_registered

        if _cleanup_registered:
            return

        # Flask debug 模式下，只在 reloader 子进程中注册清理（实际运行应用的进程）
        # WERKZEUG_RUN_MAIN=true 表示是 reloader 子进程
        # 如果不是 debug 模式，则没有这elementos环境变量，也需要注册
        is_reloader_process = os.environ.get("WERKZEUG_RUN_MAIN") == "true"
        is_debug_mode = (
            os.environ.get("FLASK_DEBUG") == "1"
            or os.environ.get("WERKZEUG_RUN_MAIN") is not None
        )

        # 在 debug 模式下，只在 reloader 子进程中注册；非 debug 模式下始终注册
        if is_debug_mode and not is_reloader_process:
            _cleanup_registered = True  # Marcar已注册，防止子进程再次尝试
            return

        # Guardar原有的信号处理器
        original_sigint = signal.getsignal(signal.SIGINT)
        original_sigterm = signal.getsignal(signal.SIGTERM)
        # SIGHUP 只在 Unix 系统存在（macOS/Linux），Windows 没有
        original_sighup = None
        has_sighup = hasattr(signal, "SIGHUP")
        if has_sighup:
            original_sighup = signal.getsignal(signal.SIGHUP)

        def cleanup_handler(signum=None, frame=None):
            """信号处理器：先清理模拟进程，再调用原处理器"""
            # 只有在有进程需要清理时才打印日志
            if cls._processes or cls._graph_memory_enabled:
                logger.info(f"收到信号 {signum}，开始清理...")
            cls.cleanup_all_simulations()

            # 调用原有的信号处理器，让 Flask 正常退出
            if signum == signal.SIGINT and callable(original_sigint):
                original_sigint(signum, frame)
            elif signum == signal.SIGTERM and callable(original_sigterm):
                original_sigterm(signum, frame)
            elif has_sighup and signum == signal.SIGHUP:
                # SIGHUP: 终端Cerrar时发送
                if callable(original_sighup):
                    original_sighup(signum, frame)
                else:
                    # Por defecto行为：正常退出
                    sys.exit(0)
            else:
                # 如果原处理器不可调用（如 SIG_DFL），则使用默认行为
                raise KeyboardInterrupt

        # Registrarse atexit 处理器（作为备用）
        atexit.register(cls.cleanup_all_simulations)

        # Registrarse信号处理器（仅在主线程中）
        try:
            # SIGTERM: kill 命令默认信号
            signal.signal(signal.SIGTERM, cleanup_handler)
            # SIGINT: Ctrl+C
            signal.signal(signal.SIGINT, cleanup_handler)
            # SIGHUP: 终端Cerrar（仅 Unix 系统）
            if has_sighup:
                signal.signal(signal.SIGHUP, cleanup_handler)
        except ValueError:
            # 不在主线程中，只能使用 atexit
            logger.warning("Ninguno法注册信号处理器（不在主线程），仅使用 atexit")

        _cleanup_registered = True

    @classmethod
    def get_running_simulations(cls) -> List[str]:
        """
        获取所有正在运行的模拟ID列表
        """
        running = []
        for sim_id, process in cls._processes.items():
            if process.poll() is None:
                running.append(sim_id)
        return running

    # ============== Interview 功能 ==============

    @classmethod
    def check_env_alive(cls, simulation_id: str) -> bool:
        """
        检查模拟环境是否存活（可以接收Interview命令）

        Args:
            simulation_id: 模拟ID

        Returns:
            True 表示环境存活，False 表示环境已Cerrar
        """
        sim_dir = os.path.join(cls.RUN_STATE_DIR, simulation_id)
        if not os.path.exists(sim_dir):
            return False

        ipc_client = SimulationIPCClient(sim_dir)
        return ipc_client.check_env_alive()

    @classmethod
    def get_env_status_detail(cls, simulation_id: str) -> Dict[str, Any]:
        """
        获取模拟环境的详细状态信息

        Args:
            simulation_id: 模拟ID

        Returns:
            状态详情字典，包含 status, twitter_available, reddit_available, timestamp
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
        采访单elementosAgent

        Args:
            simulation_id: 模拟ID
            agent_id: Agent ID
            prompt: 采访问题
            platform: 指定平台（可选）
                - "twitter": 只采访Twitter平台
                - "reddit": 只采访Reddit平台
                - None: 双平台模拟时同时采访两elementos平台，Volver整合结果
            timeout: 超时时间（秒）

        Returns:
            采访结果字典

        Raises:
            ValueError: La simulación no existe或环境未运行
            TimeoutError: Pendiente响应超时
        """
        sim_dir = os.path.join(cls.RUN_STATE_DIR, simulation_id)
        if not os.path.exists(sim_dir):
            raise ValueError(f"La simulación no existe: {simulation_id}")

        ipc_client = SimulationIPCClient(sim_dir)

        if not ipc_client.check_env_alive():
            raise ValueError(
                f"模拟Entorno no ejecutándose o cerrado，Ninguno法执行Interview: {simulation_id}"
            )

        logger.info(
            f"发送Interview命令: simulation_id={simulation_id}, agent_id={agent_id}, platform={platform}"
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
        批量采访多elementosAgent

        Args:
            simulation_id: 模拟ID
            interviews: 采访列表，每elementos元素包含 {"agent_id": int, "prompt": str, "platform": str(可选)}
            platform: 默认平台（可选，会被每elementos采访项的platform覆盖）
                - "twitter": 默认只采访Twitter平台
                - "reddit": 默认只采访Reddit平台
                - None: 双平台模拟时每elementosAgent同时采访两elementos平台
            timeout: 超时时间（秒）

        Returns:
            批量采访结果字典

        Raises:
            ValueError: La simulación no existe或环境未运行
            TimeoutError: Pendiente响应超时
        """
        sim_dir = os.path.join(cls.RUN_STATE_DIR, simulation_id)
        if not os.path.exists(sim_dir):
            raise ValueError(f"La simulación no existe: {simulation_id}")

        ipc_client = SimulationIPCClient(sim_dir)

        if not ipc_client.check_env_alive():
            raise ValueError(
                f"模拟Entorno no ejecutándose o cerrado，Ninguno法执行Interview: {simulation_id}"
            )

        logger.info(
            f"发送批量Interview命令: simulation_id={simulation_id}, count={len(interviews)}, platform={platform}"
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
        采访所有Agent（全局采访）

        使用相同的问题采访模拟中的所有Agent

        Args:
            simulation_id: 模拟ID
            prompt: 采访问题（所有Agent使用相同问题）
            platform: 指定平台（可选）
                - "twitter": 只采访Twitter平台
                - "reddit": 只采访Reddit平台
                - None: 双平台模拟时每elementosAgent同时采访两elementos平台
            timeout: 超时时间（秒）

        Returns:
            全局采访结果字典
        """
        sim_dir = os.path.join(cls.RUN_STATE_DIR, simulation_id)
        if not os.path.exists(sim_dir):
            raise ValueError(f"La simulación no existe: {simulation_id}")

        # 从配置文件获取所有Agent信息
        config_path = os.path.join(sim_dir, "simulation_config.json")
        if not os.path.exists(config_path):
            raise ValueError(f"模拟配置不存在: {simulation_id}")

        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        agent_configs = config.get("agent_configs", [])
        if not agent_configs:
            raise ValueError(f"模拟配置中没有Agent: {simulation_id}")

        # 构建批量采访列表
        interviews = []
        for agent_config in agent_configs:
            agent_id = agent_config.get("agent_id")
            if agent_id is not None:
                interviews.append({"agent_id": agent_id, "prompt": prompt})

        logger.info(
            f"发送全局Interview命令: simulation_id={simulation_id}, agent_count={len(interviews)}, platform={platform}"
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
        Cerrar模拟环境（而不是停止模拟进程）

        向模拟发送Cerrar环境命令，使其优雅退出Pendiente命令模式

        Args:
            simulation_id: 模拟ID
            timeout: 超时时间（秒）

        Returns:
            操作结果字典
        """
        sim_dir = os.path.join(cls.RUN_STATE_DIR, simulation_id)
        if not os.path.exists(sim_dir):
            raise ValueError(f"La simulación no existe: {simulation_id}")

        ipc_client = SimulationIPCClient(sim_dir)

        if not ipc_client.check_env_alive():
            return {"success": True, "message": "环境已经Cerrar"}

        logger.info(f"发送Cerrar环境命令: simulation_id={simulation_id}")

        try:
            response = ipc_client.send_close_env(timeout=timeout)

            return {
                "success": response.status.value == "completed",
                "message": "环境Cerrar命令已发送",
                "result": response.result,
                "timestamp": response.timestamp,
            }
        except TimeoutError:
            # Tiempo agotado可能是因为环境正在Cerrar
            return {
                "success": True,
                "message": "环境Cerrar命令已发送（Pendiente响应超时，环境可能正在Cerrar）",
            }

    @classmethod
    def _get_interview_history_from_db(
        cls,
        db_path: str,
        platform_name: str,
        agent_id: Optional[int] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """从单elementos数据库获取Interview历史"""
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
            logger.error(f"读取Interview历史Fallido ({platform_name}): {e}")

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
        获取Interview历史记录（从数据库读取）

        Args:
            simulation_id: 模拟ID
            platform: 平台类型（reddit/twitter/None）
                - "reddit": 只获取Reddit平台的历史
                - "twitter": 只获取Twitter平台的历史
                - None: 获取两elementos平台的所有历史
            agent_id: 指定Agent ID（可选，只获取该Agent的历史）
            limit: 每elementos平台Volver数量限制

        Returns:
            Interview历史记录列表
        """
        sim_dir = os.path.join(cls.RUN_STATE_DIR, simulation_id)

        results = []

        # 确定要查询的平台
        if platform in ("reddit", "twitter"):
            platforms = [platform]
        else:
            # 不指定platform时，查询两elementos平台
            platforms = ["twitter", "reddit"]

        for p in platforms:
            db_path = os.path.join(sim_dir, f"{p}_simulation.db")
            platform_results = cls._get_interview_history_from_db(
                db_path=db_path, platform_name=p, agent_id=agent_id, limit=limit
            )
            results.extend(platform_results)

        # 按时间降序排序
        results.sort(key=lambda x: x.get("timestamp", ""), reverse=True)

        # 如果查询了多elementos平台，限制总数
        if len(platforms) > 1 and len(results) > limit:
            results = results[:limit]

        return results
