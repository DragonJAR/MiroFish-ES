"""
Script de preset de simulación OASIS Reddit
Este script lee los parámetros del archivo de configuración para ejecutar la simulación, logrando automatización completa

Características:
- Después de completar la simulación, no cerrar el entorno inmediatamente, entrar en modo de espera de comandos
- Soporte para recibir comandos de Interview a través de IPC
- Soporte para entrevista de un solo Agent y entrevistas por lotes
- Soporte para comando de cierre remoto del entorno

Uso:
    python run_reddit_simulation.py --config /path/to/simulation_config.json
    python run_reddit_simulation.py --config /path/to/simulation_config.json --no-wait  # Cerrar inmediatamente después de completar
"""

import argparse
import asyncio
import json
import logging
import os
import random
import signal
import sys
import sqlite3
from datetime import datetime
from typing import Dict, Any, List, Optional

# Variables globales: para manejo de señales
_shutdown_event = None
_cleanup_done = False

# Agregar ruta del proyecto
_scripts_dir = os.path.dirname(os.path.abspath(__file__))
_backend_dir = os.path.abspath(os.path.join(_scripts_dir, ".."))
_project_root = os.path.abspath(os.path.join(_backend_dir, ".."))
sys.path.insert(0, _scripts_dir)
sys.path.insert(0, _backend_dir)

# Cargar archivo .env del directorio raíz del proyecto (contiene LLM_API_KEY y otros configs)
from dotenv import load_dotenv

_env_file = os.path.join(_project_root, ".env")
if os.path.exists(_env_file):
    load_dotenv(_env_file)
else:
    _backend_env = os.path.join(_backend_dir, ".env")
    if os.path.exists(_backend_env):
        load_dotenv(_backend_env)


import re


class UnicodeFormatter(logging.Formatter):
    """Formateador personalizado, convierte secuencias de escape Unicode a caracteres legibles"""

    UNICODE_ESCAPE_PATTERN = re.compile(r"\\u([0-9a-fA-F]{4})")

    def format(self, record):
        result = super().format(record)

        def replace_unicode(match):
            try:
                return chr(int(match.group(1), 16))
            except (ValueError, OverflowError):
                return match.group(0)

        return self.UNICODE_ESCAPE_PATTERN.sub(replace_unicode, result)


class MaxTokensWarningFilter(logging.Filter):
    """Filter out camel-ai warnings about max_tokens (we intentionally don't configure max_tokens, letting the model decide)"""

    def filter(self, record):
        # Filter out max_tokens warnings
        if (
            "max_tokens" in record.getMessage()
            and "Invalid or missing" in record.getMessage()
        ):
            return False
        return True


# Add filter at module load time, ensure it takes effect before camel code executes
logging.getLogger().addFilter(MaxTokensWarningFilter())


def setup_oasis_logging(log_dir: str):
    """Configure OASIS logging, use fixed filename log files"""
    os.makedirs(log_dir, exist_ok=True)

    # Clean old log files
    for f in os.listdir(log_dir):
        old_log = os.path.join(log_dir, f)
        if os.path.isfile(old_log) and f.endswith(".log"):
            try:
                os.remove(old_log)
            except OSError:
                pass

    formatter = UnicodeFormatter("%(levelname)s - %(asctime)s - %(name)s - %(message)s")

    loggers_config = {
        "social.agent": os.path.join(log_dir, "social.agent.log"),
        "social.twitter": os.path.join(log_dir, "social.twitter.log"),
        "social.rec": os.path.join(log_dir, "social.rec.log"),
        "oasis.env": os.path.join(log_dir, "oasis.env.log"),
        "table": os.path.join(log_dir, "table.log"),
    }

    for logger_name, log_file in loggers_config.items():
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.DEBUG)
        logger.handlers.clear()
        file_handler = logging.FileHandler(log_file, encoding="utf-8", mode="w")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.propagate = False


try:
    from camel.models import ModelFactory
    from camel.types import ModelPlatformType
    import oasis
    from oasis import ActionType, LLMAction, ManualAction, generate_reddit_agent_graph
except ImportError as e:
    print(f"Error: Falta dependencia {e}")
    print("Por favor instala primero: pip install oasis-ai camel-ai")
    sys.exit(1)


# Constantes relacionadas con IPC
IPC_COMMANDS_DIR = "ipc_commands"
IPC_RESPONSES_DIR = "ipc_responses"
ENV_STATUS_FILE = "env_status.json"


class CommandType:
    """Constantes de tipo de comando"""

    INTERVIEW = "interview"
    BATCH_INTERVIEW = "batch_interview"
    CLOSE_ENV = "close_env"


class IPCHandler:
    """IPC command handler"""

    def __init__(self, simulation_dir: str, env, agent_graph):
        self.simulation_dir = simulation_dir
        self.env = env
        self.agent_graph = agent_graph
        self.commands_dir = os.path.join(simulation_dir, IPC_COMMANDS_DIR)
        self.responses_dir = os.path.join(simulation_dir, IPC_RESPONSES_DIR)
        self.status_file = os.path.join(simulation_dir, ENV_STATUS_FILE)
        self._running = True

        # Ensure directories exist
        os.makedirs(self.commands_dir, exist_ok=True)
        os.makedirs(self.responses_dir, exist_ok=True)

    def update_status(self, status: str):
        """Update environment status"""
        with open(self.status_file, "w", encoding="utf-8") as f:
            json.dump(
                {"status": status, "timestamp": datetime.now().isoformat()},
                f,
                ensure_ascii=False,
                indent=2,
            )

    def poll_command(self) -> Optional[Dict[str, Any]]:
        """Poll for pending commands"""
        if not os.path.exists(self.commands_dir):
            return None

        # Get command files (sorted by time)
        command_files = []
        for filename in os.listdir(self.commands_dir):
            if filename.endswith(".json"):
                filepath = os.path.join(self.commands_dir, filename)
                command_files.append((filepath, os.path.getmtime(filepath)))

        command_files.sort(key=lambda x: x[1])

        for filepath, _ in command_files:
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                continue

        return None

    def send_response(
        self, command_id: str, status: str, result: Dict = None, error: str = None
    ):
        """Enviar respuesta"""
        response = {
            "command_id": command_id,
            "status": status,
            "result": result,
            "error": error,
            "timestamp": datetime.now().isoformat(),
        }

        response_file = os.path.join(self.responses_dir, f"{command_id}.json")
        with open(response_file, "w", encoding="utf-8") as f:
            json.dump(response, f, ensure_ascii=False, indent=2)

        # Eliminar archivo de comando
        command_file = os.path.join(self.commands_dir, f"{command_id}.json")
        try:
            os.remove(command_file)
        except OSError:
            pass

    async def handle_interview(
        self, command_id: str, agent_id: int, prompt: str
    ) -> bool:
        """
        Procesar comando de entrevista de un solo Agent

        Returns:
            True significa éxito, False significa fracaso
        """
        try:
            # Obtener Agent
            agent = self.agent_graph.get_agent(agent_id)

            # Crear acción de Interview
            interview_action = ManualAction(
                action_type=ActionType.INTERVIEW, action_args={"prompt": prompt}
            )

            # Ejecutar Interview
            actions = {agent: interview_action}
            await self.env.step(actions)

            # Obtener resultado de la base de datos
            result = self._get_interview_result(agent_id)

            self.send_response(command_id, "completed", result=result)
            print(f"  Interview completado: agent_id={agent_id}")
            return True

        except Exception as e:
            error_msg = str(e)
            print(f"  Interview fallido: agent_id={agent_id}, error={error_msg}")
            self.send_response(command_id, "failed", error=error_msg)
            return False

    async def handle_batch_interview(
        self, command_id: str, interviews: List[Dict]
    ) -> bool:
        """
        Procesar comando de entrevistas por lotes

        Args:
            interviews: [{"agent_id": int, "prompt": str}, ...]
        """
        try:
            # Construir diccionario de acciones
            actions = {}
            agent_prompts = {}  # Registrar prompt de cada agent

            for interview in interviews:
                agent_id = interview.get("agent_id")
                prompt = interview.get("prompt", "")

                try:
                    agent = self.agent_graph.get_agent(agent_id)
                    actions[agent] = ManualAction(
                        action_type=ActionType.INTERVIEW, action_args={"prompt": prompt}
                    )
                    agent_prompts[agent_id] = prompt
                except Exception as e:
                    print(f"  Advertencia: No se pudo obtener Agent {agent_id}: {e}")

            if not actions:
                self.send_response(command_id, "failed", error="No hay Agents válidos")
                return False

            # Ejecutar Interview por lotes
            await self.env.step(actions)

            # Obtener todos los resultados
            results = {}
            for agent_id in agent_prompts.keys():
                result = self._get_interview_result(agent_id)
                results[agent_id] = result

            self.send_response(
                command_id,
                "completed",
                result={"interviews_count": len(results), "results": results},
            )
            print(f"  Interview por lotes completado: {len(results)} Agents")
            return True

        except Exception as e:
            error_msg = str(e)
            print(f"  Interview por lotes fallido: {error_msg}")
            self.send_response(command_id, "failed", error=error_msg)
            return False

    def _get_interview_result(self, agent_id: int) -> Dict[str, Any]:
        """Obtener el último resultado de Interview de la base de datos"""
        db_path = os.path.join(self.simulation_dir, "reddit_simulation.db")

        result = {"agent_id": agent_id, "response": None, "timestamp": None}

        if not os.path.exists(db_path):
            return result

        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            # Consultar el último registro de Interview
            cursor.execute(
                """
                SELECT user_id, info, created_at
                FROM trace
                WHERE action = ? AND user_id = ?
                ORDER BY created_at DESC
                LIMIT 1
            """,
                (ActionType.INTERVIEW.value, agent_id),
            )

            row = cursor.fetchone()
            if row:
                user_id, info_json, created_at = row
                try:
                    info = json.loads(info_json) if info_json else {}
                    result["response"] = info.get("response", info)
                    result["timestamp"] = created_at
                except json.JSONDecodeError:
                    result["response"] = info_json

            conn.close()

        except Exception as e:
            print(f"  Error al leer resultado de Interview: {e}")

        return result

    async def process_commands(self) -> bool:
        """
        Procesar todos los comandos pendientes

        Returns:
            True significa continuar ejecutando, False significa salir
        """
        command = self.poll_command()
        if not command:
            return True

        command_id = command.get("command_id")
        command_type = command.get("command_type")
        args = command.get("args", {})

        print(f"\nRecibido comando IPC: {command_type}, id={command_id}")

        if command_type == CommandType.INTERVIEW:
            await self.handle_interview(
                command_id, args.get("agent_id", 0), args.get("prompt", "")
            )
            return True

        elif command_type == CommandType.BATCH_INTERVIEW:
            await self.handle_batch_interview(command_id, args.get("interviews", []))
            return True

        elif command_type == CommandType.CLOSE_ENV:
            print("Recibido comando de cierre de entorno")
            self.send_response(
                command_id,
                "completed",
                result={"message": "El entorno se cerrará pronto"},
            )
            return False

        else:
            self.send_response(
                command_id,
                "failed",
                error=f"Tipo de comando desconocido: {command_type}",
            )
            return True


class RedditSimulationRunner:
    """Ejecutor de simulación Reddit"""

    # Reddit acciones disponibles (no incluye INTERVIEW, INTERVIEW solo se puede activar manualmente a través de ManualAction)
    AVAILABLE_ACTIONS = [
        ActionType.LIKE_POST,
        ActionType.DISLIKE_POST,
        ActionType.CREATE_POST,
        ActionType.CREATE_COMMENT,
        ActionType.LIKE_COMMENT,
        ActionType.DISLIKE_COMMENT,
        ActionType.SEARCH_POSTS,
        ActionType.SEARCH_USER,
        ActionType.TREND,
        ActionType.REFRESH,
        ActionType.DO_NOTHING,
        ActionType.FOLLOW,
        ActionType.MUTE,
    ]

    def __init__(self, config_path: str, wait_for_commands: bool = True):
        """
        Inicializar ejecutor de simulación

        Args:
            config_path: Ruta del archivo de configuración (simulation_config.json)
            wait_for_commands: Si esperar comandos después de la simulación (por defecto True)
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.simulation_dir = os.path.dirname(config_path)
        self.wait_for_commands = wait_for_commands
        self.env = None
        self.agent_graph = None
        self.ipc_handler = None

    def _load_config(self) -> Dict[str, Any]:
        """Cargar archivo de configuración"""
        with open(self.config_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _get_profile_path(self) -> str:
        """Obtener ruta del archivo Profile"""
        return os.path.join(self.simulation_dir, "reddit_profiles.json")

    def _get_db_path(self) -> str:
        """Obtener ruta de la base de datos"""
        return os.path.join(self.simulation_dir, "reddit_simulation.db")

    def _create_model(self):
        """
        Crear modelo LLM

        Usar configuración del archivo .env del directorio raíz del proyecto (prioridad más alta):
        - LLM_API_KEY: Clave API
        - LLM_BASE_URL: URL base de API
        - LLM_MODEL_NAME: Nombre del modelo
        """
        # Preferir leer configuración de .env
        llm_api_key = os.environ.get("LLM_API_KEY", "")
        llm_base_url = os.environ.get("LLM_BASE_URL", "")
        llm_model = os.environ.get("LLM_MODEL_NAME", "")

        # Si no hay en .env, usar config como respaldo
        if not llm_model:
            llm_model = self.config.get("llm_model", "gpt-4o-mini")

        # Establecer variables de entorno requeridas por camel-ai
        if llm_api_key:
            os.environ["OPENAI_API_KEY"] = llm_api_key

        if not os.environ.get("OPENAI_API_KEY"):
            raise ValueError(
                "Falta configuración de API Key, por favor establece LLM_API_KEY en el archivo .env del directorio raíz del proyecto"
            )

        if llm_base_url:
            os.environ["OPENAI_API_BASE_URL"] = llm_base_url

        print(
            f"Config LLM: model={llm_model}, base_url={llm_base_url[:40] if llm_base_url else 'por defecto'}..."
        )

        return ModelFactory.create(
            model_platform=ModelPlatformType.OPENAI,
            model_type=llm_model,
        )

    def _get_active_agents_for_round(
        self, env, current_hour: int, round_num: int
    ) -> List:
        """
        Determinar qué Agents activar en esta ronda según el tiempo y la configuración
        """
        time_config = self.config.get("time_config", {})
        agent_configs = self.config.get("agent_configs", [])

        base_min = time_config.get("agents_per_hour_min", 5)
        base_max = time_config.get("agents_per_hour_max", 20)

        peak_hours = time_config.get("peak_hours", [9, 10, 11, 14, 15, 20, 21, 22])
        off_peak_hours = time_config.get("off_peak_hours", [0, 1, 2, 3, 4, 5])

        if current_hour in peak_hours:
            multiplier = time_config.get("peak_activity_multiplier", 1.5)
        elif current_hour in off_peak_hours:
            multiplier = time_config.get("off_peak_activity_multiplier", 0.3)
        else:
            multiplier = 1.0

        target_count = int(random.uniform(base_min, base_max) * multiplier)

        candidates = []
        for cfg in agent_configs:
            agent_id = cfg.get("agent_id", 0)
            active_hours = cfg.get("active_hours", list(range(8, 23)))
            activity_level = cfg.get("activity_level", 0.5)

            if current_hour not in active_hours:
                continue

            if random.random() < activity_level:
                candidates.append(agent_id)

        selected_ids = (
            random.sample(candidates, min(target_count, len(candidates)))
            if candidates
            else []
        )

        active_agents = []
        for agent_id in selected_ids:
            try:
                agent = env.agent_graph.get_agent(agent_id)
                active_agents.append((agent_id, agent))
            except Exception:
                pass

        return active_agents

    async def run(self, max_rounds: int = None):
        """Ejecutar simulación Reddit

        Args:
            max_rounds: Máximo número de rondas de simulación (opcional, para truncar simulaciones muy largas)
        """
        print("=" * 60)
        print("Simulación OASIS Reddit")
        print(f"Archivo de configuración: {self.config_path}")
        print(f"ID de simulación: {self.config.get('simulation_id', 'unknown')}")
        print(
            f"Modo de espera de comandos: {'Activado' if self.wait_for_commands else 'Desactivado'}"
        )
        print("=" * 60)

        time_config = self.config.get("time_config", {})
        total_hours = time_config.get("total_simulation_hours", 72)
        minutes_per_round = time_config.get("minutes_per_round", 30)
        total_rounds = (total_hours * 60) // minutes_per_round

        # Si se especifica máximo de rondas, truncar
        if max_rounds is not None and max_rounds > 0:
            original_rounds = total_rounds
            total_rounds = min(total_rounds, max_rounds)
            if total_rounds < original_rounds:
                print(
                    f"\nRondas truncadas: {original_rounds} -> {total_rounds} (max_rounds={max_rounds})"
                )

        print(f"\nParámetros de simulación:")
        print(f"  - Duración total de simulación: {total_hours} horas")
        print(f"  - Tiempo por ronda: {minutes_per_round} minutos")
        print(f"  - Número total de rondas: {total_rounds}")
        if max_rounds:
            print(f"  - Límite máximo de rondas: {max_rounds}")
        print(f"  - Cantidad de Agents: {len(self.config.get('agent_configs', []))}")

        print("\nInicializando modelo LLM...")
        model = self._create_model()

        print("Cargando Agent Profile...")
        profile_path = self._get_profile_path()
        if not os.path.exists(profile_path):
            print(f"Error: Archivo de Profile no existe: {profile_path}")
            return

        self.agent_graph = await generate_reddit_agent_graph(
            profile_path=profile_path,
            model=model,
            available_actions=self.AVAILABLE_ACTIONS,
        )

        db_path = self._get_db_path()
        if os.path.exists(db_path):
            os.remove(db_path)
            print(f"Base de datos antigua eliminada: {db_path}")

        print("Creando entorno OASIS...")
        self.env = oasis.make(
            agent_graph=self.agent_graph,
            platform=oasis.DefaultPlatformType.REDDIT,
            database_path=db_path,
            semaphore=30,  # Limitar máximo de solicitudes LLM concurrentes para evitar sobrecarga de API
        )

        await self.env.reset()
        print("Entorno inicializado\n")

        # Inicializar procesador IPC
        self.ipc_handler = IPCHandler(self.simulation_dir, self.env, self.agent_graph)
        self.ipc_handler.update_status("running")

        # Ejecutar eventos iniciales
        event_config = self.config.get("event_config", {})
        initial_posts = event_config.get("initial_posts", [])

        if initial_posts:
            print(
                f"Ejecutando eventos iniciales ({len(initial_posts)} posts iniciales)..."
            )
            initial_actions = {}
            for post in initial_posts:
                agent_id = post.get("poster_agent_id", 0)
                content = post.get("content", "")
                try:
                    agent = self.env.agent_graph.get_agent(agent_id)
                    if agent in initial_actions:
                        if not isinstance(initial_actions[agent], list):
                            initial_actions[agent] = [initial_actions[agent]]
                        initial_actions[agent].append(
                            ManualAction(
                                action_type=ActionType.CREATE_POST,
                                action_args={"content": content},
                            )
                        )
                    else:
                        initial_actions[agent] = ManualAction(
                            action_type=ActionType.CREATE_POST,
                            action_args={"content": content},
                        )
                except Exception as e:
                    print(
                        f"  Advertencia: No se pudo crear post inicial para Agent {agent_id}: {e}"
                    )

            if initial_actions:
                await self.env.step(initial_actions)
                print(f"  Se publicaron {len(initial_actions)} posts iniciales")

        # Bucle principal de simulación
        print("\nIniciando bucle de simulación...")
        start_time = datetime.now()

        for round_num in range(total_rounds):
            simulated_minutes = round_num * minutes_per_round
            simulated_hour = (simulated_minutes // 60) % 24
            simulated_day = simulated_minutes // (60 * 24) + 1

            active_agents = self._get_active_agents_for_round(
                self.env, simulated_hour, round_num
            )

            if not active_agents:
                continue

            actions = {agent: LLMAction() for _, agent in active_agents}

            await self.env.step(actions)

            if (round_num + 1) % 10 == 0 or round_num == 0:
                elapsed = (datetime.now() - start_time).total_seconds()
                progress = (round_num + 1) / total_rounds * 100
                print(
                    f"  [Day {simulated_day}, {simulated_hour:02d}:00] "
                    f"Round {round_num + 1}/{total_rounds} ({progress:.1f}%) "
                    f"- {len(active_agents)} agents active "
                    f"- elapsed: {elapsed:.1f}s"
                )

        total_elapsed = (datetime.now() - start_time).total_seconds()
        print(f"\n¡Bucle de simulación completado!")
        print(f"  - Tiempo total: {total_elapsed:.1f} segundos")
        print(f"  - Base de datos: {db_path}")

        # ¿Ingresar al modo de espera de comandos?
        if self.wait_for_commands:
            print("\n" + "=" * 60)
            print(
                "Ingresando al modo de espera de comandos - Entorno mantiene ejecución"
            )
            print("Comandos soportados: interview, batch_interview, close_env")
            print("=" * 60)

            self.ipc_handler.update_status("alive")

            # Bucle de espera de comandos (usar _shutdown_event global)
            try:
                while not _shutdown_event.is_set():
                    should_continue = await self.ipc_handler.process_commands()
                    if not should_continue:
                        break
                    try:
                        await asyncio.wait_for(_shutdown_event.wait(), timeout=0.5)
                        break  # Señal de salida recibida
                    except asyncio.TimeoutError:
                        pass
            except KeyboardInterrupt:
                print("\nSeñal de interrupción recibida")
            except asyncio.CancelledError:
                print("\nTarea cancelada")
            except Exception as e:
                print(f"\nError al procesar comandos: {e}")

            print("\nCerrando entorno...")

        # Cerrar entorno
        self.ipc_handler.update_status("stopped")
        await self.env.close()

        print("Entorno cerrado")
        print("=" * 60)


async def main():
    parser = argparse.ArgumentParser(description="Simulación OASIS Reddit")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Ruta del archivo de configuración (simulation_config.json)",
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=None,
        help="Número máximo de rondas de simulación (opcional, para truncar simulación muy larga)",
    )
    parser.add_argument(
        "--no-wait",
        action="store_true",
        default=False,
        help="Cerrar entorno inmediatamente después de la simulación, no entrar en modo de espera de comandos",
    )

    args = parser.parse_args()

    # Crear evento de shutdown al inicio de la función main
    global _shutdown_event
    _shutdown_event = asyncio.Event()

    if not os.path.exists(args.config):
        print(f"Error: Archivo de configuración no existe: {args.config}")
        sys.exit(1)

    # Inicializar configuración de logs (usar nombre de archivo fijo, limpiar logs antiguos)
    simulation_dir = os.path.dirname(args.config) or "."
    setup_oasis_logging(os.path.join(simulation_dir, "log"))

    runner = RedditSimulationRunner(
        config_path=args.config, wait_for_commands=not args.no_wait
    )
    await runner.run(max_rounds=args.max_rounds)


def setup_signal_handlers():
    """
    Configurar manejador de señales, asegurar salida correcta al recibir SIGTERM/SIGINT
    Dar al programa oportunidad de limpiar recursos correctamente (cerrar base de datos, entorno, etc.)
    """

    def signal_handler(signum, frame):
        global _cleanup_done
        sig_name = "SIGTERM" if signum == signal.SIGTERM else "SIGINT"
        print(f"\nRecibida señal {sig_name}, saliendo...")
        if not _cleanup_done:
            _cleanup_done = True
            if _shutdown_event:
                _shutdown_event.set()
        else:
            # Solo forzar salida si se recibe señal repetida
            print("Salida forzada...")
            sys.exit(1)

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)


if __name__ == "__main__":
    setup_signal_handlers()
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nPrograma interrumpido")
    except SystemExit:
        pass
    finally:
        print("Proceso de simulación exited")
