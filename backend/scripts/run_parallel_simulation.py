"""
OASIS Parallel dual-platform simulation preset script
Runs Twitter and Reddit simulation simultaneously using the same config file

Features:
- Dual-platform (Twitter + Reddit) parallel simulation
- After completing simulation, doesn't exit immediately, enters command-waiting mode
- Supports receiving Interview commands via IPC
- Supports single Agent interview and batch interview
- Supports remote shutdown command

Usage:
    python run_parallel_simulation.py --config simulation_config.json
    python run_parallel_simulation.py --config simulation_config.json --no-wait  # Exit immediately after completion
    python run_parallel_simulation.py --config simulation_config.json --twitter-only
    python run_parallel_simulation.py --config simulation_config.json --reddit-only

Log structure:
    sim_xxx/
    ├── twitter/
    │   └── actions.jsonl    # Twitter platform action log
    ├── reddit/
    │   └── actions.jsonl    # Reddit platform action log
    ├── simulation.log       # Main simulation process log
    └── run_state.json       # Run state (for API queries)
"""

# ============================================================
# Fix Windows encoding issue: Set UTF-8 encoding BEFORE all imports
# This fixes the issue of OASIS third-party library reading files without specifying encoding
# ============================================================
import sys
import os

if sys.platform == "win32":
    # Set Python default I/O encoding to UTF-8
    # This affects all open() calls without specified encoding
    os.environ.setdefault("PYTHONUTF8", "1")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")

    # Reconfigure stdout to UTF-8 (fix console Chinese character display)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    # Force set default encoding (affects open() function default encoding)
    # Note: This needs to be set at Python startup, runtime setting may not take effect
    # So we also need to monkey-patch the built-in open function
    import builtins

    _original_open = builtins.open

    def _utf8_open(
        file,
        mode="r",
        buffering=-1,
        encoding=None,
        errors=None,
        newline=None,
        closefd=True,
        opener=None,
    ):
        """
        Wrapper for open() function, default to UTF-8 encoding for text mode
        This fixes the issue of third-party libraries (like OASIS) reading files without specifying encoding
        """
        # Only set default encoding for text mode (non-binary) without specified encoding
        if encoding is None and "b" not in mode:
            encoding = "utf-8"
        return _original_open(
            file, mode, buffering, encoding, errors, newline, closefd, opener
        )

    builtins.open = _utf8_open

import argparse
import asyncio
import json
import logging
import multiprocessing
import random
import signal
import sqlite3
import warnings
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple


# Global variable: for signal handling
_shutdown_event = None
_cleanup_done = False

# Add backend directory to path
# Script is fixed at backend/scripts/ directory
_scripts_dir = os.path.dirname(os.path.abspath(__file__))
_backend_dir = os.path.abspath(os.path.join(_scripts_dir, ".."))
_project_root = os.path.abspath(os.path.join(_backend_dir, ".."))
sys.path.insert(0, _scripts_dir)
sys.path.insert(0, _backend_dir)

# Load project root .env file (contains LLM_API_KEY etc. config)
from dotenv import load_dotenv

_env_file = os.path.join(_project_root, ".env")
if os.path.exists(_env_file):
    load_dotenv(_env_file)
    print(f"Se ha cargado la configuración del entorno: {_env_file}")
else:
    # Intentar cargar backend/.env
    _backend_env = os.path.join(_backend_dir, ".env")
    if os.path.exists(_backend_env):
        load_dotenv(_backend_env)
        print(f"Se ha cargado la configuración del entorno: {_backend_env}")


class MaxTokensWarningFilter(logging.Filter):
    """Filtrar las advertencias sobre max_tokens (intencionadamente no establecemos max_tokens, dejando que el modelo decida)"""

    def filter(self, record):
        # Filtrar los logs que contengan advertencias de max_tokens
        if (
            "max_tokens" in record.getMessage()
            and "Invalid or missing" in record.getMessage()
        ):
            return False
        return True


# Agregar filtro al cargar el modulo, asegurar que funcione antes de ejecutar codigo camel
logging.getLogger().addFilter(MaxTokensWarningFilter())


def disable_oasis_logging():
    """
    Desactivar la salida detallada de logs de OASIS
    Los logs de OASIS son muy verbosos (registran cada observacion y accion del agent), usamos nuestro propio action_logger
    """
    # Desactivar todos los loggers de OASIS
    oasis_loggers = [
        "social.agent",
        "social.twitter",
        "social.rec",
        "oasis.env",
        "table",
    ]

    for logger_name in oasis_loggers:
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.CRITICAL)  # Solo registrar errores criticos
        logger.handlers.clear()
        logger.propagate = False


def init_logging_for_simulation(simulation_dir: str):
    """
    Inicializar la configuración de logs para la simulación

    Args:
        simulation_dir: Directorio de simulación
    """
    # Desactivar logs detallados de OASIS
    disable_oasis_logging()

    # Limpiar directorio de log antiguo (si existe)
    old_log_dir = os.path.join(simulation_dir, "log")
    if os.path.exists(old_log_dir):
        import shutil

        shutil.rmtree(old_log_dir, ignore_errors=True)


from action_logger import SimulationLogManager, PlatformActionLogger

try:
    from camel.models import ModelFactory
    from camel.types import ModelPlatformType
    import oasis
    from oasis import (
        ActionType,
        LLMAction,
        ManualAction,
        generate_twitter_agent_graph,
        generate_reddit_agent_graph,
    )
except ImportError as e:
    print(f"Error: Faltan dependencias {e}")
    print("Por favor instale primero: pip install oasis-ai camel-ai")
    sys.exit(1)


# Twitter acciones disponibles (no incluye INTERVIEW, INTERVIEW solo se puede activar manualmente a traves de ManualAction)
TWITTER_ACTIONS = [
    ActionType.CREATE_POST,
    ActionType.LIKE_POST,
    ActionType.REPOST,
    ActionType.FOLLOW,
    ActionType.DO_NOTHING,
    ActionType.QUOTE_POST,
]

# Reddit acciones disponibles (no incluye INTERVIEW, INTERVIEW solo se puede activar manualmente a traves de ManualAction)
REDDIT_ACTIONS = [
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


# Constantes relacionadas con IPC
IPC_COMMANDS_DIR = "ipc_commands"
IPC_RESPONSES_DIR = "ipc_responses"
ENV_STATUS_FILE = "env_status.json"


class CommandType:
    """Constantes de tipo de comando"""

    INTERVIEW = "interview"
    BATCH_INTERVIEW = "batch_interview"
    CLOSE_ENV = "close_env"


class ParallelIPCHandler:
    """
    Procesador de comandos IPC de doble plataforma

    Gestiona entornos de dos plataformas, procesa comandos Interview
    """

    def __init__(
        self,
        simulation_dir: str,
        twitter_env=None,
        twitter_agent_graph=None,
        reddit_env=None,
        reddit_agent_graph=None,
    ):
        self.simulation_dir = simulation_dir
        self.twitter_env = twitter_env
        self.twitter_agent_graph = twitter_agent_graph
        self.reddit_env = reddit_env
        self.reddit_agent_graph = reddit_agent_graph

        self.commands_dir = os.path.join(simulation_dir, IPC_COMMANDS_DIR)
        self.responses_dir = os.path.join(simulation_dir, IPC_RESPONSES_DIR)
        self.status_file = os.path.join(simulation_dir, ENV_STATUS_FILE)

        # Asegurar que existan los directorios
        os.makedirs(self.commands_dir, exist_ok=True)
        os.makedirs(self.responses_dir, exist_ok=True)

    def update_status(self, status: str):
        """Actualizar estado del entorno"""
        with open(self.status_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "status": status,
                    "twitter_available": self.twitter_env is not None,
                    "reddit_available": self.reddit_env is not None,
                    "timestamp": datetime.now().isoformat(),
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

    def poll_command(self) -> Optional[Dict[str, Any]]:
        """Sonear para obtener comandos pendientes"""
        if not os.path.exists(self.commands_dir):
            return None

        # Obtener archivos de comando (ordenados por tiempo)
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

    def _get_env_and_graph(self, platform: str):
        """
        Obtener entorno y agent_graph de la plataforma especificada

        Args:
            platform: Nombre de plataforma ("twitter" o "reddit")

        Returns:
            (env, agent_graph, platform_name) o (None, None, None)
        """
        if platform == "twitter" and self.twitter_env:
            return self.twitter_env, self.twitter_agent_graph, "twitter"
        elif platform == "reddit" and self.reddit_env:
            return self.reddit_env, self.reddit_agent_graph, "reddit"
        else:
            return None, None, None

    async def _interview_single_platform(
        self, agent_id: int, prompt: str, platform: str
    ) -> Dict[str, Any]:
        """
        Ejecutar Interview en una sola plataforma

        Returns:
            Diccionario con resultados, o diccionario con error
        """
        env, agent_graph, actual_platform = self._get_env_and_graph(platform)

        if not env or not agent_graph:
            return {
                "platform": platform,
                "error": f"{platform}plataforma no disponible",
            }

        try:
            agent = agent_graph.get_agent(agent_id)
            interview_action = ManualAction(
                action_type=ActionType.INTERVIEW, action_args={"prompt": prompt}
            )
            actions = {agent: interview_action}
            await env.step(actions)

            result = self._get_interview_result(agent_id, actual_platform)
            result["platform"] = actual_platform
            return result

        except Exception as e:
            return {"platform": platform, "error": str(e)}

    async def handle_interview(
        self, command_id: str, agent_id: int, prompt: str, platform: str = None
    ) -> bool:
        """
        Procesar comando de entrevista de un solo Agent

        Args:
            command_id: ID de comando
            agent_id: Agent ID
            prompt: Pregunta de entrevista
            platform: Plataforma especifica (opcional)
                - "twitter": Solo entrevistar plataforma Twitter
                - "reddit": Solo entrevistar plataforma Reddit
                - None/sin especificar: Entrevistar ambas plataformas, devolver resultado integrado

        Returns:
            True significa exito, False significa fracaso
        """
        # Si se especifica plataforma, solo entrevistar esa plataforma
        if platform in ("twitter", "reddit"):
            result = await self._interview_single_platform(agent_id, prompt, platform)

            if "error" in result:
                self.send_response(command_id, "failed", error=result["error"])
                print(
                    f"  Interview fallo: agent_id={agent_id}, platform={platform}, error={result['error']}"
                )
                return False
            else:
                self.send_response(command_id, "completed", result=result)
                print(
                    f"  Interview completado: agent_id={agent_id}, platform={platform}"
                )
                return True

        # Sin especificar plataforma: entrevistar ambas plataformas
        if not self.twitter_env and not self.reddit_env:
            self.send_response(
                command_id, "failed", error="no hay entornos de simulación disponibles"
            )
            return False

        results = {"agent_id": agent_id, "prompt": prompt, "platforms": {}}
        success_count = 0

        # Entrevistar ambas plataformas en paralelo
        tasks = []
        platforms_to_interview = []

        if self.twitter_env:
            tasks.append(self._interview_single_platform(agent_id, prompt, "twitter"))
            platforms_to_interview.append("twitter")

        if self.reddit_env:
            tasks.append(self._interview_single_platform(agent_id, prompt, "reddit"))
            platforms_to_interview.append("reddit")

        # Ejecutar en paralelo
        platform_results = await asyncio.gather(*tasks)

        for platform_name, platform_result in zip(
            platforms_to_interview, platform_results
        ):
            results["platforms"][platform_name] = platform_result
            if "error" not in platform_result:
                success_count += 1

        if success_count > 0:
            self.send_response(command_id, "completed", result=results)
            print(
                f"  Interview completado: agent_id={agent_id}, plataformas exitosas={success_count}/{len(platforms_to_interview)}"
            )
            return True
        else:
            errors = [
                f"{p}: {r.get('error', 'Error desconocido')}"
                for p, r in results["platforms"].items()
            ]
            self.send_response(command_id, "failed", error="; ".join(errors))
            print(
                f"  Interview fallo: agent_id={agent_id}, todas las plataformas fallaron"
            )
            return False

    async def handle_batch_interview(
        self, command_id: str, interviews: List[Dict], platform: str = None
    ) -> bool:
        """
        Procesar comando de entrevista por lotes

        Args:
            command_id: ID de comando
            interviews: [{"agent_id": int, "prompt": str, "platform": str(optional)}, ...]
            platform: Plataforma por defecto (puede ser sobreescrita por cada item de interview)
                - "twitter": Solo entrevistar plataforma Twitter
                - "reddit": Solo entrevistar plataforma Reddit
                - None/sin especificar: Cada Agent entrevista ambas plataformas
        """
        # Agrupar por plataforma
        twitter_interviews = []
        reddit_interviews = []
        both_platforms_interviews = []  # Necesita entrevistar ambas plataformas

        for interview in interviews:
            item_platform = interview.get("platform", platform)
            if item_platform == "twitter":
                twitter_interviews.append(interview)
            elif item_platform == "reddit":
                reddit_interviews.append(interview)
            else:
                # Sin especificar: entrevistar ambas plataformas
                both_platforms_interviews.append(interview)

        # Dividir both_platforms_interviews en dos plataformas
        if both_platforms_interviews:
            if self.twitter_env:
                twitter_interviews.extend(both_platforms_interviews)
            if self.reddit_env:
                reddit_interviews.extend(both_platforms_interviews)

        results = {}

        # Procesar entrevistas de plataforma Twitter
        if twitter_interviews and self.twitter_env:
            try:
                twitter_actions = {}
                for interview in twitter_interviews:
                    agent_id = interview.get("agent_id")
                    prompt = interview.get("prompt", "")
                    try:
                        agent = self.twitter_agent_graph.get_agent(agent_id)
                        twitter_actions[agent] = ManualAction(
                            action_type=ActionType.INTERVIEW,
                            action_args={"prompt": prompt},
                        )
                    except Exception as e:
                        print(
                            f"  Advertencia: No se puede obtener Twitter Agent {agent_id}: {e}"
                        )

                if twitter_actions:
                    await self.twitter_env.step(twitter_actions)

                    for interview in twitter_interviews:
                        agent_id = interview.get("agent_id")
                        result = self._get_interview_result(agent_id, "twitter")
                        result["platform"] = "twitter"
                        results[f"twitter_{agent_id}"] = result
            except Exception as e:
                print(f"  Interview por lotes Twitter fallo: {e}")

        # Procesar entrevistas de plataforma Reddit
        if reddit_interviews and self.reddit_env:
            try:
                reddit_actions = {}
                for interview in reddit_interviews:
                    agent_id = interview.get("agent_id")
                    prompt = interview.get("prompt", "")
                    try:
                        agent = self.reddit_agent_graph.get_agent(agent_id)
                        reddit_actions[agent] = ManualAction(
                            action_type=ActionType.INTERVIEW,
                            action_args={"prompt": prompt},
                        )
                    except Exception as e:
                        print(
                            f"  Advertencia: No se puede obtener Reddit Agent {agent_id}: {e}"
                        )

                if reddit_actions:
                    await self.reddit_env.step(reddit_actions)

                    for interview in reddit_interviews:
                        agent_id = interview.get("agent_id")
                        result = self._get_interview_result(agent_id, "reddit")
                        result["platform"] = "reddit"
                        results[f"reddit_{agent_id}"] = result
            except Exception as e:
                print(f"  Interview por lotes Reddit fallo: {e}")

        if results:
            self.send_response(
                command_id,
                "completed",
                result={"interviews_count": len(results), "results": results},
            )
            print(f"  Interview por lotes completado: {len(results)} Agents")
            return True
        else:
            self.send_response(
                command_id, "failed", error="No hay entrevistas exitosas"
            )
            return False

    def _get_interview_result(self, agent_id: int, platform: str) -> Dict[str, Any]:
        """Obtener la ultima entrevista de la base de datos"""
        db_path = os.path.join(self.simulation_dir, f"{platform}_simulation.db")

        result = {"agent_id": agent_id, "response": None, "timestamp": None}

        if not os.path.exists(db_path):
            return result

        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            # Consultar el ultimo registro de Interview
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
            print(f"  Error al leer el resultado de la entrevista: {e}")

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

        print(f"\nComando IPC recibido: {command_type}, id={command_id}")

        if command_type == CommandType.INTERVIEW:
            await self.handle_interview(
                command_id,
                args.get("agent_id", 0),
                args.get("prompt", ""),
                args.get("platform"),
            )
            return True

        elif command_type == CommandType.BATCH_INTERVIEW:
            await self.handle_batch_interview(
                command_id, args.get("interviews", []), args.get("platform")
            )
            return True

        elif command_type == CommandType.CLOSE_ENV:
            print("Recibido comando de cierre del entorno")
            self.send_response(
                command_id,
                "completed",
                result={"message": "El entorno se cerrara pronto"},
            )
            return False

        else:
            self.send_response(
                command_id,
                "failed",
                error=f"Comando de tipo desconocido: {command_type}",
            )
            return True


def load_config(config_path: str) -> Dict[str, Any]:
    """Cargar archivo de configuración"""
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


# Tipos de accion no core a filtrar (estas acciones tienen bajo valor para analisis)
FILTERED_ACTIONS = {"refresh", "sign_up"}

# Tabla de mapeo de tipos de accion (nombre en DB -> nombre estandar)
ACTION_TYPE_MAP = {
    "create_post": "CREATE_POST",
    "like_post": "LIKE_POST",
    "dislike_post": "DISLIKE_POST",
    "repost": "REPOST",
    "quote_post": "QUOTE_POST",
    "follow": "FOLLOW",
    "mute": "MUTE",
    "create_comment": "CREATE_COMMENT",
    "like_comment": "LIKE_COMMENT",
    "dislike_comment": "DISLIKE_COMMENT",
    "search_posts": "SEARCH_POSTS",
    "search_user": "SEARCH_USER",
    "trend": "TREND",
    "do_nothing": "DO_NOTHING",
    "interview": "INTERVIEW",
}


def get_agent_names_from_config(config: Dict[str, Any]) -> Dict[int, str]:
    """
    Obtener mapeo de agent_id a nombre de entidad

    Asi podemos mostrar nombres reales de entidades en actions.jsonl en lugar de codigos como "Agent_0"

    Args:
        config: Contenido de simulation_config.json

    Returns:
        Diccionario de mapeo agent_id -> entity_name
    """
    agent_names = {}
    agent_configs = config.get("agent_configs", [])

    for agent_config in agent_configs:
        agent_id = agent_config.get("agent_id")
        entity_name = agent_config.get("entity_name", f"Agent_{agent_id}")
        if agent_id is not None:
            agent_names[agent_id] = entity_name

    return agent_names


def fetch_new_actions_from_db(
    db_path: str, last_rowid: int, agent_names: Dict[int, str]
) -> Tuple[List[Dict[str, Any]], int]:
    """
    Obtener nuevas acciones de la base de datos y completar informacion de contexto

    Args:
        db_path: Ruta del archivo de base de datos
        last_rowid: Ultimo valor maximo de rowid leido (usar rowid en lugar de created_at porque los formatos difieren entre plataformas)
        agent_names: Mapeo agent_id -> agent_name

    Returns:
        (actions_list, new_last_rowid)
        - actions_list: Lista de acciones, cada elemento contiene agent_id, agent_name, action_type, action_args (con informacion de contexto)
        - new_last_rowid: Nuevo valor maximo de rowid
    """
    actions = []
    new_last_rowid = last_rowid

    if not os.path.exists(db_path):
        return actions, new_last_rowid

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Usar rowid para rastrear registros procesados (rowid es campo autoincremental de SQLite)
        # Esto evita problemas de formato de created_at (Twitter usa enteros, Reddit usa strings de fecha/hora)
        cursor.execute(
            """
            SELECT rowid, user_id, action, info
            FROM trace
            WHERE rowid > ?
            ORDER BY rowid ASC
        """,
            (last_rowid,),
        )

        for rowid, user_id, action, info_json in cursor.fetchall():
            # Actualizar max rowid
            new_last_rowid = rowid

            # Filtrar acciones no core
            if action in FILTERED_ACTIONS:
                continue

            # Parsear parametros de accion
            try:
                action_args = json.loads(info_json) if info_json else {}
            except json.JSONDecodeError:
                action_args = {}

            # Simplificar action_args, solo mantener campos clave (mantener contenido completo, no truncar)
            simplified_args = {}
            if "content" in action_args:
                simplified_args["content"] = action_args["content"]
            if "post_id" in action_args:
                simplified_args["post_id"] = action_args["post_id"]
            if "comment_id" in action_args:
                simplified_args["comment_id"] = action_args["comment_id"]
            if "quoted_id" in action_args:
                simplified_args["quoted_id"] = action_args["quoted_id"]
            if "new_post_id" in action_args:
                simplified_args["new_post_id"] = action_args["new_post_id"]
            if "follow_id" in action_args:
                simplified_args["follow_id"] = action_args["follow_id"]
            if "query" in action_args:
                simplified_args["query"] = action_args["query"]
            if "like_id" in action_args:
                simplified_args["like_id"] = action_args["like_id"]
            if "dislike_id" in action_args:
                simplified_args["dislike_id"] = action_args["dislike_id"]

            # Convertir nombre de tipo de accion
            action_type = ACTION_TYPE_MAP.get(action, action.upper())

            # Completar informacion de contexto (contenido del post, nombre de usuario, etc.)
            _enrich_action_context(cursor, action_type, simplified_args, agent_names)

            actions.append(
                {
                    "agent_id": user_id,
                    "agent_name": agent_names.get(user_id, f"Agent_{user_id}"),
                    "action_type": action_type,
                    "action_args": simplified_args,
                }
            )

        conn.close()
    except Exception as e:
        print(f"Error al leer acciones de la base de datos: {e}")

    return actions, new_last_rowid


def _enrich_action_context(
    cursor, action_type: str, action_args: Dict[str, Any], agent_names: Dict[int, str]
) -> None:
    """
    Enriquecer contexto de acciones (contenido del post, nombre de usuario, etc.)

    Args:
        cursor: Cursor de base de datos
        action_type: Tipo de accion
        action_args: Parametros de accion (sera modificado)
        agent_names: Mapeo agent_id -> agent_name
    """
    try:
        # Like/dislike post: completar contenido del post y autor
        if action_type in ("LIKE_POST", "DISLIKE_POST"):
            post_id = action_args.get("post_id")
            if post_id:
                post_info = _get_post_info(cursor, post_id, agent_names)
                if post_info:
                    action_args["post_content"] = post_info.get("content", "")
                    action_args["post_author_name"] = post_info.get("author_name", "")

        # Repost: completar contenido del post original y autor
        elif action_type == "REPOST":
            new_post_id = action_args.get("new_post_id")
            if new_post_id:
                # El original_post_id del repost apunta al post original
                cursor.execute(
                    """
                    SELECT original_post_id FROM post WHERE post_id = ?
                """,
                    (new_post_id,),
                )
                row = cursor.fetchone()
                if row and row[0]:
                    original_post_id = row[0]
                    original_info = _get_post_info(
                        cursor, original_post_id, agent_names
                    )
                    if original_info:
                        action_args["original_content"] = original_info.get(
                            "content", ""
                        )
                        action_args["original_author_name"] = original_info.get(
                            "author_name", ""
                        )

        # Quote post: completar contenido, autor y comentario citado
        elif action_type == "QUOTE_POST":
            quoted_id = action_args.get("quoted_id")
            new_post_id = action_args.get("new_post_id")

            if quoted_id:
                original_info = _get_post_info(cursor, quoted_id, agent_names)
                if original_info:
                    action_args["original_content"] = original_info.get("content", "")
                    action_args["original_author_name"] = original_info.get(
                        "author_name", ""
                    )

            # Obtener contenido del comentario citado (quote_content)
            if new_post_id:
                cursor.execute(
                    """
                    SELECT quote_content FROM post WHERE post_id = ?
                """,
                    (new_post_id,),
                )
                row = cursor.fetchone()
                if row and row[0]:
                    action_args["quote_content"] = row[0]

        # Follow user: completar nombre del usuario seguido
        elif action_type == "FOLLOW":
            follow_id = action_args.get("follow_id")
            if follow_id:
                # Obtener followee_id de la tabla follow
                cursor.execute(
                    """
                    SELECT followee_id FROM follow WHERE follow_id = ?
                """,
                    (follow_id,),
                )
                row = cursor.fetchone()
                if row:
                    followee_id = row[0]
                    target_name = _get_user_name(cursor, followee_id, agent_names)
                    if target_name:
                        action_args["target_user_name"] = target_name

        # Mute user: completar nombre del usuario muteado
        elif action_type == "MUTE":
            # Obtener user_id o target_id de action_args
            target_id = action_args.get("user_id") or action_args.get("target_id")
            if target_id:
                target_name = _get_user_name(cursor, target_id, agent_names)
                if target_name:
                    action_args["target_user_name"] = target_name

        # Like/dislike comment: completar contenido y autor del comentario
        elif action_type in ("LIKE_COMMENT", "DISLIKE_COMMENT"):
            comment_id = action_args.get("comment_id")
            if comment_id:
                comment_info = _get_comment_info(cursor, comment_id, agent_names)
                if comment_info:
                    action_args["comment_content"] = comment_info.get("content", "")
                    action_args["comment_author_name"] = comment_info.get(
                        "author_name", ""
                    )

        # Crear comentario: completar informacion del post comentado
        elif action_type == "CREATE_COMMENT":
            post_id = action_args.get("post_id")
            if post_id:
                post_info = _get_post_info(cursor, post_id, agent_names)
                if post_info:
                    action_args["post_content"] = post_info.get("content", "")
                    action_args["post_author_name"] = post_info.get("author_name", "")

    except Exception as e:
        # Fallo al completar contexto no afecta el flujo principal
        print(f"Error al completar contexto de accion: {e}")


def _get_post_info(
    cursor, post_id: int, agent_names: Dict[int, str]
) -> Optional[Dict[str, str]]:
    """
    Obtener información del post

    Args:
        cursor: Cursor de base de datos
        post_id: ID del post
        agent_names: Mapeo agent_id -> agent_name

    Returns:
        Diccionario con content y author_name, o None
    """
    try:
        cursor.execute(
            """
            SELECT p.content, p.user_id, u.agent_id
            FROM post p
            LEFT JOIN user u ON p.user_id = u.user_id
            WHERE p.post_id = ?
        """,
            (post_id,),
        )
        row = cursor.fetchone()
        if row:
            content = row[0] or ""
            user_id = row[1]
            agent_id = row[2]

            # Preferir usar nombre de agent_names
            author_name = ""
            if agent_id is not None and agent_id in agent_names:
                author_name = agent_names[agent_id]
            elif user_id:
                # Obtener nombre de la tabla user
                cursor.execute(
                    "SELECT name, user_name FROM user WHERE user_id = ?", (user_id,)
                )
                user_row = cursor.fetchone()
                if user_row:
                    author_name = user_row[0] or user_row[1] or ""

            return {"content": content, "author_name": author_name}
    except Exception:
        pass
    return None


def _get_user_name(cursor, user_id: int, agent_names: Dict[int, str]) -> Optional[str]:
    """
    Obtener nombre de usuario

    Args:
        cursor: Cursor de base de datos
        user_id: ID de usuario
        agent_names: Mapeo agent_id -> agent_name

    Returns:
        Nombre de usuario, o None
    """
    try:
        cursor.execute(
            """
            SELECT agent_id, name, user_name FROM user WHERE user_id = ?
        """,
            (user_id,),
        )
        row = cursor.fetchone()
        if row:
            agent_id = row[0]
            name = row[1]
            user_name = row[2]

            # Preferir usar nombre de agent_names
            if agent_id is not None and agent_id in agent_names:
                return agent_names[agent_id]
            return name or user_name or ""
    except Exception:
        pass
    return None


def _get_comment_info(
    cursor, comment_id: int, agent_names: Dict[int, str]
) -> Optional[Dict[str, str]]:
    """
    Obtener información del comentario

    Args:
        cursor: Cursor de base de datos
        comment_id: ID del comentario
        agent_names: Mapeo agent_id -> agent_name

    Returns:
        Diccionario con content y author_name, o None
    """
    try:
        cursor.execute(
            """
            SELECT c.content, c.user_id, u.agent_id
            FROM comment c
            LEFT JOIN user u ON c.user_id = u.user_id
            WHERE c.comment_id = ?
        """,
            (comment_id,),
        )
        row = cursor.fetchone()
        if row:
            content = row[0] or ""
            user_id = row[1]
            agent_id = row[2]

            # Preferir usar nombre de agent_names
            author_name = ""
            if agent_id is not None and agent_id in agent_names:
                author_name = agent_names[agent_id]
            elif user_id:
                # Obtener nombre de la tabla user
                cursor.execute(
                    "SELECT name, user_name FROM user WHERE user_id = ?", (user_id,)
                )
                user_row = cursor.fetchone()
                if user_row:
                    author_name = user_row[0] or user_row[1] or ""

            return {"content": content, "author_name": author_name}
    except Exception:
        pass
    return None


def create_model(config: Dict[str, Any], use_boost: bool = False):
    """
    Crear modelo LLM

    Soporta configuracion dual de LLM para acelerar simulacion paralela:
    - Configuracion general: LLM_API_KEY, LLM_BASE_URL, LLM_MODEL_NAME
    - Configuracion acelerada (opcional): LLM_BOOST_API_KEY, LLM_BOOST_BASE_URL, LLM_BOOST_MODEL_NAME

    Si se configura LLM acelerada, la simulacion paralela puede usar diferentes proveedores de API para diferentes plataformas, aumentando la concurrencia.

    Args:
        config: Diccionario de configuracion de simulacion
        use_boost: Si usar configuracion de LLM acelerada (si esta disponible)
    """
    # Verificar si hay configuracion acelerada
    boost_api_key = os.environ.get("LLM_BOOST_API_KEY", "")
    boost_base_url = os.environ.get("LLM_BOOST_BASE_URL", "")
    boost_model = os.environ.get("LLM_BOOST_MODEL_NAME", "")
    has_boost_config = bool(boost_api_key)

    # Seleccionar cual LLM usar segun parametros y configuracion
    if use_boost and has_boost_config:
        # Usar configuracion acelerada
        llm_api_key = boost_api_key
        llm_base_url = boost_base_url
        llm_model = boost_model or os.environ.get("LLM_MODEL_NAME", "")
        config_label = "[LLM acelerado]"
    else:
        # Usar configuracion general
        llm_api_key = os.environ.get("LLM_API_KEY", "")
        llm_base_url = os.environ.get("LLM_BASE_URL", "")
        llm_model = os.environ.get("LLM_MODEL_NAME", "")
        config_label = "[LLM general]"

    # Si no hay nombre de modelo en .env, usar config como respaldo
    if not llm_model:
        llm_model = config.get("llm_model", "gpt-4o-mini")

    # Establecer variables de entorno requeridas por camel-ai
    if llm_api_key:
        os.environ["OPENAI_API_KEY"] = llm_api_key

    if not os.environ.get("OPENAI_API_KEY"):
        raise ValueError(
            "Falta configuracion de API Key, por favor establezca LLM_API_KEY en el archivo .env del directorio raiz"
        )

    if llm_base_url:
        os.environ["OPENAI_API_BASE_URL"] = llm_base_url

    print(
        f"{config_label} model={llm_model}, base_url={llm_base_url[:40] if llm_base_url else 'por defecto'}..."
    )

    return ModelFactory.create(
        model_platform=ModelPlatformType.OPENAI,
        model_type=llm_model,
    )


def get_active_agents_for_round(
    env, config: Dict[str, Any], current_hour: int, round_num: int
) -> List:
    """Decidir qué Agent activar"""
    time_config = config.get("time_config", {})
    agent_configs = config.get("agent_configs", [])

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


class PlatformSimulation:
    """Contenedor de resultados de simulacion de plataforma"""

    def __init__(self):
        self.env = None
        self.agent_graph = None
        self.total_actions = 0


async def run_twitter_simulation(
    config: Dict[str, Any],
    simulation_dir: str,
    action_logger: Optional[PlatformActionLogger] = None,
    main_logger: Optional[SimulationLogManager] = None,
    max_rounds: Optional[int] = None,
) -> PlatformSimulation:
    """Ejecutar simulacion Twitter

    Args:
        config: Configuracion de simulacion
        simulation_dir: Directorio de simulacion
        action_logger: Registrador de acciones
        main_logger: Gestor de logs principal
        max_rounds: Maximo de rondas de simulacion (opcional, para truncar simulaciones largas)

    Returns:
        PlatformSimulation: Objeto de resultados que contiene env y agent_graph
    """
    result = PlatformSimulation()

    def log_info(msg):
        if main_logger:
            main_logger.info(f"[Twitter] {msg}")
        print(f"[Twitter] {msg}")

    log_info("Inicializando...")

    # Twitter usa configuracion general de LLM
    model = create_model(config, use_boost=False)

    # OASIS Twitter usa formato CSV
    profile_path = os.path.join(simulation_dir, "twitter_profiles.csv")
    if not os.path.exists(profile_path):
        log_info(f"Error: El archivo de perfil no existe: {profile_path}")
        return result

    result.agent_graph = await generate_twitter_agent_graph(
        profile_path=profile_path,
        model=model,
        available_actions=TWITTER_ACTIONS,
    )

    # Obtener mapeo de nombres reales de Agent desde config (usar entity_name en lugar de Agent_X por defecto)
    agent_names = get_agent_names_from_config(config)
    # Si no hay un agent en config, usar nombre por defecto de OASIS
    for agent_id, agent in result.agent_graph.get_agents():
        if agent_id not in agent_names:
            agent_names[agent_id] = getattr(agent, "name", f"Agent_{agent_id}")

    db_path = os.path.join(simulation_dir, "twitter_simulation.db")
    if os.path.exists(db_path):
        os.remove(db_path)

    result.env = oasis.make(
        agent_graph=result.agent_graph,
        platform=oasis.DefaultPlatformType.TWITTER,
        database_path=db_path,
        semaphore=30,  # Limitar maximo de solicitudes LLM concurrentes para evitar sobrecarga de API
    )

    await result.env.reset()
    log_info("Entorno iniciado")

    if action_logger:
        action_logger.log_simulation_start(config)

    total_actions = 0
    last_rowid = 0  # Rastrear ultimo numero de fila procesada (usar rowid para evitar diferencias de formato created_at)

    # Ejecutar eventos iniciales
    event_config = config.get("event_config", {})
    initial_posts = event_config.get("initial_posts", [])

    # Registrar inicio de round 0 (fase de eventos iniciales)
    if action_logger:
        action_logger.log_round_start(0, 0)  # round 0, simulated_hour 0

    initial_action_count = 0
    if initial_posts:
        initial_actions = {}
        for post in initial_posts:
            agent_id = post.get("poster_agent_id", 0)
            content = post.get("content", "")
            try:
                agent = result.env.agent_graph.get_agent(agent_id)
                initial_actions[agent] = ManualAction(
                    action_type=ActionType.CREATE_POST, action_args={"content": content}
                )

                if action_logger:
                    action_logger.log_action(
                        round_num=0,
                        agent_id=agent_id,
                        agent_name=agent_names.get(agent_id, f"Agent_{agent_id}"),
                        action_type="CREATE_POST",
                        action_args={"content": content},
                    )
                    total_actions += 1
                    initial_action_count += 1
            except Exception:
                pass

        if initial_actions:
            await result.env.step(initial_actions)
            log_info(f"Publicados {len(initial_actions)} posts iniciales")

    # Registrar fin de round 0
    if action_logger:
        action_logger.log_round_end(0, initial_action_count)

    # Bucle principal de simulacion
    time_config = config.get("time_config", {})
    total_hours = time_config.get("total_simulation_hours", 72)
    minutes_per_round = time_config.get("minutes_per_round", 30)
    total_rounds = (total_hours * 60) // minutes_per_round

    # Si se especifica maximo de rondas, truncar
    if max_rounds is not None and max_rounds > 0:
        original_rounds = total_rounds
        total_rounds = min(total_rounds, max_rounds)
        if total_rounds < original_rounds:
            log_info(
                f"Rondas truncadas: {original_rounds} -> {total_rounds} (max_rounds={max_rounds})"
            )

    start_time = datetime.now()

    for round_num in range(total_rounds):
        # Verificar si se recibio senal de salida
        if _shutdown_event and _shutdown_event.is_set():
            if main_logger:
                main_logger.info(
                    f"Recibida senal de salida, deteniendo simulacion en ronda {round_num + 1}"
                )
            break

        simulated_minutes = round_num * minutes_per_round
        simulated_hour = (simulated_minutes // 60) % 24
        simulated_day = simulated_minutes // (60 * 24) + 1

        active_agents = get_active_agents_for_round(
            result.env, config, simulated_hour, round_num
        )

        # Registrar inicio de round independientemente de si hay agents activos
        if action_logger:
            action_logger.log_round_start(round_num + 1, simulated_hour)

        if not active_agents:
            # Registrar fin de round sin agents activos (actions_count=0)
            if action_logger:
                action_logger.log_round_end(round_num + 1, 0)
            continue

        actions = {agent: LLMAction() for _, agent in active_agents}
        await result.env.step(actions)

        # Obtener y registrar acciones realmente ejecutadas de la base de datos
        actual_actions, last_rowid = fetch_new_actions_from_db(
            db_path, last_rowid, agent_names
        )

        round_action_count = 0
        for action_data in actual_actions:
            if action_logger:
                action_logger.log_action(
                    round_num=round_num + 1,
                    agent_id=action_data["agent_id"],
                    agent_name=action_data["agent_name"],
                    action_type=action_data["action_type"],
                    action_args=action_data["action_args"],
                )
                total_actions += 1
                round_action_count += 1

        if action_logger:
            action_logger.log_round_end(round_num + 1, round_action_count)

        if (round_num + 1) % 20 == 0:
            progress = (round_num + 1) / total_rounds * 100
            log_info(
                f"Day {simulated_day}, {simulated_hour:02d}:00 - Round {round_num + 1}/{total_rounds} ({progress:.1f}%)"
            )

    # Nota: No cerrar entorno, mantener para Interview

    if action_logger:
        action_logger.log_simulation_end(total_rounds, total_actions)

    result.total_actions = total_actions
    elapsed = (datetime.now() - start_time).total_seconds()
    log_info(
        f"Bucle de simulacion completado! Tiempo: {elapsed:.1f}segundos, acciones totales: {total_actions}"
    )

    return result


async def run_reddit_simulation(
    config: Dict[str, Any],
    simulation_dir: str,
    action_logger: Optional[PlatformActionLogger] = None,
    main_logger: Optional[SimulationLogManager] = None,
    max_rounds: Optional[int] = None,
) -> PlatformSimulation:
    """Ejecutar simulacion Reddit

    Args:
        config: Configuracion de simulacion
        simulation_dir: Directorio de simulacion
        action_logger: Registrador de acciones
        main_logger: Gestor de logs principal
        max_rounds: Maximo de rondas de simulacion (opcional, para truncar simulaciones largas)

    Returns:
        PlatformSimulation: Objeto de resultados que contiene env y agent_graph
    """
    result = PlatformSimulation()

    def log_info(msg):
        if main_logger:
            main_logger.info(f"[Reddit] {msg}")
        print(f"[Reddit] {msg}")

    log_info("Inicializando...")

    # Reddit usa configuracion de LLM acelerada (si existe, si no usa general)
    model = create_model(config, use_boost=True)

    profile_path = os.path.join(simulation_dir, "reddit_profiles.json")
    if not os.path.exists(profile_path):
        log_info(f"Error: El archivo de perfil no existe: {profile_path}")
        return result

    result.agent_graph = await generate_reddit_agent_graph(
        profile_path=profile_path,
        model=model,
        available_actions=REDDIT_ACTIONS,
    )

    # Obtener mapeo de nombres reales de Agent desde config (usar entity_name en lugar de Agent_X por defecto)
    agent_names = get_agent_names_from_config(config)
    # Si no hay un agent en config, usar nombre por defecto de OASIS
    for agent_id, agent in result.agent_graph.get_agents():
        if agent_id not in agent_names:
            agent_names[agent_id] = getattr(agent, "name", f"Agent_{agent_id}")

    db_path = os.path.join(simulation_dir, "reddit_simulation.db")
    if os.path.exists(db_path):
        os.remove(db_path)

    result.env = oasis.make(
        agent_graph=result.agent_graph,
        platform=oasis.DefaultPlatformType.REDDIT,
        database_path=db_path,
        semaphore=30,  # Limitar maximo de solicitudes LLM concurrentes para evitar sobrecarga de API
    )

    await result.env.reset()
    log_info("Entorno iniciado")

    if action_logger:
        action_logger.log_simulation_start(config)

    total_actions = 0
    last_rowid = 0  # Rastrear ultimo numero de fila procesada (usar rowid para evitar diferencias de formato created_at)

    # Ejecutar eventos iniciales
    event_config = config.get("event_config", {})
    initial_posts = event_config.get("initial_posts", [])

    # Registrar inicio de round 0 (fase de eventos iniciales)
    if action_logger:
        action_logger.log_round_start(0, 0)  # round 0, simulated_hour 0

    initial_action_count = 0
    if initial_posts:
        initial_actions = {}
        for post in initial_posts:
            agent_id = post.get("poster_agent_id", 0)
            content = post.get("content", "")
            try:
                agent = result.env.agent_graph.get_agent(agent_id)
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

                if action_logger:
                    action_logger.log_action(
                        round_num=0,
                        agent_id=agent_id,
                        agent_name=agent_names.get(agent_id, f"Agent_{agent_id}"),
                        action_type="CREATE_POST",
                        action_args={"content": content},
                    )
                    total_actions += 1
                    initial_action_count += 1
            except Exception:
                pass

        if initial_actions:
            await result.env.step(initial_actions)
            log_info(f"Publicados {len(initial_actions)} posts iniciales")

    # Registrar fin de round 0
    if action_logger:
        action_logger.log_round_end(0, initial_action_count)

    # Bucle principal de simulacion
    time_config = config.get("time_config", {})
    total_hours = time_config.get("total_simulation_hours", 72)
    minutes_per_round = time_config.get("minutes_per_round", 30)
    total_rounds = (total_hours * 60) // minutes_per_round

    # Si se especifica maximo de rondas, truncar
    if max_rounds is not None and max_rounds > 0:
        original_rounds = total_rounds
        total_rounds = min(total_rounds, max_rounds)
        if total_rounds < original_rounds:
            log_info(
                f"Rondas truncadas: {original_rounds} -> {total_rounds} (max_rounds={max_rounds})"
            )

    start_time = datetime.now()

    for round_num in range(total_rounds):
        # Verificar si se recibio senal de salida
        if _shutdown_event and _shutdown_event.is_set():
            if main_logger:
                main_logger.info(
                    f"Recibida senal de salida, deteniendo simulacion en ronda {round_num + 1}"
                )
            break

        simulated_minutes = round_num * minutes_per_round
        simulated_hour = (simulated_minutes // 60) % 24
        simulated_day = simulated_minutes // (60 * 24) + 1

        active_agents = get_active_agents_for_round(
            result.env, config, simulated_hour, round_num
        )

        # Registrar inicio de round independientemente de si hay agents activos
        if action_logger:
            action_logger.log_round_start(round_num + 1, simulated_hour)

        if not active_agents:
            # Registrar fin de round sin agents activos (actions_count=0)
            if action_logger:
                action_logger.log_round_end(round_num + 1, 0)
            continue

        actions = {agent: LLMAction() for _, agent in active_agents}
        await result.env.step(actions)

        # Obtener y registrar acciones realmente ejecutadas de la base de datos
        actual_actions, last_rowid = fetch_new_actions_from_db(
            db_path, last_rowid, agent_names
        )

        round_action_count = 0
        for action_data in actual_actions:
            if action_logger:
                action_logger.log_action(
                    round_num=round_num + 1,
                    agent_id=action_data["agent_id"],
                    agent_name=action_data["agent_name"],
                    action_type=action_data["action_type"],
                    action_args=action_data["action_args"],
                )
                total_actions += 1
                round_action_count += 1

        if action_logger:
            action_logger.log_round_end(round_num + 1, round_action_count)

        if (round_num + 1) % 20 == 0:
            progress = (round_num + 1) / total_rounds * 100
            log_info(
                f"Day {simulated_day}, {simulated_hour:02d}:00 - Round {round_num + 1}/{total_rounds} ({progress:.1f}%)"
            )

    # Nota: No cerrar entorno, mantener para Interview

    if action_logger:
        action_logger.log_simulation_end(total_rounds, total_actions)

    result.total_actions = total_actions
    elapsed = (datetime.now() - start_time).total_seconds()
    log_info(
        f"Bucle de simulacion completado! Tiempo: {elapsed:.1f}segundos, acciones totales: {total_actions}"
    )

    return result


async def main():
    parser = argparse.ArgumentParser(description="Simulacion paralela dual de OASIS")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Ruta del archivo de configuracion (simulation_config.json)",
    )
    parser.add_argument(
        "--twitter-only", action="store_true", help="Solo ejecutar simulacion Twitter"
    )
    parser.add_argument(
        "--reddit-only", action="store_true", help="Solo ejecutar simulacion Reddit"
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=None,
        help="Maximo de rondas de simulacion (opcional, para truncar simulaciones largas)",
    )
    parser.add_argument(
        "--no-wait",
        action="store_true",
        default=False,
        help="Cerrar entorno inmediatamente despues de la simulacion, no entrar en modo de espera de comandos",
    )

    args = parser.parse_args()

    # Crear evento de shutdown al inicio de main para que todo el programa pueda responder a senales de salida
    global _shutdown_event
    _shutdown_event = asyncio.Event()

    if not os.path.exists(args.config):
        print(f"Error: Archivo de configuracion no existe: {args.config}")
        sys.exit(1)

    config = load_config(args.config)
    simulation_dir = os.path.dirname(args.config) or "."
    wait_for_commands = not args.no_wait

    # Inicializar configuracion de logs (desactivar logs de OASIS, limpiar archivos antiguos)
    init_logging_for_simulation(simulation_dir)

    # Crear gestor de logs
    log_manager = SimulationLogManager(simulation_dir)
    twitter_logger = log_manager.get_twitter_logger()
    reddit_logger = log_manager.get_reddit_logger()

    log_manager.info("=" * 60)
    log_manager.info("Simulacion paralela dual de OASIS")
    log_manager.info(f"Archivo de configuracion: {args.config}")
    log_manager.info(f"ID de simulacion: {config.get('simulation_id', 'unknown')}")
    log_manager.info(
        f"Modo de espera de comandos: {'Activado' if wait_for_commands else 'Desactivado'}"
    )
    log_manager.info("=" * 60)

    time_config = config.get("time_config", {})
    total_hours = time_config.get("total_simulation_hours", 72)
    minutes_per_round = time_config.get("minutes_per_round", 30)
    config_total_rounds = (total_hours * 60) // minutes_per_round

    log_manager.info("Parametros de simulacion:")
    log_manager.info(f"  - Duracion total de simulacion: {total_hours} horas")
    log_manager.info(f"  - Tiempo por ronda: {minutes_per_round} minutos")
    log_manager.info(f"  - Total de rondas configuradas: {config_total_rounds}")
    if args.max_rounds:
        log_manager.info(f"  - maximo de rondas limite: {args.max_rounds}")
        if args.max_rounds < config_total_rounds:
            log_manager.info(
                f"  - Rondas realmente ejecutadas: {args.max_rounds} (truncado)"
            )
    log_manager.info(f"  - Cantidad de Agents: {len(config.get('agent_configs', []))}")

    log_manager.info("Estructura de logs:")
    log_manager.info("  - Log principal: simulation.log")
    log_manager.info("  - Acciones Twitter: twitter/actions.jsonl")
    log_manager.info("  - Acciones Reddit: reddit/actions.jsonl")
    log_manager.info("=" * 60)

    start_time = datetime.now()

    # Almacenar resultados de simulacion de dos plataformas
    twitter_result: Optional[PlatformSimulation] = None
    reddit_result: Optional[PlatformSimulation] = None

    if args.twitter_only:
        twitter_result = await run_twitter_simulation(
            config, simulation_dir, twitter_logger, log_manager, args.max_rounds
        )
    elif args.reddit_only:
        reddit_result = await run_reddit_simulation(
            config, simulation_dir, reddit_logger, log_manager, args.max_rounds
        )
    else:
        # Ejecutar en paralelo (cada plataforma usa su propio registrador)
        results = await asyncio.gather(
            run_twitter_simulation(
                config, simulation_dir, twitter_logger, log_manager, args.max_rounds
            ),
            run_reddit_simulation(
                config, simulation_dir, reddit_logger, log_manager, args.max_rounds
            ),
        )
        twitter_result, reddit_result = results

    total_elapsed = (datetime.now() - start_time).total_seconds()
    log_manager.info("=" * 60)
    log_manager.info(f"Bucle de simulacion completado! Tiempo total: {total_elapsed:.1f}segundos")

    # Entrar en modo de espera de comandos
    if wait_for_commands:
        log_manager.info("")
        log_manager.info("=" * 60)
        log_manager.info(
            "Entrando en modo de espera de comandos - Entorno sigue ejecutandose"
        )
        log_manager.info("Comandos soportados: interview, batch_interview, close_env")
        log_manager.info("=" * 60)

        # Crear procesador IPC
        ipc_handler = ParallelIPCHandler(
            simulation_dir=simulation_dir,
            twitter_env=twitter_result.env if twitter_result else None,
            twitter_agent_graph=twitter_result.agent_graph if twitter_result else None,
            reddit_env=reddit_result.env if reddit_result else None,
            reddit_agent_graph=reddit_result.agent_graph if reddit_result else None,
        )
        ipc_handler.update_status("alive")

        # Bucle de espera de comandos (usar _shutdown_event global)
        try:
            while not _shutdown_event.is_set():
                should_continue = await ipc_handler.process_commands()
                if not should_continue:
                    break
                # Usar wait_for en lugar de sleep para poder responder a shutdown_event
                try:
                    await asyncio.wait_for(_shutdown_event.wait(), timeout=0.5)
                    break  # Senal de salida recibida
                except asyncio.TimeoutError:
                    pass  # Timeout continuar bucle
        except KeyboardInterrupt:
            print("\nRecibida senal de interrupcion")
        except asyncio.CancelledError:
            print("\nTarea cancelada")
        except Exception as e:
            print(f"\nError al procesar comandos: {e}")

        log_manager.info("\nCerrando entorno...")
        ipc_handler.update_status("stopped")

    # Cerrando entorno
    if twitter_result and twitter_result.env:
        await twitter_result.env.close()
        log_manager.info("[Twitter] entorno cerrado")

    if reddit_result and reddit_result.env:
        await reddit_result.env.close()
        log_manager.info("[Reddit] entorno cerrado")

    log_manager.info("=" * 60)
    log_manager.info(f"Todo completado!")
    log_manager.info(f"Archivos de log:")
    log_manager.info(f"  - {os.path.join(simulation_dir, 'simulation.log')}")
    log_manager.info(f"  - {os.path.join(simulation_dir, 'twitter', 'actions.jsonl')}")
    log_manager.info(f"  - {os.path.join(simulation_dir, 'reddit', 'actions.jsonl')}")
    log_manager.info("=" * 60)


def setup_signal_handlers(loop=None):
    """
    Configurar manejador de senales, asegurar salida correcta al recibir SIGTERM/SIGINT

    Escenario de simulacion persistente: No salir despues de la simulacion, esperar comandos de interview
    Cuando se reciba una senal de terminacion, se necesita:
    1. Notificar al bucle asyncio para salir de la espera
    2. Dar al programa oportunidad de limpiar recursos normalmente (cerrar base de datos, entorno, etc.)
    3. Luego salir
    """

    def signal_handler(signum, frame):
        global _cleanup_done
        sig_name = "SIGTERM" if signum == signal.SIGTERM else "SIGINT"
        print(f"\nRecibida senal {sig_name}, saliendo...")

        if not _cleanup_done:
            _cleanup_done = True
            # Configurar evento para notificar al bucle asyncio de salida (dar oportunidad de limpiar recursos)
            if _shutdown_event:
                _shutdown_event.set()

        # No llamar sys.exit() directamente, dejar que el bucle asyncio salga normalmente y limpie recursos
        # Solo forzar salida si se recibe senal repetidamente
        else:
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
        # Limpiar rastreador de recursos de multiprocessing (evitar advertencia al salir)
        try:
            from multiprocessing import resource_tracker

            resource_tracker._resource_tracker._stop()
        except Exception:
            pass
        print("Proceso de simulacion terminado")
