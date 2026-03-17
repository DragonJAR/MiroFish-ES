"""
Gestión de configuración
Cargar configuración uniformemente desde el archivo .env en la raíz del proyecto
"""

import os
from dotenv import load_dotenv

# Cargar el archivo .env desde la raíz del proyecto
# Ruta: MiroFish/.env (relativo a backend/app/config.py)
project_root_env = os.path.join(os.path.dirname(__file__), "../../.env")

if os.path.exists(project_root_env):
    load_dotenv(project_root_env, override=True)
else:
    # Si no existe .env en la raíz, intentar cargar variables de entorno (para producción)
    load_dotenv(override=True)


class Config:
    """Clase de configuración Flask"""

    # Configuración Flask
    SECRET_KEY = os.environ.get("SECRET_KEY", "mirofish-secret-key")
    DEBUG = os.environ.get("FLASK_DEBUG", "True").lower() == "true"

    # Configuración JSON - Deshabilitar escape ASCII, mostrar chino directamente (en lugar de formato \uXXXX)
    JSON_AS_ASCII = False

    # Configuración LLM (usar formato OpenAI uniforme)
    LLM_API_KEY = os.environ.get("LLM_API_KEY")
    LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "https://api.z.ai/api/paas/v4")
    LLM_MODEL_NAME = os.environ.get("LLM_MODEL_NAME", "glm-4.5")

    # Configuración LLM Fallback (cuando el principal falla)
    LLM_FALLBACK_API_KEY = os.environ.get("LLM_FALLBACK_API_KEY")
    LLM_FALLBACK_BASE_URL = os.environ.get(
        "LLM_FALLBACK_BASE_URL", "https://api.minimax.io/v1"
    )
    LLM_FALLBACK_MODEL = os.environ.get("LLM_FALLBACK_MODEL", "MiniMax-M2.5")

    # Configuración Zep
    ZEP_API_KEY = os.environ.get("ZEP_API_KEY")

    # Configuración de subida de archivos
    MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50MB
    UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "../uploads")
    ALLOWED_EXTENSIONS = {"pdf", "md", "txt", "markdown"}

    # Configuración de procesamiento de texto
    DEFAULT_CHUNK_SIZE = 500  # Tamaño de fragmento predeterminado
    DEFAULT_CHUNK_OVERLAP = 50  # Tamaño de superposición predeterminado

    # Configuración de simulación OASIS
    OASIS_DEFAULT_MAX_ROUNDS = int(os.environ.get("OASIS_DEFAULT_MAX_ROUNDS", "10"))
    OASIS_SIMULATION_DATA_DIR = os.path.join(
        os.path.dirname(__file__), "../uploads/simulations"
    )

    # Configuración de acciones disponibles de plataforma OASIS
    OASIS_TWITTER_ACTIONS = [
        "CREATE_POST",
        "LIKE_POST",
        "REPOST",
        "FOLLOW",
        "DO_NOTHING",
        "QUOTE_POST",
    ]
    OASIS_REDDIT_ACTIONS = [
        "LIKE_POST",
        "DISLIKE_POST",
        "CREATE_POST",
        "CREATE_COMMENT",
        "LIKE_COMMENT",
        "DISLIKE_COMMENT",
        "SEARCH_POSTS",
        "SEARCH_USER",
        "TREND",
        "REFRESH",
        "DO_NOTHING",
        "FOLLOW",
        "MUTE",
    ]

    # Configuración de Report Agent
    REPORT_AGENT_MAX_TOOL_CALLS = int(
        os.environ.get("REPORT_AGENT_MAX_TOOL_CALLS", "5")
    )
    REPORT_AGENT_MAX_REFLECTION_ROUNDS = int(
        os.environ.get("REPORT_AGENT_MAX_REFLECTION_ROUNDS", "2")
    )
    REPORT_AGENT_TEMPERATURE = float(os.environ.get("REPORT_AGENT_TEMPERATURE", "0.5"))

    @classmethod
    def validate(cls):
        """Validar configuraciones necesarias"""
        errors = []
        if not cls.LLM_API_KEY:
            errors.append("LLM_API_KEY no está configurado")
        if not cls.ZEP_API_KEY:
            errors.append("ZEP_API_KEY no está configurado")
        return errors
