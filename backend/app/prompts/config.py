"""
Configuración de locale para prompts del backend.
"""

import os
from typing import Optional


def get_backend_locale() -> str:
    """
    Obtiene el locale configurado para el backend.

    Returns:
        Locale ('es' o 'zh'), por defecto 'es'
    """
    return os.environ.get("BACKEND_LOCALE", "es")


def set_backend_locale(locale: str):
    """Establece el locale para el backend."""
    os.environ["BACKEND_LOCALE"] = locale


# Locale actual
CURRENT_LOCALE = get_backend_locale()
