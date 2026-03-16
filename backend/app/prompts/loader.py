"""
Cargador de prompts internacionalizados.
Carga prompts desde archivos JSON según el locale configurado.
"""

import json
import os
from typing import Dict, Optional, Any
from functools import lru_cache

from . import config


# Directorio base de templates
TEMPLATES_DIR = os.path.join(os.path.dirname(__file__), "templates")


def _get_locale() -> str:
    """Obtiene el locale actual."""
    return config.get_backend_locale()


@lru_cache(maxsize=32)
def _load_prompts_file(service: str, locale: str) -> Dict[str, Any]:
    """
    Carga un archivo de prompts desde el sistema de archivos.

    Args:
        service: Nombre del servicio (zep, simulation, ontology, oasis, report)
        locale: Locale ('es' o 'zh')

    Returns:
        Diccionario de prompts
    """
    file_path = os.path.join(TEMPLATES_DIR, service, f"{locale}.json")

    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)

    return {}


def get_prompt(service: str, key: str, default: Optional[str] = None) -> str:
    """
    Obtiene un prompt específico según el locale.

    Args:
        service: Nombre del servicio (zep, simulation, ontology, oasis, report)
        key: Clave del prompt
        default: Valor por defecto si no se encuentra

    Returns:
        El prompt solicitado o el default
    """
    locale = _get_locale()

    # Intentar cargar del archivo del locale actual
    prompts = _load_prompts_file(service, locale)

    if key in prompts:
        return prompts[key]

    # Fallback a zh si no se encuentra
    if locale != "zh":
        prompts_zh = _load_prompts_file(service, "zh")
        if key in prompts_zh:
            return prompts_zh[key]

    # Devolver default o la key si no hay default
    return default if default is not None else key


def load_prompt(service: str, key: str, **kwargs) -> str:
    """
    Obtiene un prompt y lo formatea con los argumentos dados.

    Args:
        service: Nombre del servicio
        key: Clave del prompt
        **kwargs: Argumentos para formatear el prompt

    Returns:
        Prompt formateado
    """
    template = get_prompt(service, key)

    if kwargs:
        try:
            return template.format(**kwargs)
        except (KeyError, ValueError):
            # Si falla el formateo, devolver el template original
            return template

    return template


# Alias para compatibilidad
def get_locale() -> str:
    """Alias para obtener el locale actual."""
    return _get_locale()
