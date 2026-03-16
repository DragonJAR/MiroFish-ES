"""
Sistema de internacionalización (i18n) para prompts del backend.

Uso:
    from app.prompts import load_prompt, get_locale

    # Obtener un prompt
    prompt = load_prompt("zep", "sub_query_generation")

    # Obtener un prompt con formateo
    prompt = load_prompt("report", "section_template", section_title="Introducción")
"""

from .loader import load_prompt, get_prompt, get_locale
from .config import get_backend_locale, set_backend_locale

__all__ = [
    "load_prompt",
    "get_prompt",
    "get_locale",
    "get_backend_locale",
    "set_backend_locale",
]
