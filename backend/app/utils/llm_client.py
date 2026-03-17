"""
Cliente LLM
Utiliza formato OpenAI para todas las llamadas
"""

import json
import re
import time
from typing import Optional, Dict, Any, List
from openai import OpenAI
from openai import RateLimitError, APIError, Timeout

from ..config import Config


class LLMClient:
    """Cliente LLM con soporte para fallback automatico"""

    # Configuracion de reintentos
    MAX_RETRIES = 5
    INITIAL_DELAY = 2  # segundos
    MAX_DELAY = 120  # segundos
    TIMEOUT = 180  # 3 minutos

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        use_fallback: bool = False,
    ):
        # Configuracion del proveedor principal
        self.api_key = api_key or Config.LLM_API_KEY
        self.base_url = base_url or Config.LLM_BASE_URL
        self.model = model or Config.LLM_MODEL_NAME

        # Configuracion del proveedor fallback
        self.fallback_api_key = getattr(Config, "LLM_FALLBACK_API_KEY", None)
        self.fallback_base_url = getattr(Config, "LLM_FALLBACK_BASE_URL", None)
        self.fallback_model = getattr(Config, "LLM_FALLBACK_MODEL", None)

        # Si use_fallback es True, usar el proveedor fallback directamente
        if use_fallback and self.fallback_api_key:
            self.api_key = self.fallback_api_key
            self.base_url = self.fallback_base_url or "https://api.minimax.io/v1"
            self.model = self.fallback_model or "MiniMax-M2.5"

        if not self.api_key:
            raise ValueError("LLM_API_KEY no configurada")

        self.client = OpenAI(
            api_key=self.api_key, base_url=self.base_url, timeout=self.TIMEOUT
        )

    def _calculate_delay(self, attempt: int) -> float:
        """Calcula el delay con exponential backoff"""
        delay = self.INITIAL_DELAY * (2**attempt)
        return min(delay, self.MAX_DELAY)

    def _is_retryable_error(self, error: Exception) -> bool:
        """Determina si el error es retryable"""
        if isinstance(error, RateLimitError):
            return True
        if isinstance(error, APIError):
            return True
        if isinstance(error, Timeout):
            return True
        if isinstance(error, (ConnectionError, OSError)):
            return True
        return False

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 4096,
        response_format: Optional[Dict] = None,
    ) -> str:
        """
        Enviar peticion de chat - con fallback automatico

        Si el proveedor principal falla (rate limit), automaticamente
        cambia al proveedor fallback y reintenta.
        """
        # Intentar con el proveedor principal
        try:
            return self._chat_with_retries(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format=response_format,
            )
        except Exception as primary_error:
            # Si hay error y tenemos fallback configurado, probar con fallback
            if self.fallback_api_key and self._is_retryable_error(primary_error):
                print(
                    f"⚠️ Proveedor principal fallo: {type(primary_error).__name__}. Cambiando a fallback..."
                )
                return self._chat_with_fallback(
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    response_format=response_format,
                )
            # No hay fallback o error no retryable, lanzar
            raise

    def _chat_with_fallback(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 4096,
        response_format: Optional[Dict] = None,
    ) -> str:
        """
        Ejecuta chat usando el proveedor fallback.
        Crea un cliente temporal con la configuracion fallback.
        """
        # Crear cliente temporal con fallback
        fallback_client = LLMClient(
            api_key=self.fallback_api_key,
            base_url=self.fallback_base_url or "https://api.minimax.io/v1",
            model=self.fallback_model or "MiniMax-M2.5",
            use_fallback=True,
        )

        try:
            return fallback_client._chat_with_retries(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format=response_format,
            )
        except Exception as fallback_error:
            print(f"❌ Fallback tambion fallo: {fallback_error}")
            raise

    def _chat_with_retries(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 4096,
        response_format: Optional[Dict] = None,
    ) -> str:
        """
        Metodo interno que ejecuta el chat con reintentos.
        """
        last_error = None

        for attempt in range(self.MAX_RETRIES):
            try:
                kwargs = {
                    "model": self.model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                }

                if response_format:
                    kwargs["response_format"] = response_format

                response = self.client.chat.completions.create(**kwargs)
                content = response.choices[0].message.content

                if content is None:
                    raise ValueError("El LLM devolvio contenido vacio")

                # Si es JSON mode, remover markdown si existe
                if response_format and response_format.get("type") == "json_object":
                    content = re.sub(
                        r"^```(?:json)?\s*\n?", "", content, flags=re.IGNORECASE
                    )
                    content = re.sub(r"\n?```\s*$", "", content)
                    content = content.strip()

                return content

            except Exception as e:
                last_error = e

                if not self._is_retryable_error(e):
                    raise

                if attempt < self.MAX_RETRIES - 1:
                    delay = self._calculate_delay(attempt)
                    print(
                        f"⚠️ Error en LLM (intento {attempt + 1}/{self.MAX_RETRIES}): {type(e).__name__}. Reintentando en {delay}s..."
                    )
                    time.sleep(delay)
                else:
                    print(f"❌ LLM fallo despues de {self.MAX_RETRIES} intentos")
                    raise

        raise last_error if last_error else Exception("Error desconocido en LLM")

    def chat_json(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 4096,
    ) -> Dict[str, Any]:
        """
        Enviar peticion de chat y devolver JSON

        Args:
            messages: Lista de mensajes
            temperature: Parametro de temperatura
            max_tokens: Maximo de tokens

        Returns:
            Objeto JSON parseado
        """
        response = self.chat(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )
        # Limpiar marcadores de codigo markdown
        cleaned_response = response.strip()
        cleaned_response = re.sub(
            r"^```(?:json)?\s*\n?", "", cleaned_response, flags=re.IGNORECASE
        )
        cleaned_response = re.sub(r"\n?```\s*$", "", cleaned_response)
        cleaned_response = cleaned_response.strip()

        try:
            return json.loads(cleaned_response)
        except json.JSONDecodeError:
            raise ValueError(f"El JSON devuelto por el LLM es invalido: {cleaned_response}")
