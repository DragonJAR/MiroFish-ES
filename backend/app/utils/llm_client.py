"""
Cliente LLM centralizado con rate limiting adaptativo y circuit breaker.

Este módulo es la ÚNICA fuente de verdad para rate limits en MiroFish.
Todo el código que hace llamadas LLM debe pasar por aquí o por el
CamelRateLimitWrapper (para código CAMEL-OASIS).

Patrón DRY: la lógica de retry, fallback y rate limit está aquí,
no dispersada en múltiples lugares.

Arquitectura:
  LLMClient (este archivo)
    ├── Retry con exponential backoff + jitter
    ├── Fallback a segundo proveedor (MiniMax) cuando primario falla
    ├── Circuit breaker (N fallos consecutivos → auto-switch a fallback)
    ├── Rate limit adaptativo (lee Retry-After header)
    └── Global cooldown lock (thread-safe, serializa requests cuando hay 429)

  CamelRateLimitWrapper (camel_rate_limit_wrapper.py)
    └── Envuelve ChatAgent de CAMEL para que use LLMClient
       en vez de su propio retry infinito con backoff corto.
"""

import json
import re
import time
import random
import threading
import logging
from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass, field
from openai import OpenAI
from openai import RateLimitError, APIError, Timeout

from ..config import Config

logger = logging.getLogger("mirofish.llm_client")


@dataclass
class RateLimitConfig:
    """Configuración de rate limiting para LLMClient."""

    max_retries: int = 5
    initial_delay: float = 2.0  # segundos
    max_delay: float = 120.0  # segundos
    timeout: float = 180.0  # 3 minutos
    # Circuit breaker
    consecutive_failures_to_trip: int = 3
    cooldown_after_trip: float = 30.0  # segundos antes de reintentar primario
    # Retry-After header (estándar HTTP 429)
    respect_retry_after_header: bool = True
    retry_after_default: float = 10.0  # segundos si no hay header


@dataclass
class CircuitBreakerState:
    """Estado interno del circuit breaker por proveedor."""

    consecutive_failures: int = 0
    is_open: bool = False  # True = no se usa este proveedor
    last_failure_time: float = 0.0
    lock: threading.Lock = field(default_factory=threading.Lock)


class LLMClient:
    """
    Cliente LLM con soporte para fallback automático, circuit breaker
    y rate limiting adaptativo.

    El circuit breaker monitorea fallos consecutivos de cada proveedor.
    Cuando se tripped (N fallos seguidos), auto-switch al fallback.
    Después del cooldown, reintenta el primario una vez — si falla, vuelve al fallback.

    Usage:
        client = LLMClient()  # usa Config.LLM_*
        response = client.chat(messages=[...])

        # Para código CAMEL, usar CamelRateLimitWrapper en vez de esto.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        use_fallback: bool = False,
        config: Optional[RateLimitConfig] = None,
    ):
        # Configuración de reintentos y rate limiting
        self.cfg = config or RateLimitConfig()

        # Configuración del proveedor principal
        self.api_key = api_key or Config.LLM_API_KEY
        self.base_url = base_url or Config.LLM_BASE_URL
        self.model = model or Config.LLM_MODEL_NAME

        # Configuración del proveedor fallback
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
            api_key=self.api_key, base_url=self.base_url, timeout=self.cfg.timeout
        )

        # Circuit breaker state — uno por proveedor
        self._primary_cb = CircuitBreakerState()
        self._fallback_cb = CircuitBreakerState()

        # Lock global para rate limiting (serializa todas las requests)
        # Cuando un proveedor devuelve 429, todas las requests esperan
        # un cooldown global antes de continuar
        self._global_lock = threading.Lock()
        self._global_cooldown_until: float = 0.0
        self._provider_cooldown_until: Dict[str, float] = {}  # por proveedor

    # ─── Circuit breaker helpers ────────────────────────────────────────────

    def _check_circuit_breaker(
        self, cb: CircuitBreakerState, provider_name: str
    ) -> bool:
        """Retorna True si el circuit breaker está OPEN (proveedor no disponible)."""
        with cb.lock:
            if not cb.is_open:
                return False
            # ¿Pasó el cooldown?
            if time.time() >= cb.last_failure_time + self.cfg.cooldown_after_trip:
                # Reintentar una vez
                cb.is_open = False
                cb.consecutive_failures = 0
                logger.info(f"Circuit breaker: {provider_name} closed, retrying")
                return False
            return True

    def _trip_circuit_breaker(self, cb: CircuitBreakerState, provider_name: str):
        """Abre el circuit breaker después de fallos consecutivos."""
        with cb.lock:
            cb.consecutive_failures += 1
            if cb.consecutive_failures >= self.cfg.consecutive_failures_to_trip:
                cb.is_open = True
                cb.last_failure_time = time.time()
                logger.warning(
                    f"Circuit breaker OPEN for {provider_name} after "
                    f"{cb.consecutive_failures} consecutive failures. "
                    f"Will retry after {self.cfg.cooldown_after_trip}s"
                )

    def _record_success(self, cb: CircuitBreakerState):
        """Resetea el circuit breaker tras un éxito."""
        with cb.lock:
            cb.consecutive_failures = 0
            cb.is_open = False

    # ─── Rate limit helpers ──────────────────────────────────────────────────

    def _is_retryable_error(self, error: Exception) -> bool:
        """Determina si el error es retryable (rate limit, network, etc)."""
        if isinstance(error, RateLimitError):
            return True
        if isinstance(error, APIError):
            return True
        if isinstance(error, Timeout):
            return True
        if isinstance(error, (ConnectionError, OSError)):
            return True
        return False

    def _extract_retry_after(self, error: Exception) -> Optional[float]:
        """Extrae Retry-After header de una excepción RateLimitError."""
        if isinstance(error, RateLimitError):
            # RateLimitError puede tener headers con retry_after
            if hasattr(error, "response") and error.response is not None:
                retry_after = error.response.headers.get("retry-after")
                if retry_after:
                    try:
                        return float(retry_after)
                    except (ValueError, TypeError):
                        pass
        return None

    def _calculate_delay(
        self, attempt: int, retry_after: Optional[float] = None
    ) -> float:
        """Calcula el delay con exponential backoff, respetando Retry-After si está disponible."""
        if retry_after and self.cfg.respect_retry_after_header:
            # Retry-After es la fuente definitiva
            return min(float(retry_after), self.cfg.max_delay)
        # Exponential backoff con jitter
        delay = self.cfg.initial_delay * (2**attempt)
        jitter = random.uniform(0.5, 1.5)
        return min(delay * jitter, self.cfg.max_delay)

    def _acquire_rate_limit_lock(self, provider_name: str):
        """
        Adquiere el lock global de rate limiting.
        Si otro thread ya está en cooldown por 429, esperamos.
        """
        while True:
            self._global_lock.acquire()
            now = time.time()
            # ¿Hay cooldown activo en algún proveedor?
            active_cooldown = False
            for provider, until in list(self._provider_cooldown_until.items()):
                if now < until:
                    active_cooldown = True
                    remaining = until - now
                    self._global_lock.release()
                    logger.debug(
                        f"Rate limit cooldown active for {provider}: {remaining:.1f}s remaining"
                    )
                    time.sleep(
                        min(remaining + 0.1, 5.0)
                    )  # esperar al menor de: resto o 5s
                    break
            if not active_cooldown:
                break

    def _release_rate_limit_lock(self):
        """Libera el lock global."""
        self._global_lock.release()

    def _set_provider_cooldown(self, provider_name: str, delay: float):
        """Registra un cooldown para un proveedor específico."""
        self._provider_cooldown_until[provider_name] = time.time() + delay

    # ─── Chat methods ────────────────────────────────────────────────────────

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 4096,
        response_format: Optional[Dict] = None,
    ) -> str:
        """
        Enviar petición de chat - con retry, fallback y circuit breaker.

        Si el proveedor principal falla por rate limit u otro error retryable,
        reintenta con backoff. Si el circuit breaker está abierto, salta
        directamente al fallback. Si se agotan los retries del primario,
        intenta el fallback una vez.
        """
        # Check circuit breaker del primario
        if self._check_circuit_breaker(self._primary_cb, "primary"):
            logger.info(
                "Primary provider circuit breaker open, using fallback directly"
            )
            return self._chat_with_fallback(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format=response_format,
            )

        # Intentar con el proveedor principal
        try:
            return self._chat_with_retries(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format=response_format,
                cb=self._primary_cb,
            )
        except Exception as primary_error:
            if self.fallback_api_key and self._is_retryable_error(primary_error):
                # Trip circuit breaker para primario
                self._trip_circuit_breaker(self._primary_cb, "primary")
                logger.warning(
                    f"Primary provider failed ({type(primary_error).__name__}). "
                    f"Switching to fallback. "
                    f"Primary CB consecutive failures: {self._primary_cb.consecutive_failures}"
                )
                return self._chat_with_fallback(
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    response_format=response_format,
                )
            raise

    def _chat_with_fallback(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 4096,
        response_format: Optional[Dict] = None,
    ) -> str:
        """Ejecuta chat usando el proveedor fallback con su propio circuit breaker."""
        if not self.fallback_api_key:
            raise RuntimeError("Fallback provider not configured but primary failed")

        # Check fallback circuit breaker
        if self._check_circuit_breaker(self._fallback_cb, "fallback"):
            raise RuntimeError(
                f"Fallback provider circuit breaker is open. "
                f"Both primary and fallback are unavailable."
            )

        fallback_client = LLMClient(
            api_key=self.fallback_api_key,
            base_url=self.fallback_base_url or "https://api.minimax.io/v1",
            model=self.fallback_model or "MiniMax-M2.5",
            use_fallback=True,
        )
        # Hereda la config pero usa sus propios CB states
        fallback_client._fallback_cb = (
            self._fallback_cb
        )  # compartir estado de CB del fallback

        try:
            return fallback_client._chat_with_retries(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format=response_format,
                cb=self._fallback_cb,
            )
        except Exception as fallback_error:
            self._trip_circuit_breaker(self._fallback_cb, "fallback")
            logger.error(f"Fallback provider also failed: {fallback_error}")
            raise RuntimeError(
                f"Both primary and fallback LLM providers failed: "
                f"primary={primary_error if 'primary_error' in dir() else 'N/A'}, "
                f"fallback={fallback_error}"
            )

    def _chat_with_retries(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
        response_format: Optional[Dict],
        cb: CircuitBreakerState,
    ) -> str:
        """Método interno que ejecuta el chat con reintentos y rate limiting."""
        last_error = None

        for attempt in range(self.cfg.max_retries):
            self._acquire_rate_limit_lock(self.model)

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
                    raise ValueError("El LLM devolvió contenido vacío")

                # Limpiar markdown si es JSON mode
                if response_format and response_format.get("type") == "json_object":
                    content = re.sub(
                        r"^```(?:json)?\s*\n?", "", content, flags=re.IGNORECASE
                    )
                    content = re.sub(r"\n?```\s*$", "", content)
                    content = content.strip()

                # Éxito — reset circuit breaker
                self._record_success(cb)
                self._release_rate_limit_lock()
                return content

            except RateLimitError as e:
                self._release_rate_limit_lock()
                last_error = e
                retry_after = self._extract_retry_after(e)
                delay = self._calculate_delay(attempt, retry_after)

                if retry_after:
                    logger.warning(
                        f"Rate limit (Retry-After={retry_after}s) on {self.model} "
                        f"(attempt {attempt + 1}/{self.cfg.max_retries}). "
                        f"Waiting {delay:.1f}s before retry..."
                    )
                else:
                    logger.warning(
                        f"Rate limit on {self.model} "
                        f"(attempt {attempt + 1}/{self.cfg.max_retries}): "
                        f"{type(e).__name__}. Retrying in {delay:.1f}s..."
                    )

                # Registrar cooldown para este proveedor
                self._set_provider_cooldown(self.model, delay)

                if attempt < self.cfg.max_retries - 1:
                    time.sleep(delay)
                # Si es el último intento, el circuit breaker se trippea desde chat()

            except Exception as e:
                self._release_rate_limit_lock()
                last_error = e

                if not self._is_retryable_error(e):
                    self._record_success(cb)
                    raise

                if attempt < self.cfg.max_retries - 1:
                    delay = self._calculate_delay(attempt)
                    logger.warning(
                        f"LLM error on {self.model} "
                        f"(attempt {attempt + 1}/{self.cfg.max_retries}): "
                        f"{type(e).__name__}: {str(e)[:100]}. "
                        f"Retrying in {delay:.1f}s..."
                    )
                    time.sleep(delay)

            finally:
                # Siempre liberar lock al final del try/except
                pass

        # Agotamos retries
        self._trip_circuit_breaker(cb, self.model)
        logger.error(f"LLM failed after {self.cfg.max_retries} retries on {self.model}")
        raise last_error or Exception(
            f"Unknown error in LLM chat after {self.cfg.max_retries} retries"
        )

    def chat_json(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 4096,
    ) -> Dict[str, Any]:
        """
        Enviar petición de chat y devolver JSON parseado.

        Utiliza chat() como base, luego aplica 3 capas de parseo:
        1. json.loads directo
        2. json_repair strict (errores de sintaxis)
        3. json_repair lenient (reestructuración)
        """
        response = self.chat(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )

        cleaned = response.strip()
        cleaned = re.sub(r"^```(?:json)?\s*\n?", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\n?```\s*$", "", cleaned)
        cleaned = cleaned.strip()

        # Attempt 1: standard JSON parse
        try:
            result = json.loads(cleaned)
            if isinstance(result, dict):
                return result
            raise ValueError(
                f"LLM JSON returned non-dict type: {type(result).__name__}"
            )
        except (json.JSONDecodeError, ValueError) as e:
            logger.debug(f"JSON parse failed (attempt 1): {e}")

        # Attempt 2: json_repair strict
        try:
            repaired = json_repair.loads(cleaned, strict=True)
            if isinstance(repaired, dict):
                return repaired
        except Exception as e:
            logger.debug(f"JSON repair strict failed: {e}")

        # Attempt 3: json_repair lenient
        try:
            logger.warning(
                f"Using lenient JSON repair. Raw (first 200): {cleaned[:200]}"
            )
            repaired = json_repair.loads(cleaned, strict=False)
            if isinstance(repaired, dict):
                return repaired
        except Exception as e:
            logger.error(f"JSON repair lenient failed: {e}")

        raise ValueError(
            f"Invalid JSON from LLM (could not parse or repair): "
            f"first 200 chars: {cleaned[:200]}"
        )
