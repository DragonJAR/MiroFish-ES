"""
LLM客户端封装
统一使用OpenAI格式调用
"""

import json
import re
import time
from typing import Optional, Dict, Any, List
from openai import OpenAI
from openai import RateLimitError, APIError, Timeout

from ..config import Config


class LLMClient:
    """LLM客户端"""

    # Configuración de reintentos
    MAX_RETRIES = 5
    INITIAL_DELAY = 2  # segundos
    MAX_DELAY = 120  # segundos
    TIMEOUT = 180  # 3 minutos

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
    ):
        self.api_key = api_key or Config.LLM_API_KEY
        self.base_url = base_url or Config.LLM_BASE_URL
        self.model = model or Config.LLM_MODEL_NAME

        if not self.api_key:
            raise ValueError("LLM_API_KEY 未配置")

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
        发送聊天请求

        Args:
            messages: 消息列表
            temperature: 温度参数
            max_tokens: 最大token数
            response_format: 响应格式（如JSON模式）

        Returns:
            模型响应文本
        """
        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        if response_format:
            kwargs["response_format"] = response_format

        # Sistema de reintentos
        last_error = None
        for attempt in range(self.MAX_RETRIES):
            try:
                response = self.client.chat.completions.create(**kwargs)
                content = response.choices[0].message.content

                # Limpiar thinking blocks
                if content:
                    content = re.sub(r"<think>[\s\S]*?</think>", "", content).strip()
                return content

            except Exception as e:
                last_error = e

                if not self._is_retryable_error(e):
                    # Error no retryable, lanzar inmediatamente
                    raise

                if attempt < self.MAX_RETRIES - 1:
                    delay = self._calculate_delay(attempt)
                    print(
                        f"⚠️ Error en LLM (intento {attempt + 1}/{self.MAX_RETRIES}): {type(e).__name__}. Reintentando en {delay}s..."
                    )
                    time.sleep(delay)
                else:
                    # Todos los reintentos agotados
                    print(f"❌ LLM falló después de {self.MAX_RETRIES} intentos")
                    raise

        # Código unreachable, pero por seguridad
        raise last_error if last_error else Exception("Error desconocido en LLM")

    def chat_json(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 4096,
    ) -> Dict[str, Any]:
        """
        发送聊天请求并返回JSON

        Args:
            messages: 消息列表
            temperature: 温度参数
            max_tokens: 最大token数

        Returns:
            解析后的JSON对象
        """
        response = self.chat(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )
        # 清理markdown代码块标记
        cleaned_response = response.strip()
        cleaned_response = re.sub(
            r"^```(?:json)?\s*\n?", "", cleaned_response, flags=re.IGNORECASE
        )
        cleaned_response = re.sub(r"\n?```\s*$", "", cleaned_response)
        cleaned_response = cleaned_response.strip()

        try:
            return json.loads(cleaned_response)
        except json.JSONDecodeError:
            raise ValueError(f"LLM返回的JSON格式无效: {cleaned_response}")
