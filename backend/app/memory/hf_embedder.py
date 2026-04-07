"""
Embedder local usando HuggingFace sentence-transformers.

Implementa la interfaz EmbedderClient de Graphiti para que funcione
sin depender de servicios externos de embeddings (OpenAI, etc.).

Modelo default: all-MiniLM-L6-v2 (384 dims, ~80MB, rápido y preciso).
Configurable vía HF_EMBEDDING_MODEL y HF_EMBEDDING_DEVICE en .env.

Dispositivo (auto-detectado):
  - CUDA disponible → cuda (Linux/Windows con NVIDIA GPU)
  - Sin GPU → cpu (funciona en todas las plataformas)
  Override manual con HF_EMBEDDING_DEVICE=cpu|cuda|mps
"""

import os
import asyncio
import logging
from typing import List

from graphiti_core.embedder.client import EmbedderClient

logger = logging.getLogger(__name__)

# Thread pool para ejecutar embeddings sincrónicos sin bloquear el event loop
_executor = None


def _get_executor():
    """Lazy init del thread pool executor."""
    global _executor
    if _executor is None:
        from concurrent.futures import ThreadPoolExecutor

        _executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="hf-embedder")
    return _executor


class HuggingFaceEmbedder(EmbedderClient):
    """
    Embedder local basado en sentence-transformers.

    Compatible con la interfaz EmbedderClient de graphiti_core.
    Los métodos son async pero delegan a sentence-transformers (sync)
    via thread pool para no bloquear el event loop.
    """

    # Modelos recomendados (nombre HuggingFace → dimensiones)
    MODELS = {
        "all-MiniLM-L6-v2": 384,  # Default: rápido, ligero, bueno general
        "all-mpnet-base-v2": 768,  # Más preciso, más pesado (~420MB)
        "paraphrase-multilingual-MiniLM-L12-v2": 384,  # Multilingüe (ES, EN, etc.)
        "bge-small-en-v1.5": 384,  # BGE, optimizado para retrieval
        "bge-base-en-v1.5": 768,  # BGE, mejor calidad
    }

    @staticmethod
    def _detect_device() -> str:
        """
        Detectar mejor dispositivo disponible (cross-platform).

        Prioridad:
        1. HF_EMBEDDING_DEVICE (override manual del usuario)
        2. CUDA (GPU NVIDIA en Linux/Windows)
        3. CPU (funciona en todas las plataformas)

        MPS (Apple Silicon) no se usa por defecto porque Graphiti corre los
        embeddings en thread pool workers y MPS tiene problemas con meta
        tensors en contextos multi-hilo. El usuario puede forzarlo con
        HF_EMBEDDING_DEVICE=mps si lo desea.
        """
        # 1. Override manual
        env_device = os.environ.get("HF_EMBEDDING_DEVICE", "").strip().lower()
        if env_device in ("cpu", "cuda", "mps"):
            return env_device

        # 2. CUDA (GPU NVIDIA)
        try:
            import torch

            if torch.cuda.is_available():
                device_name = torch.cuda.get_device_name(0)
                logger.info(f"CUDA detectada: {device_name}")
                return "cuda"
        except Exception:
            pass  # torch no disponible o sin CUDA

        # 3. CPU (fallback universal)
        logger.info("Sin GPU detectada, usando CPU para embeddings")
        return "cpu"

    def __init__(self, model_name: str = None, embedding_dim: int = None):
        model_name = model_name or os.environ.get(
            "HF_EMBEDDING_MODEL", "all-MiniLM-L6-v2"
        )

        self.model_name = model_name
        self.embedding_dim = embedding_dim or self.MODELS.get(
            model_name,
            384,  # default dim
        )
        self.device = self._detect_device()

        # Carga eager en el hilo principal — NO lazy loading.
        # PyTorch falla con "Cannot copy out of meta tensor" si el modelo
        # se inicializa desde un thread pool worker (especialmente con MPS).
        logger.info(
            f"HuggingFaceEmbedder: {model_name} (dim={self.embedding_dim}, device={self.device})"
        )

        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(self.model_name, device=self.device)
        logger.info(f"Modelo {self.model_name} cargado en {self.device}")

    @property
    def model(self):
        """Modelo ya cargado en __init__ (eager loading)."""
        return self._model

    async def create(self, input_data) -> List[float]:
        """
        Crear embedding para un texto o lista de textos.

        Si es un solo string → retorna un solo vector.
        Si es una lista → retorna una lista de vectores.

        (Implementa la interfaz async de EmbedderClient)
        """
        loop = asyncio.get_event_loop()
        executor = _get_executor()

        if isinstance(input_data, str):
            vector = await loop.run_in_executor(
                executor, self._encode_single, input_data
            )
            return vector
        elif isinstance(input_data, list) and all(
            isinstance(x, str) for x in input_data
        ):
            # EmbedderClient.create() contract: MUST return List[float], not List[List[float]].
            # When input is a list, encode ONLY the first element to get a single vector.
            # This matches the expected interface where a list input returns one embedding.
            vector = await loop.run_in_executor(
                executor, self._encode_single, input_data[0]
            )
            return vector
        else:
            # Token IDs u otros formatos — no soportados, usar default
            logger.warning(
                f"Input type no soportado en HF embedder: {type(input_data)}. "
                f"Retornando vector de ceros."
            )
            return [0.0] * self.embedding_dim

    async def create_batch(self, input_data_list: List[str]) -> List[List[float]]:
        """Crear embeddings para una lista de textos (batch)."""
        loop = asyncio.get_event_loop()
        executor = _get_executor()
        return await loop.run_in_executor(executor, self._encode_batch, input_data_list)

    # ── Métodos sincrónicos (corren en thread pool) ──

    def _encode_single(self, text: str) -> List[float]:
        """Encodear un solo texto (sync, correr en thread pool)."""
        embedding = self.model.encode(text, normalize_embeddings=True)
        return embedding.tolist()

    def _encode_batch(self, texts: List[str]) -> List[List[float]]:
        """Encodear batch de textos (sync, correr en thread pool)."""
        embeddings = self.model.encode(texts, normalize_embeddings=True)
        return [e.tolist() for e in embeddings]
