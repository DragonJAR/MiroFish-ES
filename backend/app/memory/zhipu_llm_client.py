"""
Wrapper LLM client compatible con Graphiti para proveedores OpenAI-compatible
que no soportan structured output nativo (z.ai/ZhipuAI, etc.).

Problemas que resuelve:
1. z.ai con response_format=json_schema devuelve JSON envuelto en ```json ... ```
2. z.ai con max_tokens bajos agota tokens en reasoning y content queda vacío
3. z.ai no soporta beta.chat.completions.parse() (structured output nativo)

Hereda de OpenAIGenericClient y sobreescribe _generate_response para limpiar
la respuesta antes de json.loads().
"""

import json
import re
import logging
from typing import Any

from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient
from graphiti_core.llm_client.config import LLMConfig

logger = logging.getLogger(__name__)


def _strip_json_markdown(text: str) -> str:
    """
    Limpiar respuesta JSON que viene envuelta en markdown code blocks.

    z.ai y otros proveedores a veces retornan:
      ```json
      {"key": "value"}
      ```

    Esto causaría json.loads() → ValueError.
    """
    text = text.strip()

    # Patrón: ```json ... ``` o ``` ... ```
    match = re.match(r"^```(?:json)?\s*\n?(.*?)\n?\s*```$", text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Si empieza con ``` pero no cerró correctamente
    if text.startswith("```"):
        # Sacar la primera línea (```json) y último ``` si existe
        lines = text.split("\n")
        if lines:
            lines = [l for l in lines if not l.strip().startswith("```")]
        return "\n".join(lines).strip()

    return text


def _extract_json_from_response(text: str) -> dict[str, Any]:
    """
    Extraer JSON de una respuesta que puede contener markdown,
    texto extra, o reasoning envuelto.

    Estrategia:
    1. Intentar json.loads directo
    2. Intentar strippear markdown
    3. Buscar el primer { ... } o [ ... ] en el texto
    """
    text = text.strip()

    # 1. Directo
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        pass

    # 2. Strip markdown
    cleaned = _strip_json_markdown(text)
    if cleaned != text:
        try:
            return json.loads(cleaned)
        except (json.JSONDecodeError, ValueError):
            pass

    # 3. Buscar JSON object o array en el texto
    # Buscar el primer { que tenga un } correspondiente
    for start_char, end_char in [("{", "}"), ("[", "]")]:
        start_idx = text.find(start_char)
        if start_idx == -1:
            continue

        # Contar profundidad para encontrar el cierre correcto
        depth = 0
        for i in range(start_idx, len(text)):
            if text[i] == start_char:
                depth += 1
            elif text[i] == end_char:
                depth -= 1
                if depth == 0:
                    candidate = text[start_idx : i + 1]
                    try:
                        return json.loads(candidate)
                    except (json.JSONDecodeError, ValueError):
                        break

    # 4. Último recurso: reemplazar caracteres problemáticos y reintentar
    # Algunos modelos agregan trailing commas o comentarios
    cleaned = re.sub(r",\s*([}\]])", r"\1", cleaned)  # trailing commas
    cleaned = re.sub(r"//.*$", "", cleaned, flags=re.MULTILINE)  # // comments
    try:
        return json.loads(cleaned)
    except (json.JSONDecodeError, ValueError):
        pass

    # No se pudo parsear
    raise ValueError(
        f"No se pudo extraer JSON de la respuesta del LLM. "
        f"Respuesta (primeros 200 chars): {text[:200]}"
    )


# ── Mapeos de campos comunes que z.ai inventa ──
# z.ai tiende a generar nombres de campos descriptivos en vez de los que
# pide el JSON schema. Estos mapeos corrigen los más comunes.
_FIELD_ALIASES = {
    "entity_name": "name",
    "node_name": "name",
    "entity_type": "entity_type_id",
    "node_type": "entity_type_id",
    "description_text": "description",
    "entity_description": "description",
    "source_text": "source_description",
    "edge_type": "name",
    "relation_type": "name",
    "target": "target_node_uuid",
    "source": "source_node_uuid",
}


def _normalize_response(result: Any, response_model: type) -> Any:
    """
    Normalizar la respuesta del LLM para que coincida con el response_model.

    Maneja:
    1. Lista devuelta en vez de dict (envuelve en el campo correcto)
    2. Campos con nombres incorrectos (mapea aliases → nombres correctos)
    3. Listas de objetos con campos incorrectos
    """
    schema_fields = response_model.model_fields

    # Caso 1: Devolvió lista pero se espera dict
    if isinstance(result, list):
        # Buscar campo de tipo lista en el schema
        for field_name, field_info in schema_fields.items():
            type_str = str(field_info.annotation).lower()
            if "list" in type_str:
                result = {field_name: result}
                logger.debug(f"Normalized: list → {{'{field_name}': [...]}}")
                break
        else:
            # Si no hay campo de tipo lista, no podemos normalizar
            return result

    if not isinstance(result, dict):
        return result

    # Caso 2: Normalizar keys del dict principal
    result = _normalize_dict_keys(result, schema_fields)

    # Caso 3: Normalizar objects dentro de listas
    for field_name, field_info in schema_fields.items():
        if field_name not in result:
            continue
        type_str = str(field_info.annotation).lower()
        if "list" in type_str and isinstance(result[field_name], list):
            # Obtener el tipo de los elementos de la lista
            item_type = _get_list_item_type(field_info.annotation)
            if item_type and hasattr(item_type, "model_fields"):
                item_fields = item_type.model_fields
                result[field_name] = [
                    _normalize_dict_keys(item, item_fields)
                    if isinstance(item, dict)
                    else item
                    for item in result[field_name]
                ]

    return result


def _normalize_dict_keys(data: dict, expected_fields: dict) -> dict:
    """
    Normalizar las keys de un dict para que coincidan con los campos esperados.

    Estrategia:
    1. Si la key ya existe en expected_fields → mantener
    2. Si la key está en _FIELD_ALIASES → mapear
    3. Si el nombre esperado contiene substring de la key → mapear fuzzy
    """
    normalized = {}
    used_keys = set()

    for expected_key in expected_fields:
        if expected_key in data:
            normalized[expected_key] = data[expected_key]
            used_keys.add(expected_key)
            continue

        # Buscar alias directo
        found = False
        for data_key, alias_target in _FIELD_ALIASES.items():
            if data_key in data and alias_target == expected_key:
                normalized[expected_key] = data[data_key]
                used_keys.add(data_key)
                found = True
                break
        if found:
            continue

        # Buscar fuzzy: si el key de data contiene el nombre esperado (o viceversa)
        for data_key in data:
            if data_key in used_keys:
                continue
            expected_lower = expected_key.lower().replace("_", "")
            data_lower = data_key.lower().replace("_", "")
            if expected_lower in data_lower or data_lower in expected_lower:
                normalized[expected_key] = data[data_key]
                used_keys.add(data_key)
                break

    # Mantener campos extras que no se mapearon
    for key, value in data.items():
        if key not in used_keys:
            normalized[key] = value

    return normalized


def _get_list_item_type(annotation):
    """
    Extraer el tipo de los items de una List[SomeType] annotation.
    """
    import typing

    origin = typing.get_origin(annotation)
    if origin is list:
        args = typing.get_args(annotation)
        if args:
            return args[0]
    return None


class ZhipuAILLMClient(OpenAIGenericClient):
    """
    OpenAIGenericClient adaptado para ZhipuAI (z.ai).

    z.ai tiene particularidades que rompen Graphiti:
    - Con response_format=json_schema, devuelve JSON envuelto en ```json ... ```
    - Con max_tokens bajos, agota en reasoning y content queda vacío
    - No soporta structured output nativo (beta.chat.completions.parse)

    Este wrapper limpia la respuesta y asegura que json.loads() funcione.
    """

    async def _generate_response(
        self,
        messages: list,
        response_model=None,
        max_tokens: int = 16384,
        model_size=None,
    ) -> dict[str, Any]:
        """
        Generar respuesta LLM con limpieza de JSON.

        Sobreescribe OpenAIGenericClient._generate_response para:
        1. Limpiar respuestas envueltas en markdown
        2. Extraer JSON de respuestas con texto extra
        """
        import openai

        openai_messages = []
        for m in messages:
            m.content = self._clean_input(m.content)
            if m.role == "user":
                openai_messages.append({"role": "user", "content": m.content})
            elif m.role == "system":
                openai_messages.append({"role": "system", "content": m.content})

        try:
            # Construir response_format
            response_format = {"type": "json_object"}
            if response_model is not None:
                schema_name = getattr(response_model, "__name__", "structured_response")
                json_schema = response_model.model_json_schema()
                response_format = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": schema_name,
                        "schema": json_schema,
                    },
                }

            response = await self.client.chat.completions.create(
                model=self.model or "gpt-4.1-mini",
                messages=openai_messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                response_format=response_format,
            )

            raw_content = response.choices[0].message.content or ""

            # Si el content está vacío pero hay reasoning_content,
            # puede que el modelo haya gastado todo en reasoning
            if not raw_content.strip():
                reasoning = response.choices[0].message.reasoning_content or ""
                if reasoning:
                    logger.warning(
                        f"LLM devolvió content vacío con {len(reasoning)} chars de reasoning. "
                        f"Probablemente max_tokens insuficiente. "
                        f"max_tokens={self.max_tokens}"
                    )
                raise ValueError(
                    "El LLM devolvió una respuesta vacía. "
                    "Posiblemente max_tokens insuficiente para reasoning + output."
                )

            # Limpiar y extraer JSON (maneja ```json ... ``` wrappers)
            result = _extract_json_from_response(raw_content)

            # ── Normalización para z.ai ──
            # z.ai tiene 3 problemas principales:
            # 1. A veces devuelve una lista cuando Graphiti espera un dict
            # 2. Inventa nombres de campos (ej: "entity_name" en vez de "name")
            # 3. Ignora el JSON schema y devuelve lo que quiere
            #
            # Normalizamos basándonos en el response_model cuando existe.
            if response_model is not None:
                result = _normalize_response(result, response_model)

            return result

        except openai.RateLimitError as e:
            raise
        except ValueError:
            # Ya logueado en _extract_json_from_response
            raise
        except Exception as e:
            logger.error(f"Error en LLM response: {e}")
            raise
