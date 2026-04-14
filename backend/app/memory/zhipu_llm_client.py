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
import os
import re
import logging
import typing
from typing import Any

from pydantic import BaseModel

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
    cleaned = text

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

    # 4. Reemplazar caracteres problemáticos y reintentar
    # Algunos modelos agregan trailing commas o comentarios
    cleaned = re.sub(r",\s*([}\]])", r"\1", cleaned)  # trailing commas
    cleaned = re.sub(r"//.*$", "", cleaned, flags=re.MULTILINE)  # // comments
    try:
        return json.loads(cleaned)
    except (json.JSONDecodeError, ValueError):
        pass

    # 5. json_repair como último recurso — repara JSONs con errores de sintaxis
    try:
        import json_repair

        return json_repair.loads(text)
    except Exception:
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
    # ── Entity name (el LLM usa varios nombres para este campo) ──
    "entity": "name",
    "entity_name": "name",
    "node_name": "name",
    "node": "name",
    "label": "name",
    "title": "name",
    # ── Entity type ──
    "entity_type": "entity_type_id",
    "node_type": "entity_type_id",
    "type": "entity_type_id",
    "category": "entity_type_id",
    # ── Description ──
    "description_text": "description",
    "entity_description": "description",
    "desc": "description",
    # ── Source ──
    "source_text": "source_description",
    # ── Edges/Relations ──
    "edge_type": "name",
    "relation_type": "name",
    "relation": "name",
    "target": "target_node_uuid",
    "source": "source_node_uuid",
    # ── Edge deduplication fields (GLM often uses wrong names) ──
    "contradicting_facts": "contradicted_facts",
    "conflicting_facts": "contradicted_facts",
    "conflicts": "contradicted_facts",
    "duplicate": "duplicate_facts",
    "duplicates": "duplicate_facts",
    "duplicate_ids": "duplicate_facts",
    # ── Entity/Edge wrappers (GLM often misses these) ──
    "entities": "extracted_entities",
    "nodes": "extracted_entities",
    "items": "extracted_entities",
    "relations": "edges",
    "relationships": "edges",
    "links": "edges",
    "resolutions": "entity_resolutions",
    "duplicates_list": "entity_resolutions",
}


def _detect_and_wrap_container(result: Any, response_model: type) -> Any:
    """
    Detectar si la respuesta del LLM está FALTANTE el wrapper key del container.

    GLM-4.5 a veces devuelve los campos de la entidad directamente SIN el
    wrapper key. Por ejemplo:

    LLM devuelve:   {"entity_name": "Mexico", "entity_type": 5, "description": "..."}
    Graphiti espera: {"extracted_entities": [{"name": "Mexico", "entity_type_id": 5}]}

    Este método detecta ese caso y envuelve la respuesta en el key correcto.

    Estrategias (en orden de confianza):
    A) Response ES una lista → wrap en {missing_key: result}
    B) Response es dict con >50% overlap con inner model → wrap como lista de 1 item
    C) Algún key del response contiene una lista que coincide con inner model → usar ese key
    """
    if not isinstance(result, (dict, list)):
        return result

    schema_fields = response_model.model_fields

    if isinstance(result, dict):
        result_keys = set(result.keys())
    else:
        result_keys = set()

    # Encontrar keys faltantes (los que el schema requiere pero no están en result)
    missing_keys = [
        f
        for f, info in schema_fields.items()
        if info.is_required() and f not in result_keys
    ]

    # Solo proceder si hay EXACTAMENTE un missing key que es list[SomeModel]
    if len(missing_keys) != 1:
        return result

    missing_key = missing_keys[0]

    if isinstance(result, list):
        logger.debug(f"Strategy A: Wrapping list result in '{missing_key}'")
        return {missing_key: result}

    if not isinstance(result, dict):
        return result

    field_annotation = schema_fields[missing_key].annotation

    # Verificar si el campo faltante es list[SomeModel]
    if not _is_list_of_models(field_annotation):
        return result

    inner_model = _get_list_item_type(field_annotation)
    if inner_model is None:
        return result

    inner_fields = (
        inner_model.model_fields if hasattr(inner_model, "model_fields") else {}
    )

    # ── Strategy A: Response ES una lista ──
    if isinstance(result, list):
        logger.debug(f"Strategy A: Wrapping list result in '{missing_key}'")
        return {missing_key: result}

    # ── Strategy B: Dict cuyas keys overlap con el inner model (>50%) ──
    if isinstance(result, dict):
        # Aplicar aliases de _FIELD_ALIASES para calcular overlap correctamente
        # El LLM devuelve "entity_type" pero el schema espera "entity_type_id"
        aliased_keys = set()
        for k in result.keys():
            aliased_key = _FIELD_ALIASES.get(k, k)
            aliased_keys.add(aliased_key.lower().replace("_", ""))

        inner_fields_lower = set(
            k.lower().replace("_", "") for k in inner_fields.keys()
        )

        if inner_fields_lower:
            overlap = len(inner_fields_lower & aliased_keys) / len(inner_fields_lower)
            if overlap > 0.5:
                logger.debug(
                    f"Strategy B: Wrapping dict (%.0f%% overlap) in '{missing_key}'"
                    % (overlap * 100)
                )
                return {missing_key: [result]}

        # ── Strategy C: Algún value es una lista que matchea el inner model ──
        for key, value in result.items():
            if not isinstance(value, list):
                continue

            # Verificar si los items de la lista son dicts con overlap
            # (aplicar aliases para que entity_type → entity_type_id)
            matching_items = 0
            for item in value:
                if not isinstance(item, dict):
                    continue
                # Aplicar aliases para calcular overlap correctamente
                item_aliased_keys = set()
                for k in item.keys():
                    aliased_key = _FIELD_ALIASES.get(k, k)
                    item_aliased_keys.add(aliased_key.lower().replace("_", ""))

                if item_aliased_keys and inner_fields_lower:
                    item_overlap = len(inner_fields_lower & item_aliased_keys) / len(
                        inner_fields_lower
                    )
                    if item_overlap > 0.5:
                        matching_items += 1

            # Si TODOS o la mayoría de los items matchean, usar este key
            if matching_items >= len(value) * 0.5 and matching_items > 0:
                logger.debug(f"Strategy C: Using list in '{key}' as '{missing_key}'")
                wrapped = {missing_key: value}
                # Remover el key viejo del result (si aún tiene el list)
                # pero solo si ya lo movimos
                return wrapped

        # ── Strategy D: Dict de str→str que parece {name: field_value} ──
        # GLM-4.5 a veces devuelve {"Latinoamérica": "Región con..."} cuando
        # espera {"summaries": [{"name": "Latinoamérica", "summary": "Región con..."}]}
        # Detectar: todos los keys y values son strings, y el inner model
        # tiene un campo "name" y otro campo string principal.
        all_str_keys = all(isinstance(k, str) for k in result.keys())
        all_str_values = all(isinstance(v, str) for v in result.values())

        if all_str_keys and all_str_values and len(result) > 0 and inner_fields_lower:
            # Buscar el campo que NO es "name" en el inner model
            inner_field_names = set(inner_fields.keys())
            non_name_fields = inner_field_names - {"name", "summary"}

            if "name" in inner_field_names:
                # Determinar qué campo representan los values
                # Si hay un campo "summary", description", o similar, usar ese
                value_field = None
                for candidate in ["summary", "description", "text", "content", "value"]:
                    if candidate in inner_field_names:
                        value_field = candidate
                        break
                if value_field is None and non_name_fields:
                    value_field = next(iter(non_name_fields))

                if value_field is not None:
                    wrapped_list = [
                        {"name": k, value_field: v} for k, v in result.items()
                    ]
                    logger.debug(
                        f"Strategy D: Dict of name→{value_field} wrapped "
                        f"({len(wrapped_list)} items) in '{missing_key}'"
                    )
                    return {missing_key: wrapped_list}

    return result


def _is_list_of_models(annotation) -> bool:
    """
    Verificar si una annotation es list[SomeModel] donde SomeModel es BaseModel.
    """
    origin = typing.get_origin(annotation)
    args = typing.get_args(annotation)
    return (
        origin is list
        and len(args) > 0
        and isinstance(args[0], type)
        and issubclass(args[0], BaseModel)
    )


def _normalize_response(result: Any, response_model: type) -> Any:
    """
    Normalizar la respuesta del LLM para que coincida con el response_model.

    Orden de operaciones:
    1. _detect_and_wrap_container() — detecta si falta el wrapper key del container
    2. Normalizar keys del dict principal (aliases, fuzzy matching)
    3. Normalizar objects dentro de listas
    4. Campos faltantes requeridos → búsqueda profunda recursiva
    5. Type coercion (str → int)
    """
    # ── Paso 0: Detectar si falta el wrapper key del container ──
    result = _detect_and_wrap_container(result, response_model)

    schema_fields = response_model.model_fields

    # Caso 1: Devolvió lista pero se espera dict
    if isinstance(result, list):
        for field_name, field_info in schema_fields.items():
            type_str = str(field_info.annotation).lower()
            if "list" in type_str:
                result = {field_name: result}
                logger.debug(f"Normalized: list → {{'{field_name}': [...]}}")
                break
        else:
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
            item_type = _get_list_item_type(field_info.annotation)
            if item_type and hasattr(item_type, "model_fields"):
                item_fields = item_type.model_fields
                normalized_items = []
                for item in result[field_name]:
                    if isinstance(item, dict):
                        item = _normalize_dict_keys(item, item_fields)
                        # Auto-fill missing or None required str fields in list items
                        for if_name, if_info in item_fields.items():
                            if if_info.is_required():
                                if_ann = if_info.annotation
                                needs_fill = (if_name not in item) or (
                                    item.get(if_name) is None
                                )
                                if needs_fill and if_ann is str:
                                    logger.warning(
                                        f"Auto-filling required str field '{if_name}' in "
                                        f"{item_type.__name__} with '' (was: {item.get(if_name)})"
                                    )
                                    item[if_name] = ""
                        normalized_items.append(item)
                    else:
                        # Skip non-dict items (GLM sometimes puts stray values in lists)
                        logger.warning(
                            f"Skipping non-dict item in '{field_name}' list: "
                            f"{type(item).__name__} = {str(item)[:100]}"
                        )
                result[field_name] = normalized_items

    # Caso 4: Si faltan campos requeridos, buscar recursivamente en todo el dict
    required_missing = [
        f for f, info in schema_fields.items() if info.is_required() and f not in result
    ]
    if required_missing:
        result = _deep_search_missing_keys(result, required_missing)

    # Bug 2 fix: type coercion for int fields (e.g. entity_type_id "5" → 5)
    for field_name, field_info in schema_fields.items():
        if field_name not in result:
            continue
        # Only process fields typed as plain int (not Optional[int])
        annotation = field_info.annotation
        if _is_plain_int(annotation) and isinstance(result[field_name], str):
            if result[field_name].isdigit():
                result[field_name] = int(result[field_name])
                logger.debug(f"Coerced {field_name}: str → int")

    # Step 6: Auto-fill missing required list fields with empty defaults
    # Covers list[int], list[str], list[BaseModel] — all safe to default to []
    for field_name, field_info in schema_fields.items():
        if field_name not in result and field_info.is_required():
            annotation = field_info.annotation
            if _is_list_of_primitives(annotation) or _is_list_of_models(annotation):
                logger.warning(
                    f"Auto-filling missing required field '{field_name}' with [] "
                    f"in {response_model.__name__}. GLM-4.5 likely omitted it."
                )
                result[field_name] = []

    return result


def _deep_search_missing_keys(data: dict, missing_keys: list[str]) -> dict:
    """
    Buscar recursivamente en todo el dict (incluidos dicts anidados y listas)
    para encontrar los keys faltantes. Si un key se encuentra en un nivel más profundo,
    lo mueve al nivel raíz.
    """
    for key in list(missing_keys):
        found = _find_key_anywhere(data, key)
        # Bug 1 fix: empty string "", empty list [], empty dict {} also fail Pydantic validation
        if found is not None and found != "" and found != [] and found != {}:
            data[key] = found
    return data


def _find_key_anywhere(obj: Any, target_key: str) -> Any:
    """Buscar un key recursivamente en cualquier nivel del dict/lista."""
    if isinstance(obj, dict):
        if target_key in obj:
            return obj[target_key]
        for key, value in obj.items():
            if isinstance(value, (dict, list)):
                found = _find_key_anywhere(value, target_key)
                if found is not None:
                    return found
    elif isinstance(obj, list):
        for item in obj:
            if isinstance(item, (dict, list)):
                found = _find_key_anywhere(item, target_key)
                if found is not None:
                    return found
    return None


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


def _is_plain_int(annotation) -> bool:
    """
    Check if annotation is a plain int (not Optional[int], not Union[..., int], etc.).
    """
    import typing

    # None means no generic origin (plain type)
    if typing.get_origin(annotation) is None:
        # Check if it's exactly int
        return annotation is int
    return False


def _is_list_of_primitives(annotation) -> bool:
    """Check if annotation is list[int] or list[str] or list[float] or list[bool]."""
    origin = typing.get_origin(annotation)
    if origin is not list:
        return False
    args = typing.get_args(annotation)
    if len(args) != 1:
        return False
    return args[0] in (int, str, float, bool)


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
        3. Fallback a lista vacía cuando GLM devuelve texto plano (Bug A fix)
        """
        import openai

        openai_messages = []
        for m in messages:
            m.content = self._clean_input(m.content)
            if m.role == "user":
                openai_messages.append({"role": "user", "content": m.content})
            elif m.role == "system":
                openai_messages.append({"role": "system", "content": m.content})

        # Bug A fix: Retry logic for GLM plain text responses
        max_retries = 1
        for attempt in range(max_retries + 1):
            try:
                # Construir response_format
                response_format = {"type": "json_object"}
                if response_model is not None:
                    schema_name = getattr(
                        response_model, "__name__", "structured_response"
                    )
                    json_schema = response_model.model_json_schema()
                    response_format = {
                        "type": "json_schema",
                        "json_schema": {
                            "name": schema_name,
                            "schema": json_schema,
                        },
                    }

                # Retry with stronger system prompt on second attempt
                if attempt > 0:
                    logger.warning(
                        f"Retrying LLM call (attempt {attempt + 1}/{max_retries + 1}) with stronger JSON enforcement"
                    )
                    if openai_messages and openai_messages[0].get("role") == "system":
                        openai_messages[0]["content"] += (
                            "\n\nIMPORTANT: You MUST respond with valid JSON only. No conversational text, no explanations, no other content. Your response MUST be parseable by JSON.parse()."
                        )

                response = await self.client.chat.completions.create(
                    model=self.model or os.environ.get("LLM_MODEL_NAME", "glm-4.5"),
                    messages=openai_messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    response_format=response_format,
                )

                if not response.choices:
                    raise ValueError("API returned empty choices array")
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

                # Debug temporal: ver qué devuelve el LLM antes de normalizar
                logger.debug(f"LLM raw response (first 300 chars): {str(result)[:300]}")

                # ── Normalización para z.ai ──
                if response_model is not None:
                    model_name = getattr(response_model, "__name__", "?")
                    raw_keys = (
                        list(result.keys())
                        if isinstance(result, dict)
                        else type(result).__name__
                    )
                    result = _normalize_response(result, response_model)
                    new_keys = list(result.keys()) if isinstance(result, dict) else []
                    if raw_keys != new_keys:
                        logger.debug(
                            f"Normalized {model_name}: {raw_keys} → {new_keys}"
                        )
                    else:
                        logger.debug(
                            f"No normalization needed for {model_name}: {raw_keys}"
                        )

                # Final safety: if result is not a dict but response_model expects one,
                # raise an error instead of silently returning auto-filled garbage
                if response_model is not None and not isinstance(result, dict):
                    model_name = getattr(response_model, "__name__", "?")
                    raise ValueError(
                        f"LLM returned {type(result).__name__} instead of dict for {model_name}. "
                        f"Cannot auto-fill from invalid type. Raw: {str(result)[:200]}"
                    )

                return result

            except ValueError as e:
                # Bug A fix: Handle GLM plain text responses
                if response_model is not None and "No se pudo extraer JSON" in str(e):
                    schema_fields = response_model.model_fields
                    # Find if there's exactly one required field of type list[Model]
                    list_fields = [
                        f
                        for f, info in schema_fields.items()
                        if info.is_required() and _is_list_of_models(info.annotation)
                    ]

                    if len(list_fields) == 1:
                        field_name = list_fields[0]
                        model_name = getattr(response_model, "__name__", "?")
                        logger.warning(
                            f"GLM devolvió texto plano en lugar de JSON para {model_name}. "
                            f"Retornando lista vacía en '{field_name}' como fallback. "
                            f"Raw response: {raw_content[:200]}"
                        )
                        return {field_name: []}
                raise
            except openai.RateLimitError as e:
                raise
            except Exception as e:
                logger.error(f"Error en LLM response: {e}")
                raise
