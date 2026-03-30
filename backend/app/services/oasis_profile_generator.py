"""
OASIS Agent Profile Generator
Converts entities from Zep graph to Agent Profile format required by OASIS simulation platform

Optimization improvements:
1. Call Zep retrieval to enrich node information
2. Optimize prompts to generate very detailed personas
3. Distinguish individual entities from abstract group entities
"""

import json
import random
import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime

from openai import OpenAI

from ..config import Config
from ..utils.logger import get_logger
from ..memory import get_memory_backend, EntityNode

# Intentar importar el sistema de prompts i18n
try:
    from ..prompts import load_prompt as _load_prompt

    _PROMPTS_AVAILABLE = True
except ImportError:
    _PROMPTS_AVAILABLE = False
    _load_prompt = None

logger = get_logger("mirofish.oasis_profile")


@dataclass
class OasisAgentProfile:
    """OASIS Agent Profile data structure"""

    # Common fields
    user_id: int
    user_name: str
    name: str
    bio: str
    persona: str

    # Optional fields - Reddit style
    karma: int = 1000

    # Optional fields - Twitter style
    friend_count: int = 100
    follower_count: int = 150
    statuses_count: int = 500

    # Extra persona info
    age: Optional[int] = None
    gender: Optional[str] = None
    mbti: Optional[str] = None
    country: Optional[str] = None
    profession: Optional[str] = None
    interested_topics: List[str] = field(default_factory=list)

    # Source entity info
    source_entity_uuid: Optional[str] = None
    source_entity_type: Optional[str] = None

    created_at: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d"))

    def to_reddit_format(self) -> Dict[str, Any]:
        """Convert to Reddit platform format"""
        profile = {
            "user_id": self.user_id,
            "username": self.user_name,  # OASIS library requires field name username (no underscore)
            "name": self.name,
            "bio": self.bio,
            "persona": self.persona,
            "karma": self.karma,
            "created_at": self.created_at,
        }

        # Agregar información extra del perfil (si existe)
        if self.age:
            profile["age"] = self.age
        if self.gender:
            profile["gender"] = self.gender
        if self.mbti:
            profile["mbti"] = self.mbti
        if self.country:
            profile["country"] = self.country
        if self.profession:
            profile["profession"] = self.profession
        if self.interested_topics:
            profile["interested_topics"] = self.interested_topics

        return profile

    def to_twitter_format(self) -> Dict[str, Any]:
        """Convert to Twitter platform format"""
        profile = {
            "user_id": self.user_id,
            "username": self.user_name,  # OASIS library requires field name username (no underscore)
            "name": self.name,
            "bio": self.bio,
            "persona": self.persona,
            "friend_count": self.friend_count,
            "follower_count": self.follower_count,
            "statuses_count": self.statuses_count,
            "created_at": self.created_at,
        }

        # Agregar información extra del perfil
        if self.age:
            profile["age"] = self.age
        if self.gender:
            profile["gender"] = self.gender
        if self.mbti:
            profile["mbti"] = self.mbti
        if self.country:
            profile["country"] = self.country
        if self.profession:
            profile["profession"] = self.profession
        if self.interested_topics:
            profile["interested_topics"] = self.interested_topics

        return profile

    def to_dict(self) -> Dict[str, Any]:
        """Convert to complete dictionary format"""
        return {
            "user_id": self.user_id,
            "user_name": self.user_name,
            "name": self.name,
            "bio": self.bio,
            "persona": self.persona,
            "karma": self.karma,
            "friend_count": self.friend_count,
            "follower_count": self.follower_count,
            "statuses_count": self.statuses_count,
            "age": self.age,
            "gender": self.gender,
            "mbti": self.mbti,
            "country": self.country,
            "profession": self.profession,
            "interested_topics": self.interested_topics,
            "source_entity_uuid": self.source_entity_uuid,
            "source_entity_type": self.source_entity_type,
            "created_at": self.created_at,
        }


class OasisProfileGenerator:
    """
    OASIS Profile Generator

    Converts entities from Zep graph to Agent Profile required by OASIS simulation

    Optimization features:
    1. Call Zep graph retrieval to get richer context
    2. Generate very detailed personas (including basic info, career, personality traits, social media behavior, etc.)
    3. Distinguish individual entities from abstract group entities
    """

    # MBTI type list
    MBTI_TYPES = [
        "INTJ",
        "INTP",
        "ENTJ",
        "ENTP",
        "INFJ",
        "INFP",
        "ENFJ",
        "ENFP",
        "ISTJ",
        "ISFJ",
        "ESTJ",
        "ESFJ",
        "ISTP",
        "ISFP",
        "ESTP",
        "ESFP",
    ]

    # Common countries list
    COUNTRIES = [
        "China",
        "US",
        "UK",
        "Japan",
        "Germany",
        "France",
        "Canada",
        "Australia",
        "Brazil",
        "India",
        "South Korea",
    ]

    # Individual type entities (need to generate specific persona)
    INDIVIDUAL_ENTITY_TYPES = [
        "student",
        "alumni",
        "professor",
        "person",
        "publicfigure",
        "expert",
        "faculty",
        "official",
        "journalist",
        "activist",
    ]

    # Group/institution type entities (need to generate group representative persona)
    GROUP_ENTITY_TYPES = [
        "university",
        "governmentagency",
        "organization",
        "ngo",
        "mediaoutlet",
        "company",
        "institution",
        "group",
        "community",
    ]

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model_name: Optional[str] = None,
        backend=None,
        graph_id: Optional[str] = None,
    ):
        self.api_key = api_key or Config.LLM_API_KEY
        self.base_url = base_url or Config.LLM_BASE_URL
        self.model_name = model_name or Config.LLM_MODEL_NAME

        if not self.api_key:
            raise ValueError("LLM_API_KEY not configured")

        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

        # Memory backend para retrieval
        self.backend = backend or get_memory_backend()
        self.graph_id = graph_id

    def generate_profile_from_entity(
        self, entity: EntityNode, user_id: int, use_llm: bool = True
    ) -> OasisAgentProfile:
        """
        Generate OASIS Agent Profile from Zep entity

        Args:
            entity: Zep entity node
            user_id: User ID (for OASIS)
            use_llm: Whether to use LLM to generate detailed persona

        Returns:
            OasisAgentProfile
        """
        entity_type = entity.get_entity_type() or "Entity"

        # Información básica
        name = entity.name
        user_name = self._generate_username(name)

        # Construir información de contexto
        context = self._build_entity_context(entity)

        if use_llm:
            # Usar LLM para generar perfil detallado
            profile_data = self._generate_profile_with_llm(
                entity_name=name,
                entity_type=entity_type,
                entity_summary=entity.summary,
                entity_attributes=entity.attributes,
                context=context,
            )
        else:
            # Usar reglas para generar perfil básico
            profile_data = self._generate_profile_rule_based(
                entity_name=name,
                entity_type=entity_type,
                entity_summary=entity.summary,
                entity_attributes=entity.attributes,
            )

        return OasisAgentProfile(
            user_id=user_id,
            user_name=user_name,
            name=name,
            bio=profile_data.get("bio", f"{entity_type}: {name}"),
            persona=profile_data.get(
                "persona", entity.summary or f"A {entity_type} named {name}."
            ),
            karma=profile_data.get("karma", random.randint(500, 5000)),
            friend_count=profile_data.get("friend_count", random.randint(50, 500)),
            follower_count=profile_data.get(
                "follower_count", random.randint(100, 1000)
            ),
            statuses_count=profile_data.get(
                "statuses_count", random.randint(100, 2000)
            ),
            age=profile_data.get("age"),
            gender=profile_data.get("gender"),
            mbti=profile_data.get("mbti"),
            country=profile_data.get("country"),
            profession=profile_data.get("profession"),
            interested_topics=profile_data.get("interested_topics", []),
            source_entity_uuid=entity.uuid,
            source_entity_type=entity_type,
        )

    def _generate_username(self, name: str) -> str:
        """Generate username"""
        # Eliminar caracteres especiales, convertir a minúsculas
        username = name.lower().replace(" ", "_")
        username = "".join(c for c in username if c.isalnum() or c == "_")

        # Add random suffix to avoid duplicates
        suffix = random.randint(100, 999)
        return f"{username}_{suffix}"

    def _search_zep_for_entity(self, entity: EntityNode) -> Dict[str, Any]:
        """
        Usar función de búsqueda híbrida del grafo Zep para obtener información rica relacionada con la entidad

        Zep no tiene interfaz de búsqueda híbrida integrada, se deben buscar edges y nodes por separado y luego fusionar los resultados.
        Usar solicitudes paralelas para buscar simultáneamente, mejorar eficiencia.

        Args:
            entity: Objeto nodo de entidad

        Returns:
            Diccionario que contiene facts, node_summaries y context
        """
        import concurrent.futures

        results = {"facts": [], "node_summaries": [], "context": ""}

        # Debe tener graph_id para buscar
        if not self.graph_id:
            logger.debug(f"Saltar búsqueda Zep: graph_id no configurado")
            return results

        entity_name = entity.name

        comprehensive_query = f"Toda la información, actividades, eventos, relaciones y contexto sobre {entity_name}"

        def search_edges():
            """Buscar edges (hechos/relaciones) - con mecanismo de reintento"""
            max_retries = 3
            last_exception = None
            delay = 2.0

            for attempt in range(max_retries):
                try:
                    search_result = self.backend.search(
                        query=comprehensive_query,
                        graph_id=self.graph_id,
                        limit=30,
                        mode="quick",
                    )
                    return search_result
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        logger.debug(
                            f"Zep edge search attempt {attempt + 1}  failed: {str(e)[:80]}, retrying..."
                        )
                        time.sleep(delay)
                        delay *= 2
                    else:
                        logger.debug(
                            f"Zep edge search after {max_retries}  attempts still failed: {e}"
                        )
            return None

        def search_nodes():
            """Buscar nodos (resumen de entidad) - con mecanismo de reintento"""
            max_retries = 3
            last_exception = None
            delay = 2.0

            for attempt in range(max_retries):
                try:
                    search_result = self.backend.search(
                        query=comprehensive_query,
                        graph_id=self.graph_id,
                        limit=20,
                        mode="quick",
                    )
                    return search_result
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        logger.debug(
                            f"Zep node search attempt {attempt + 1}  failed: {str(e)[:80]}, retrying..."
                        )
                        time.sleep(delay)
                        delay *= 2
                    else:
                        logger.debug(
                            f"Zepnodobúsquedaen {max_retries}  attempts still failed: {e}"
                        )
            return None

        try:
            # Ejecutar búsqueda de edges y nodes en paralelo
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                edge_future = executor.submit(search_edges)
                node_future = executor.submit(search_nodes)

                # Obtener resultados
                edge_result = edge_future.result(timeout=30)
                node_result = node_future.result(timeout=30)

            # Procesar resultados de búsqueda de edges
            all_facts = set()
            if edge_result and hasattr(edge_result, "facts"):
                all_facts = set(edge_result.facts[:20])
            results["facts"] = list(all_facts)

            # Procesar resultados de búsqueda de nodes
            all_summaries = set()
            if node_result and hasattr(node_result, "nodes"):
                for node in node_result.nodes:
                    if "summary" in node and node["summary"]:
                        all_summaries.add(node["summary"])
                    if "name" in node and node["name"] and node["name"] != entity_name:
                        all_summaries.add(f"Entidad relacionada: {node['name']}")
            results["node_summaries"] = list(all_summaries)

            # Construir contexto  (completo)
            context_parts = []
            if results["facts"]:
                context_parts.append(
                    "Información de hechos:\n"
                    + "\n".join(f"- {f}" for f in results["facts"][:20])
                )
            if results["node_summaries"]:
                context_parts.append(
                    "Entidad relacionada:\n"
                    + "\n".join(f"- {s}" for s in results["node_summaries"][:10])
                )
            results["context"] = "\n\n".join(context_parts)

            logger.info(
                f"Zep hybrid search completed: {entity_name}, obtained {len(results['facts'])}  facts, {len(results['node_summaries'])}  related nodes"
            )

        except concurrent.futures.TimeoutError:
            logger.warning(f"Tiempo de espera de búsqueda Zep agotado ({entity_name})")
        except Exception as e:
            logger.warning(f"Búsqueda Zep fallida ({entity_name}): {e}")

        return results

        comprehensive_query = f"Toda la información, actividades, eventos, relaciones y contexto sobre {entity_name}"

        def search_edges():
            """Buscar edges (hechos/relaciones) - con mecanismo de reintento"""
            max_retries = 3
            last_exception = None
            delay = 2.0

            for attempt in range(max_retries):
                try:
                    return self.zep_client.graph.search(
                        query=comprehensive_query,
                        graph_id=self.graph_id,
                        limit=30,
                        scope="edges",
                        reranker="rrf",
                    )
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        logger.debug(
                            f"Zep edge search attempt {attempt + 1}  failed: {str(e)[:80]}, retrying..."
                        )
                        time.sleep(delay)
                        delay *= 2
                    else:
                        logger.debug(
                            f"Zep edge search after {max_retries}  attempts still failed: {e}"
                        )
            return None

        def search_nodes():
            """Buscar nodos (resumen de entidad) - con mecanismo de reintento"""
            max_retries = 3
            last_exception = None
            delay = 2.0

            for attempt in range(max_retries):
                try:
                    return self.zep_client.graph.search(
                        query=comprehensive_query,
                        graph_id=self.graph_id,
                        limit=20,
                        scope="nodes",
                        reranker="rrf",
                    )
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        logger.debug(
                            f"Zep node search attempt {attempt + 1}  failed: {str(e)[:80]}, retrying..."
                        )
                        time.sleep(delay)
                        delay *= 2
                    else:
                        logger.debug(
                            f"Zepnodobúsquedaen {max_retries}  attempts still failed: {e}"
                        )
            return None

        try:
            # Ejecutar búsqueda de edges y nodes en paralelo
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                edge_future = executor.submit(search_edges)
                node_future = executor.submit(search_nodes)

                # Obtener resultados
                edge_result = edge_future.result(timeout=30)
                node_result = node_future.result(timeout=30)

            # Procesar resultados de búsqueda de edges
            all_facts = set()
            if edge_result and hasattr(edge_result, "edges") and edge_result.edges:
                for edge in edge_result.edges:
                    if hasattr(edge, "fact") and edge.fact:
                        all_facts.add(edge.fact)
            results["facts"] = list(all_facts)

            # Procesar resultados de búsqueda de nodes
            all_summaries = set()
            if node_result and hasattr(node_result, "nodes") and node_result.nodes:
                for node in node_result.nodes:
                    if hasattr(node, "summary") and node.summary:
                        all_summaries.add(node.summary)
                    if hasattr(node, "name") and node.name and node.name != entity_name:
                        all_summaries.add(f"Entidad relacionada: {node.name}")
            results["node_summaries"] = list(all_summaries)

            # Construir contexto  (completo)
            context_parts = []
            if results["facts"]:
                context_parts.append(
                    "Información de hechos:\n"
                    + "\n".join(f"- {f}" for f in results["facts"][:20])
                )
            if results["node_summaries"]:
                context_parts.append(
                    "Entidad relacionada:\n"
                    + "\n".join(f"- {s}" for s in results["node_summaries"][:10])
                )
            results["context"] = "\n\n".join(context_parts)

            logger.info(
                f"Zep hybrid search completed: {entity_name}, obtained {len(results['facts'])}  facts, {len(results['node_summaries'])}  related nodes"
            )

        except concurrent.futures.TimeoutError:
            logger.warning(f"Tiempo de espera de búsqueda Zep agotado ({entity_name})")
        except Exception as e:
            logger.warning(f"Búsqueda Zep fallida ({entity_name}): {e}")

        return results

    def _build_entity_context(self, entity: EntityNode) -> str:
        """
        Construir información completa de contexto de la entidad

        Incluye:
        1. Información de edges de la entidad misma (hechos)
        2. Información detallada de nodos relacionados
        3. Información rica obtenida de búsqueda híbrida Zep
        """
        context_parts = []

        # 1. Agregar atributos de entidad
        if entity.attributes:
            attrs = []
            for key, value in entity.attributes.items():
                if value and str(value).strip():
                    attrs.append(f"- {key}: {value}")
            if attrs:
                context_parts.append("### Atributos de entidad\n" + "\n".join(attrs))

        # 2. Agregar información de edges relacionados (hechos/relaciones)
        existing_facts = set()
        if entity.related_edges:
            relationships = []
            for edge in entity.related_edges:  # Sin límite de cantidad
                fact = edge.get("fact", "")
                edge_name = edge.get("edge_name", "")
                direction = edge.get("direction", "")

                if fact:
                    relationships.append(f"- {fact}")
                    existing_facts.add(fact)
                elif edge_name:
                    if direction == "outgoing":
                        relationships.append(
                            f"- {entity.name} --[{edge_name}]--> (Entidad relacionada)"
                        )
                    else:
                        relationships.append(
                            f"- (Entidad relacionada) --[{edge_name}]--> {entity.name}"
                        )

            if relationships:
                context_parts.append(
                    "### Hechos y relaciones relacionados\n" + "\n".join(relationships)
                )

        # 3. Agregar información detallada de nodos relacionados
        if entity.related_nodes:
            related_info = []
            for node in entity.related_nodes:  # Sin límite de cantidad
                node_name = node.get("name", "")
                node_labels = node.get("labels", [])
                node_summary = node.get("summary", "")

                # Filtrar etiquetas por defecto
                custom_labels = [l for l in node_labels if l not in ["Entity", "Node"]]
                label_str = f" ({', '.join(custom_labels)})" if custom_labels else ""

                if node_summary:
                    related_info.append(f"- **{node_name}**{label_str}: {node_summary}")
                else:
                    related_info.append(f"- **{node_name}**{label_str}")

            if related_info:
                context_parts.append(
                    "### Información de entidades relacionadas\n"
                    + "\n".join(related_info)
                )

        # 4. Usar búsqueda híbrida Zep para obtener información más rica
        zep_results = self._search_zep_for_entity(entity)

        if zep_results.get("facts"):
            # Deduplicar: excluir hechos ya existentes
            new_facts = [f for f in zep_results["facts"] if f not in existing_facts]
            if new_facts:
                context_parts.append(
                    "### Información de hechos encontrada por búsqueda Zep\n"
                    + "\n".join(f"- {f}" for f in new_facts[:15])
                )

        if zep_results.get("node_summaries"):
            context_parts.append(
                "### Nodos relacionados encontrados por búsqueda Zep\n"
                + "\n".join(f"- {s}" for s in zep_results["node_summaries"][:10])
            )

        return "\n\n".join(context_parts)

    def _is_individual_entity(self, entity_type: str) -> bool:
        """Determinar si es una entidad de tipo individual"""
        return entity_type.lower() in self.INDIVIDUAL_ENTITY_TYPES

    def _is_group_entity(self, entity_type: str) -> bool:
        """Determinar si es una entidad de tipo grupo/institucional"""
        return entity_type.lower() in self.GROUP_ENTITY_TYPES

    def _generate_profile_with_llm(
        self,
        entity_name: str,
        entity_type: str,
        entity_summary: str,
        entity_attributes: Dict[str, Any],
        context: str,
    ) -> Dict[str, Any]:
        """
        Usar LLM para generar perfil muy detallado

        Distinguir según el tipo de entidad:
        - Entidad individual: generar personaje específico
        - Entidad grupo/institucional: generar cuenta representativa
        """

        is_individual = self._is_individual_entity(entity_type)

        if is_individual:
            prompt = self._build_individual_persona_prompt(
                entity_name, entity_type, entity_summary, entity_attributes, context
            )
        else:
            prompt = self._build_group_persona_prompt(
                entity_name, entity_type, entity_summary, entity_attributes, context
            )

        # Intentar generar múltiples veces hasta tener éxito o alcanzar el máximo de reintentos
        max_attempts = 3
        last_error = None

        for attempt in range(max_attempts):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {
                            "role": "system",
                            "content": self._get_system_prompt(is_individual),
                        },
                        {"role": "user", "content": prompt},
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.7
                    - (attempt * 0.1),  # Reducir temperatura en cada reintento
                    # No establecer max_tokens, dejar que el LLM decida
                )

                content = response.choices[0].message.content

                # Verificar si fue truncado (finish_reason no es'stop'）
                finish_reason = response.choices[0].finish_reason
                if finish_reason == "length":
                    logger.warning(
                        f"Salida del LLM truncada (attempt {attempt + 1}), Intentando reparar..."
                    )
                    content = self._fix_truncated_json(content)

                # Intentar parsear JSON
                try:
                    result = json.loads(content)

                    # Verificar campos requeridos
                    if "bio" not in result or not result["bio"]:
                        result["bio"] = (
                            entity_summary[:200]
                            if entity_summary
                            else f"{entity_type}: {entity_name}"
                        )
                    if "persona" not in result or not result["persona"]:
                        result["persona"] = (
                            entity_summary or f"{entity_name} es un/una {entity_type}."
                        )

                    return result

                except json.JSONDecodeError as je:
                    logger.warning(
                        f"Parsing JSON fallido (attempt {attempt + 1}): {str(je)[:80]}"
                    )

                    # Intentando repararJSON
                    result = self._try_fix_json(
                        content, entity_name, entity_type, entity_summary
                    )
                    if result.get("_fixed"):
                        del result["_fixed"]
                        return result

                    last_error = je

            except Exception as e:
                logger.warning(
                    f"Llamada LLM fallida (attempt {attempt + 1}): {str(e)[:80]}"
                )
                last_error = e
                import time

                time.sleep(1 * (attempt + 1))  # Retroceso exponencial

        logger.warning(
            f"Generación de perfil con LLM fallida（{max_attempts}）: {last_error}, usarreglagenerar"
        )
        return self._generate_profile_rule_based(
            entity_name, entity_type, entity_summary, entity_attributes
        )

    def _fix_truncated_json(self, content: str) -> str:
        """Reparar JSON truncado (salida truncada por límite de max_tokens)"""
        import re

        # Si el JSON está truncado, intentar cerrarlo
        content = content.strip()

        # Calcular paréntesis sin cerrar
        open_braces = content.count("{") - content.count("}")
        open_brackets = content.count("[") - content.count("]")

        # Verificar si hay cadenas sin cerrar
        # Verificación simple: si después de la última comilla no hay coma o cierre,puedepor truncado
        if content and content[-1] not in '",}]':
            # Intentar cerrar cadena
            content += '"'

        # Cerrar paréntesis
        content += "]" * open_brackets
        content += "}" * open_braces

        return content

    def _try_fix_json(
        self, content: str, entity_name: str, entity_type: str, entity_summary: str = ""
    ) -> Dict[str, Any]:
        """Intentar reparar JSON dañado"""
        import re

        # 1. Primero intentar reparar el truncamiento
        content = self._fix_truncated_json(content)

        # 2. Intentar extraer parte JSON
        json_match = re.search(r"\{[\s\S]*\}", content)
        if json_match:
            json_str = json_match.group()

            # 3. Manejar problemas de saltos de línea en cadenas
            # Encontrar todos los valores de cadena y reemplazar saltos de línea
            def fix_string_newlines(match):
                s = match.group(0)
                # Reemplazar saltos de línea reales dentro de cadenas con espacios
                s = s.replace("\n", " ").replace("\r", " ")
                # Reemplazar espacios redundantes
                s = re.sub(r"\s+", " ", s)
                return s

            # Coincidir valores de cadena JSON
            json_str = re.sub(
                r'"[^"\\]*(?:\\.[^"\\]*)*"', fix_string_newlines, json_str
            )

            # 4. Intentar parsear
            try:
                result = json.loads(json_str)
                result["_fixed"] = True
                return result
            except json.JSONDecodeError as e:
                # 5. Si sigue fallando, intentar reparación más agresiva
                try:
                    # Remover todos los caracteres de control
                    json_str = re.sub(r"[\x00-\x1f\x7f-\x9f]", " ", json_str)
                    # Reemplazar todo espacio continuo
                    json_str = re.sub(r"\s+", " ", json_str)
                    result = json.loads(json_str)
                    result["_fixed"] = True
                    return result
                except:
                    pass

        # 6. Intentar extraer información parcial del contenido
        bio_match = re.search(r'"bio"\s*:\s*"([^"]*)"', content)
        persona_match = re.search(
            r'"persona"\s*:\s*"([^"]*)', content
        )  # Posiblemente truncado

        bio = (
            bio_match.group(1)
            if bio_match
            else (
                entity_summary[:200]
                if entity_summary
                else f"{entity_type}: {entity_name}"
            )
        )
        persona = (
            persona_match.group(1)
            if persona_match
            else (entity_summary or f"{entity_name} es un/una {entity_type}.")
        )

        # Si se extrajo contenido significativo, marcar como reparado
        if bio_match or persona_match:
            logger.info(f"Extraída información parcial del JSON dañado")
            return {"bio": bio, "persona": persona, "_fixed": True}

        # 7. Fallo total, retornar estructura básica
        logger.warning(f"Reparación de JSON fallida, retornando estructura básica")
        return {
            "bio": entity_summary[:200]
            if entity_summary
            else f"{entity_type}: {entity_name}",
            "persona": entity_summary or f"{entity_name} es un/una {entity_type}.",
        }

    def _get_system_prompt(self, is_individual: bool) -> str:
        """Obtener prompt del sistema"""
        if _PROMPTS_AVAILABLE:
            return _load_prompt("oasis", "profile_system")
        # Fallback al original en chino
        return "Eres un experto en generación de perfiles de usuarios de redes sociales. Genera perfiles detallados y realistas para simulación de opinión pública, reproduciendo al máximo la realidad existente. Debes retornar JSON válido, ninguna cadena de texto puede contener saltos de línea sin escapar. Usa español."

    def _build_individual_persona_prompt(
        self,
        entity_name: str,
        entity_type: str,
        entity_summary: str,
        entity_attributes: Dict[str, Any],
        context: str,
    ) -> str:
        """Construir prompt de perfil detallado para entidad individual"""

        attrs_str = (
            json.dumps(entity_attributes, ensure_ascii=False)
            if entity_attributes
            else "Ninguno"
        )
        context_str = context[:3000] if context else "Sin contexto adicional"

        # Usar prompts i18n si están disponibles
        if _PROMPTS_AVAILABLE:
            return _load_prompt(
                "oasis",
                "individual_persona",
                entity_name=entity_name,
                entity_type=entity_type,
                entity_summary=entity_summary,
                entity_attributes=attrs_str,
                context=context_str,
            )

        # Fallback al original en chino
        return f"""paraentidadgenerardetalladomediousuarioperfil,tiene。

entidadnombre: {entity_name}
entidadtipo: {entity_type}
entidaddebe: {entity_summary}
entidadatributo: {attrs_str}

contextoinformación:
{context_str}

por favorgenerarJSON，incluircon:

1. bio: medio，200
2. persona: descripción de perfil detallado（2000），incluir:
   - información（、、、queen）
   - （debe、conevento、relación）
   - （MBTItipo、、）
   - medioejecutarpara（frecuencia、、、）
   - （para、puedepor/）
   - （、、individual）
   - individual（perfildebe，debeestoconevento，conestoeneventotieneaccióncon）
3. age: （debees）
4. gender: ，debees: "male" o "female"
5. mbti: MBTItipo（INTJ、ENFP）
6. country: （usarchino，"China"）
7. profession: 
8. interested_topics: 

debe:
- quetienedebeeso，nodebeusarejecutar
- personadebeesdescripción
- usarchino（genderdebemale/female）
- debeconentidadinformación
- agedebeestiene，genderdebees"male"o"female"
"""

    def _build_group_persona_prompt(
        self,
        entity_name: str,
        entity_type: str,
        entity_summary: str,
        entity_attributes: Dict[str, Any],
        context: str,
    ) -> str:
        """Construir prompt de perfil detallado para entidad grupo/institucional"""

        attrs_str = (
            json.dumps(entity_attributes, ensure_ascii=False)
            if entity_attributes
            else "Ninguno"
        )
        context_str = context[:3000] if context else "Sin contexto adicional"

        # Usar prompts i18n si están disponibles
        if _PROMPTS_AVAILABLE:
            return _load_prompt(
                "oasis",
                "group_persona",
                entity_name=entity_name,
                entity_type=entity_type,
                entity_summary=entity_summary,
                entity_attributes=attrs_str,
                context=context_str,
            )

        # Fallback al original en chino
        return f"""Generar configuración detallada de cuenta de redes sociales para entidad institucional/grupo, reproduciendo al máximo la realidad existente.

Nombre de entidad: {entity_name}
Tipo de entidad: {entity_type}
Resumen de entidad: {entity_summary}
Atributos de entidad: {attrs_str}

Información de contexto:
{context_str}

Por favor generar JSON con los siguientes campos:

1. bio: Biografía oficial de la cuenta, 200 caracteres, profesional y apropiada
2. persona: Descripción detallada de la cuenta (2000 caracteres de texto plano), debe incluir:
   - Información básica institucional (nombre oficial, naturaleza institucional, antecedentes de fundación, funciones principales)
   - Posicionamiento de la cuenta (tipo de cuenta, audiencia objetivo, funciones principales)
   - Estilo de expresión (características del lenguaje, expresiones comunes, temas taboo)
   - Características del contenido publicado (tipos de contenido, frecuencia de publicación, horarios activos)
   - Actitud y posición (posición oficial sobre temas centrales, manejo de controversias)
   - Notas especiales (perfil del grupo que representa, hábitos de operación)
   - Memoria institucional (parte importante del perfil, presentar la conexión de esta institución con el evento, y las acciones/reacciones previas de la institución en el evento)
3. age: Llenar siempre con 30 (edad virtual de la cuenta institucional)
4. gender: Llenar siempre con "other" (cuentas institucionales usan "other" para indicar no individual)
5. mbti: Tipo MBTI, usado para describir el estilo de la cuenta, como ISTJ representa riguroso y conservador
6. country: País (usar español, como "China")
7. profession: Descripción de función institucional
8. interested_topics: Arreglo de áreas de interés

Importante:
- Todos los valores de campos deben ser strings o números, no se permiten valores null
- persona debe ser una descripción de texto fluida, no usar saltos de línea
- usar español (excepto el campo gender que debe usar inglés "other")
- age debe ser el entero 30, gender debe ser el string "other"
- El discurso de la cuenta institucional debe ser consistente con su identidad"""

    def _generate_profile_rule_based(
        self,
        entity_name: str,
        entity_type: str,
        entity_summary: str,
        entity_attributes: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Usar reglas para generar perfil básico"""

        # Generar diferente perfil según tipo de entidad
        entity_type_lower = entity_type.lower()

        if entity_type_lower in ["student", "alumni"]:
            return {
                "bio": f"{entity_type} with interests in academics and social issues.",
                "persona": f"{entity_name} is a {entity_type.lower()} who is actively engaged in academic and social discussions. They enjoy sharing perspectives and connecting with peers.",
                "age": random.randint(18, 30),
                "gender": random.choice(["male", "female"]),
                "mbti": random.choice(self.MBTI_TYPES),
                "country": random.choice(self.COUNTRIES),
                "profession": "Student",
                "interested_topics": ["Education", "Social Issues", "Technology"],
            }

        elif entity_type_lower in ["publicfigure", "expert", "faculty"]:
            return {
                "bio": f"Expert and thought leader in their field.",
                "persona": f"{entity_name} is a recognized {entity_type.lower()} who shares insights and opinions on important matters. They are known for their expertise and influence in public discourse.",
                "age": random.randint(35, 60),
                "gender": random.choice(["male", "female"]),
                "mbti": random.choice(["ENTJ", "INTJ", "ENTP", "INTP"]),
                "country": random.choice(self.COUNTRIES),
                "profession": entity_attributes.get("occupation", "Expert"),
                "interested_topics": ["Politics", "Economics", "Culture & Society"],
            }

        elif entity_type_lower in ["mediaoutlet", "socialmediaplatform"]:
            return {
                "bio": f"Official account for {entity_name}. News and updates.",
                "persona": f"{entity_name} is a media entity that reports news and facilitates public discourse. The account shares timely updates and engages with the audience on current events.",
                "age": 30,  # Edad virtual institucional
                "gender": "other",  # institucional usa other
                "mbti": "ISTJ",  # Estilo institucional: riguroso y conservador
                "country": "China",
                "profession": "Media",
                "interested_topics": [
                    "General News",
                    "Current Events",
                    "Public Affairs",
                ],
            }

        elif entity_type_lower in [
            "university",
            "governmentagency",
            "ngo",
            "organization",
        ]:
            return {
                "bio": f"Official account of {entity_name}.",
                "persona": f"{entity_name} is an institutional entity that communicates official positions, announcements, and engages with stakeholders on relevant matters.",
                "age": 30,  # Edad virtual institucional
                "gender": "other",  # institucional usa other
                "mbti": "ISTJ",  # Estilo institucional: riguroso y conservador
                "country": "China",
                "profession": entity_type,
                "interested_topics": [
                    "Public Policy",
                    "Community",
                    "Official Announcements",
                ],
            }

        else:
            # perfil por defecto
            return {
                "bio": entity_summary[:150]
                if entity_summary
                else f"{entity_type}: {entity_name}",
                "persona": entity_summary
                or f"{entity_name} is a {entity_type.lower()} participating in social discussions.",
                "age": random.randint(25, 50),
                "gender": random.choice(["male", "female"]),
                "mbti": random.choice(self.MBTI_TYPES),
                "country": random.choice(self.COUNTRIES),
                "profession": entity_type,
                "interested_topics": ["General", "Social Issues"],
            }

    def set_graph_id(self, graph_id: str):
        """Configurar ID del grafo para búsqueda Zep"""
        self.graph_id = graph_id

    def generate_profiles_from_entities(
        self,
        entities: List[EntityNode],
        use_llm: bool = True,
        progress_callback: Optional[callable] = None,
        graph_id: Optional[str] = None,
        parallel_count: int = 5,
        realtime_output_path: Optional[str] = None,
        output_platform: str = "reddit",
    ) -> List[OasisAgentProfile]:
        """
        Generar Agent Profile en lote desde entidades (soporta generación paralela)

        Args:
            entities: lista de entidades
            use_llm: si usar LLM para generar perfil detallado
            progress_callback: función de callback de progreso (current, total, message)
            graph_id: ID del grafo, para búsqueda Zep obtener contexto más rico
            parallel_count: cantidad de generación paralela, por defecto 5
            realtime_output_path: ruta de archivo para escritura en tiempo real（siproveer，generarescribir）
            output_platform: formato de plataforma de salida ("reddit" o "twitter")

        Returns:
            lista de Agent Profile
        """
        import concurrent.futures
        from threading import Lock

        # Configurar graph_id para búsqueda Zep
        if graph_id:
            self.graph_id = graph_id

        total = len(entities)
        profiles = [None] * total  # Preasignar lista para mantener orden
        completed_count = [0]  # Usar lista para modificar en closure
        lock = Lock()

        # tiempo realescribirarchivofunción
        def save_profiles_realtime():
            """Guardar en tiempo real los profiles generados al archivo"""
            if not realtime_output_path:
                return

            with lock:
                # Filtrar profiles ya generados
                existing_profiles = [p for p in profiles if p is not None]
                if not existing_profiles:
                    return

                try:
                    if output_platform == "reddit":
                        # Reddit JSON
                        profiles_data = [
                            p.to_reddit_format() for p in existing_profiles
                        ]
                        with open(realtime_output_path, "w", encoding="utf-8") as f:
                            json.dump(profiles_data, f, ensure_ascii=False, indent=2)
                    else:
                        # Twitter CSV
                        import csv

                        profiles_data = [
                            p.to_twitter_format() for p in existing_profiles
                        ]
                        if profiles_data:
                            fieldnames = list(profiles_data[0].keys())
                            with open(
                                realtime_output_path, "w", encoding="utf-8", newline=""
                            ) as f:
                                writer = csv.DictWriter(f, fieldnames=fieldnames)
                                writer.writeheader()
                                writer.writerows(profiles_data)
                except Exception as e:
                    logger.warning(f"Guardado de profiles en tiempo real fallido: {e}")

        def generate_single_profile(idx: int, entity: EntityNode) -> tuple:
            """función de trabajo para generar un solo profile"""
            entity_type = entity.get_entity_type() or "Entity"

            try:
                profile = self.generate_profile_from_entity(
                    entity=entity, user_id=idx, use_llm=use_llm
                )

                # Salida en tiempo real del perfil generado a consola y logs
                self._print_generated_profile(entity.name, entity_type, profile)

                return idx, profile, None

            except Exception as e:
                logger.error(f"generar entidad {entity.name} perfilfallido: {str(e)}")
                # crearbásicoprofile
                fallback_profile = OasisAgentProfile(
                    user_id=idx,
                    user_name=self._generate_username(entity.name),
                    name=entity.name,
                    bio=f"{entity_type}: {entity.name}",
                    persona=entity.summary or f"A participant in social discussions.",
                    source_entity_uuid=entity.uuid,
                    source_entity_type=entity_type,
                )
                return idx, fallback_profile, str(e)

        logger.info(
            f"Iniciar generación paralela {total} Agentperfil（yejecutar: {parallel_count}）..."
        )
        print(f"\n{'=' * 60}")
        print(
            f"Iniciar generación de Agent profile -  {total} entidades, número paralelo: {parallel_count}"
        )
        print(f"{'=' * 60}\n")

        # Usar pool de threads para ejecución en paralelo
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=parallel_count
        ) as executor:
            # quetiene
            future_to_entity = {
                executor.submit(generate_single_profile, idx, entity): (idx, entity)
                for idx, entity in enumerate(entities)
            }

            # Recolectar resultados
            for future in concurrent.futures.as_completed(future_to_entity):
                idx, entity = future_to_entity[future]
                entity_type = entity.get_entity_type() or "Entity"

                try:
                    result_idx, profile, error = future.result()
                    profiles[result_idx] = profile

                    with lock:
                        completed_count[0] += 1
                        current = completed_count[0]

                    # tiempo realescribirarchivo
                    save_profiles_realtime()

                    if progress_callback:
                        progress_callback(
                            current,
                            total,
                            f"Completado {current}/{total}: {entity.name}（{entity_type}）",
                        )

                    if error:
                        logger.warning(
                            f"[{current}/{total}] {entity.name} usarrespaldoperfil: {error}"
                        )
                    else:
                        logger.info(
                            f"[{current}/{total}] generar perfil exitosamente: {entity.name} ({entity_type})"
                        )

                except Exception as e:
                    logger.error(f"procesarentidad {entity.name} : {str(e)}")
                    with lock:
                        completed_count[0] += 1
                    profiles[idx] = OasisAgentProfile(
                        user_id=idx,
                        user_name=self._generate_username(entity.name),
                        name=entity.name,
                        bio=f"{entity_type}: {entity.name}",
                        persona=entity.summary
                        or "A participant in social discussions.",
                        source_entity_uuid=entity.uuid,
                        source_entity_type=entity_type,
                    )
                    # tiempo realescribirarchivo（haceresrespaldoperfil）
                    save_profiles_realtime()

        print(f"\n{'=' * 60}")
        print(
            f"perfilgenerarcompletado！generar {len([p for p in profiles if p])} Agent"
        )
        print(f"{'=' * 60}\n")

        return profiles

    def _print_generated_profile(
        self, entity_name: str, entity_type: str, profile: OasisAgentProfile
    ):
        """Salida en tiempo real del perfil generado a consola (contenido completo, sin truncar)"""
        separator = "-" * 70

        # Construir contenido de salida completo (sin truncar)）
        topics_str = (
            ", ".join(profile.interested_topics)
            if profile.interested_topics
            else "Ninguno"
        )

        output_lines = [
            f"\n{separator}",
            f"[Ya generado] {entity_name} ({entity_type})",
            f"{separator}",
            f"usuario: {profile.user_name}",
            f"",
            f"[]",
            f"{profile.bio}",
            f"",
            f"[Perfil detallado]",
            f"{profile.persona}",
            f"",
            f"[Atributos básicos]",
            f": {profile.age} | : {profile.gender} | MBTI: {profile.mbti}",
            f": {profile.profession} | : {profile.country}",
            f": {topics_str}",
            separator,
        ]

        output = "\n".join(output_lines)

        # hasta（，loggerno）
        print(output)

    def save_profiles(
        self,
        profiles: List[OasisAgentProfile],
        file_path: str,
        platform: str = "reddit",
    ):
        """
        Guardar Profile en archivo（）

        Requisitos de formato de plataforma OASIS:
        - Twitter: CSV
        - Reddit: JSON

        Args:
            profiles: lista de Profile
            file_path: ruta de archivo
            platform: tipo de plataforma ("reddit" o "twitter")
        """
        if platform == "twitter":
            self._save_twitter_csv(profiles, file_path)
        else:
            self._save_reddit_json(profiles, file_path)

    def _save_twitter_csv(self, profiles: List[OasisAgentProfile], file_path: str):
        """
        Guardar Twitter Profile en formato CSV（OASISdebe）

        Campos CSV requeridos por OASIS Twitter：
        - user_id: ID de usuario（CSVdesde0iniciar）
        - name: Nombre real del usuario
        - username: Nombre de usuario en el sistema
        - user_char: descripción de perfil detallado（hastaprompt de sistema LLM，guiar comportamiento del Agent）
        - description: breve biografía pública（enpágina de perfil del usuario）

        Diferencia entre user_char y description：
        - user_char: usar internamente，prompt de sistema LLM，decidir cómo el Agent piensa y actúa
        - description: mostrar externamente，biografía visible para otros usuarios
        """
        import csv

        # Asegurar que extensión sea .csv
        if not file_path.endswith(".csv"):
            file_path = file_path.replace(".json", ".csv")

        with open(file_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            # escribirencabezado requerido por OASIS
            headers = ["user_id", "name", "username", "user_char", "description"]
            writer.writerow(headers)

            # escribir filas de datos
            for idx, profile in enumerate(profiles):
                # user_char: perfil completo (bio + persona), para prompt de sistema LLM
                user_char = profile.bio
                if profile.persona and profile.persona != profile.bio:
                    user_char = f"{profile.bio} {profile.persona}"
                # Manejar saltos de línea (usar espacios en CSV)）
                user_char = user_char.replace("\n", " ").replace("\r", " ")

                # description: corta para mostrar externamente
                description = profile.bio.replace("\n", " ").replace("\r", " ")

                row = [
                    idx,  # user_id: desde0iniciarID
                    profile.name,  # name:
                    profile.user_name,  # username: usuario
                    user_char,  # user_char: perfil completo (usado internamente por LLM)
                    description,  # description: corta (mostrada externamente)
                ]
                writer.writerow(row)

        logger.info(
            f"guardar {len(profiles)} Twitter Profilehasta {file_path} (OASIS CSV)"
        )

    def _normalize_gender(self, gender: Optional[str]) -> str:
        """
        genderparaOASISdebe

        OASIS requiere: male, female, other
        """
        if not gender:
            return "other"

        gender_lower = gender.lower().strip()

        # Mapeo de chino
        gender_map = {
            "": "male",
            "": "female",
            "institucional": "other",
            "su": "other",
            # tiene
            "male": "male",
            "female": "female",
            "other": "other",
        }

        return gender_map.get(gender_lower, "other")

    def _save_reddit_json(self, profiles: List[OasisAgentProfile], file_path: str):
        """
        guardarReddit ProfileparaJSON

        Usar formato consistente con to_reddit_format(), asegurar que OASIS pueda leer correctamente.
        Debe incluir campo user_id, ¡esto es clave para coincidencia en OASIS agent_graph.get_agent()!

        Campos requeridos:
        - user_id: ID de usuario（，en initial_posts  poster_agent_id）
        - username: usuario
        - name: nombre
        - bio:
        - persona: detalladoperfil
        - age: （）
        - gender: "male", "female", o "other"
        - mbti: MBTItipo
        - country:
        """
        data = []
        for idx, profile in enumerate(profiles):
            # usarcon to_reddit_format()
            item = {
                "user_id": profile.user_id
                if profile.user_id is not None
                else idx,  # Clave: debe incluir user_id
                "username": profile.user_name,
                "name": profile.name,
                "bio": profile.bio[:150] if profile.bio else f"{profile.name}",
                "persona": profile.persona
                or f"{profile.name} is a participant in social discussions.",
                "karma": profile.karma if profile.karma else 1000,
                "created_at": profile.created_at,
                # Campos requeridos por OASIS - asegurar valores por defecto
                "age": profile.age if profile.age else 30,
                "gender": self._normalize_gender(profile.gender),
                "mbti": profile.mbti if profile.mbti else "ISTJ",
                "country": profile.country if profile.country else "China",
            }

            #
            if profile.profession:
                item["profession"] = profile.profession
            if profile.interested_topics:
                item["interested_topics"] = profile.interested_topics

            data.append(item)

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        logger.info(
            f"guardar {len(profiles)} Reddit Profilehasta {file_path} (JSON，incluiruser_id)"
        )

    # métodopara，
    def save_profiles_to_json(
        self,
        profiles: List[OasisAgentProfile],
        file_path: str,
        platform: str = "reddit",
    ):
        """[] por favorusar save_profiles() método"""
        logger.warning("save_profiles_to_json，por favorusarsave_profilesmétodo")
        self.save_profiles(profiles, file_path, platform)
