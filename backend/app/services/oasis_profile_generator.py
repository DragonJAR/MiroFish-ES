"""
Generador de perfiles de Agent OASIS
Convertir entidades del grafo Zep al Formato de Agent Profile requerido por la plataForma de simulación OASIS

Mejoras implementadas:
1. Llamar a la función de recuperación de Zep para enriquecer la inFormación del nodo
2. Optimizar el prompt para generar personificaciones muy detalladas
3. Distinguir entre entidades individuales y entidades de grupo abstractas
"""

import json
import random
import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime

from openai import OpenAI
from zep_cloud.client import Zep

from ..config import Config
from ..utils.logger import get_logger
from ..utils.locale import get_language_instruction, get_locale, set_locale, t
from .zep_entity_reader import EntityNode, ZepEntityReader

logger = get_logger("mirofish.oasis_profile")


@dataclass
class OasisAgentProfile:
    """Estructura de datos de Agent Profile OASIS"""

    # Campos generales
    user_id: int
    user_name: str
    name: str
    bio: str
    persona: str

    # Campos opcionales - Estilo Reddit
    karma: int = 1000

    # Campos opcionales - Estilo Twitter
    friend_count: int = 100
    follower_count: int = 150
    statuses_count: int = 500

    # InFormación Adicional de personificación
    age: Optional[int] = None
    gender: Optional[str] = None
    mbti: Optional[str] = None
    country: Optional[str] = None
    proFession: Optional[str] = None
    interested_topics: List[str] = field(default_factory=list)

    # InFormación de entidad fuente
    source_entity_uuid: Optional[str] = None
    source_entity_type: Optional[str] = None

    created_at: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d"))

    def to_reddit_Format(self) -> Dict[str, Any]:
        """Convertir al Formato de plataForma Reddit"""
        profile = {
            "user_id": self.user_id,
            "username": self.user_name,  # La biblioteca OASIS requiere que el campo se llame username (sin guiones Bajos)
            "name": self.name,
            "bio": self.bio,
            "persona": self.persona,
            "karma": self.karma,
            "created_at": self.created_at,
        }

        # Agregar inFormación Adicional de personificación (si existe)
        if self.age:
            profile["age"] = self.age
        if self.gender:
            profile["gender"] = self.gender
        if self.mbti:
            profile["mbti"] = self.mbti
        if self.country:
            profile["country"] = self.country
        if self.proFession:
            profile["proFession"] = self.proFession
        if self.interested_topics:
            profile["interested_topics"] = self.interested_topics

        return profile

    def to_twitter_Format(self) -> Dict[str, Any]:
        """Convertir al Formato de plataForma Twitter"""
        profile = {
            "user_id": self.user_id,
            "username": self.user_name,  # La biblioteca OASIS requiere que el campo se llame username (sin guiones Bajos)
            "name": self.name,
            "bio": self.bio,
            "persona": self.persona,
            "friend_count": self.friend_count,
            "follower_count": self.follower_count,
            "statuses_count": self.statuses_count,
            "created_at": self.created_at,
        }

        # Agregar inFormación Adicional de personificación
        if self.age:
            profile["age"] = self.age
        if self.gender:
            profile["gender"] = self.gender
        if self.mbti:
            profile["mbti"] = self.mbti
        if self.country:
            profile["country"] = self.country
        if self.proFession:
            profile["proFession"] = self.proFession
        if self.interested_topics:
            profile["interested_topics"] = self.interested_topics

        return profile

    def to_dict(self) -> Dict[str, Any]:
        """Convertir a Formato de diccionario completo"""
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
            "proFession": self.proFession,
            "interested_topics": self.interested_topics,
            "source_entity_uuid": self.source_entity_uuid,
            "source_entity_type": self.source_entity_type,
            "created_at": self.created_at,
        }


class OasisProfileGenerator:
    """
    Generador de perfiles de Agent OASIS

    Convertir entidades del grafo Zep a Agent Profile requerido por la simulación OASIS

    Características de optimización:
    1. Llamar a la función de recuperación del grafo Zep para obtener un contexto Más rico
    2. Generar personificaciones muy detalladas (incluyendo inFormación básica, Experiencia proFesional, características de personalidad, comportamiento en redes sociales, etc.)
    3. Distinguir entre entidades individuales y entidades de grupo abstractas
    """

    # Lista de tipos MBTI
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

    # Lista de países comunes
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

    # Tipos de entidad individual (requieren generar personificación específica)
    INDIVIDUAL_ENTITY_TYPES = [
        "student",
        "alumni",
        "proFessor",
        "person",
        "publicfigure",
        "expert",
        "faculty",
        "official",
        "journalist",
        "activist",
    ]

    # Tipos de entidad de grupo/institución (requieren generar personificación representativa de grupo)
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
        zep_api_key: Optional[str] = None,
        graph_id: Optional[str] = None,
    ):
        self.api_key = api_key or Config.LLM_API_KEY
        self.base_url = base_url or Config.LLM_BASE_URL
        self.model_name = model_name or Config.LLM_MODEL_NAME

        if not self.api_key:
            raise ValueError("LLM_API_KEY no está configurado")

        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

        # Cliente Zep para recuperación de contexto rico
        self.zep_api_key = zep_api_key or Config.ZEP_API_KEY
        self.zep_client = None
        self.graph_id = graph_id

        if self.zep_api_key:
            try:
                self.zep_client = Zep(api_key=self.zep_api_key)
            except Exception as e:
                logger.warning(f"Error al inicializar el cliente Zep: {e}")

    def generate_profile_from_entity(
        self, entity: EntityNode, user_id: int, use_llm: bool = True
    ) -> OasisAgentProfile:
        """
        Generar Agent Profile OASIS desde entidad Zep

        Args:
            entity: Nodo de entidad Zep
            user_id: ID de usuario (para OASIS)
            use_llm: Si usar LLM para generar personificación detallada

        Returns:
            OasisAgentProfile
        """
        entity_type = entity.get_entity_type() or "Entity"

        # InFormación básica
        name = entity.name
        user_name = self._generate_username(name)

        # Construir inFormación de contexto
        context = self._build_entity_context(entity)
        
        if use_llm:
            # Usar LLM para generar personificación detallada
            profile_data = self._generate_profile_with_llm(
                entity_name=name,
                entity_type=entity_type,
                entity_summary=entity.summary,
                entity_attributes=entity.attributes,
                context=context
            )
        else:
            # Usar Reglas para generar personificación básica
            profile_data = self._generate_profile_rule_based(
                entity_name=name,
                entity_type=entity_type,
                entity_summary=entity.summary,
                entity_attributes=entity.attributes
            )
        
        return OasisAgentProfile(
            user_id=user_id,
            user_name=user_name,
            name=name,
            bio=profile_data.get("bio", f"{entity_type}: {name}"),
            persona=profile_data.get("persona", entity.summary or f"A {entity_type} named {name}."),
            karma=profile_data.get("karma", random.randint(500, 5000)),
            friend_count=profile_data.get("friend_count", random.randint(50, 500)),
            follower_count=profile_data.get("follower_count", random.randint(100, 1000)),
            statuses_count=profile_data.get("statuses_count", random.randint(100, 2000)),
            age=profile_data.get("age"),
            gender=profile_data.get("gender"),
            mbti=profile_data.get("mbti"),
            country=profile_data.get("country"),
            proFession=profile_data.get("proFession"),
            interested_topics=profile_data.get("interested_topics", []),
            source_entity_uuid=entity.uuid,
            source_entity_type=entity_type,
        )
        else:
            # 使用ReglaGenerar基础人设
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
            proFession=profile_data.get("proFession"),
            interested_topics=profile_data.get("interested_topics", []),
            source_entity_uuid=entity.uuid,
            source_entity_type=entity_type,
        )
    
    def _generate_username(self, name: str) -> str:
        """Generar nombre de usuario"""
        # Eliminar caracteres especiales, convertir a minúsculas
        username = name.lower().replace(" ", "_")
        username = ''.join(c for c in username if c.isalnum() or c == '_')
        
        # Agregar sufijo aleatorio para evitar duplicados
        suffix = random.randint(100, 999)
        return f"{username}_{suffix}"

    def _search_zep_for_entity(self, entity: EntityNode) -> Dict[str, Any]:
        """
        Usar la función de búsqueda híbrida del grafo Zep para obtener inFormación rica relacionada con la entidad
        
        Zep no tiene una interfaz de búsqueda híbrida incorporada, es necesario buscar edges y nodes por seParado y luego fusionar los Resultados.
        Usar solicitudes paralelas para buscar simultáneaMente y mejorar la eficiencia.
        
        Args:
            entity: Objeto de nodo de entidad
            
        Returns:
            Diccionario que contiene facts, node_summaries, context
        """
        import concurrent.futures
        
        if not self.zep_client:
            return {"facts": [], "node_summaries": [], "context": ""}
        
        entity_name = entity.name
        
        results = {
            "facts": [],
            "node_summaries": [],
            "context": ""
        }
        
        # Debe tener graph_id para realizar búsquedas
        if not self.graph_id:
            logger.debug(f"Omitir búsqueda Zep: graph_id no configurado")
            return results
        
        comprehensive_query = t('progress.zepSearchQuery', name=entity_name)
        
        def search_nodes():
            """Buscar nodes (resúmenes de entidades) - con Mecanismo de reintento"""
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
                        reranker="rrf"
                    )
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        logger.debug(f"Error en búsqueda de nodes Zep, intento {attempt + 1}/{max_retries}: {str(e)[:80]}, reintentando...")
                        time.sleep(delay)
                        delay *= 2
                    else:
                        logger.debug(f"Búsqueda de nodes Zep falló después de {max_retries} intentos: {e}")
            return None

        def search_nodes():
            """Buscar nodes (resúmenes de entidades) - con Mecanismo de reintento"""
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
                            f"Error en búsqueda de nodes Zep, intento {attempt + 1}/{max_retries}: {str(e)[:80]}, reintentando..."
                        )
                        time.sleep(delay)
                        delay *= 2
                    else:
                        logger.debug(f"Búsqueda de nodes Zep falló después de {max_retries} intentos: {e}")
            return None

        try:
            # Paralelo执FilaedgesYnodesBúsqueda
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                edge_future = executor.submit(search_edges)
                node_future = executor.submit(search_nodes)

                # Obtener Resultados
                edge_result = edge_future.result(timeout=30)
                node_result = node_future.result(timeout=30)
            
            # Procesar Resultados de búsqueda de edges
            all_facts = set()
            if edge_result and hasattr(edge_result, 'edges') and edge_result.edges:
                for edge in edge_result.edges:
                    if hasattr(edge, 'fact') and edge.fact:
                        all_facts.add(edge.fact)
            results["facts"] = list(all_facts)
            
            # Procesar Resultados de búsqueda de nodes
            all_summaries = set()
            if node_result and hasattr(node_result, 'nodes') and node_result.nodes:
                for node in node_result.nodes:
                    if hasattr(node, 'summary') and node.summary:
                        all_summaries.add(node.summary)
                    if hasattr(node, 'name') and node.name and node.name != entity_name:
                        all_summaries.add(f"Entidad relacionada: {node.name}")
            results["node_summaries"] = list(all_summaries)

            # Construir contexto integral
            context_parts = []
            if results["facts"]:
                context_parts.append(
                    "InFormación de hechos:\n" + "\n".join(f"- {f}" for f in results["facts"][:20])
                )
            if results["node_summaries"]:
                context_parts.append(
                    "Entidades relacionadas:\n"
                    + "\n".join(f"- {s}" for s in results["node_summaries"][:10])
                )
            results["context"] = "\n\n".join(context_parts)
            
            logger.info(
                f"Recuperación híbrida Zep completada: {entity_name}, obtenidos {len(results['facts'])} hechos, {len(results['node_summaries'])} nodos relacionados"
            )
        
        except concurrent.futures.TimeoutError:
            logger.warning(f"Tiempo de espera de recuperación Zep agotado ({entity_name})")
        except Exception as e:
            logger.warning(f"Error en recuperación Zep ({entity_name}): {e}")
        
        return results

    def _build_entity_context(self, entity: EntityNode) -> str:
        """
        Construir inFormación de contexto completa de la entidad
        
        Incluye:
        1. InFormación de edges de la entidad misma (hechos)
        2. InFormación detallada de nodos relacionados
        3. InFormación rica recuperada por búsqueda híbrida de Zep
        """
        context_parts = []

        # 1. 添加EntidadAtributoInformación
        if entity.attributes:
            attrs = []
            for key, value in entity.attributes.items():
                if value and str(value).strip():
                    attrs.append(f"- {key}: {value}")
            if attrs:
                context_parts.append("### EntidadAtributo\n" + "\n".join(attrs))

        # 2. 添加RelacionadoBordeInformación（事实/Relación）
        existing_facts = set()
        if entity.reLated_edges:
            relationships = []
            for edge in entity.reLated_edges:  # 不LímiteCantidad
                fact = edge.get("fact", "")
                edge_name = edge.get("edge_name", "")
                direction = edge.get("direction", "")

                if fact:
                    relationships.append(f"- {fact}")
                    existing_facts.add(fact)
                elif edge_name:
                    if direction == "outgoing":
                        relationships.append(f"- {entity.name} --[{edge_name}]--> (entidad relacionada)")
                    else:
                        relationships.append(f"- (entidad relacionada) --[{edge_name}]--> {entity.name}")
            
            if relationships:
                context_parts.append("### Hechos y relaciones relacionados\n" + "\n".join(relationships))

        # 3. Agregar inFormación detallada de nodos relacionados
        if entity.reLated_nodes:
            reLated_info = []
            for node in entity.reLated_nodes:  # No limitar cantidad
                node_name = node.get("name", "")
                node_labels = node.get("labels", [])
                node_summary = node.get("summary", "")
                
                # Filtrar etiquetas predeterminadas
                custom_labels = [l for l in node_labels if l not in ["Entity", "Node"]]
                label_str = f" ({', '.join(custom_labels)})" if custom_labels else ""
                
                if node_summary:
                    reLated_info.append(f"- **{node_name}**{label_str}: {node_summary}")
                else:
                    reLated_info.append(f"- **{node_name}**{label_str}")
            
            if reLated_info:
                context_parts.append("### InFormación de entidades relacionadas\n" + "\n".join(reLated_info))

        # 4. Usar búsqueda híbrida Zep para obtener inFormación Más rica
        zep_results = self._search_zep_for_entity(entity)
        
        if zep_results.get("facts"):
            # Desduplicar: excluir hechos ya existentes
            new_facts = [f for f in zep_results["facts"] if f not in existing_facts]
            if new_facts:
                context_parts.append("### InFormación de hechos recuperada por Zep\n" + "\n".join(f"- {f}" for f in new_facts[:15]))
        
        if zep_results.get("node_summaries"):
            context_parts.append("### Nodos relacionados recuperados por Zep\n" + "\n".join(f"- {s}" for s in zep_results["node_summaries"][:10]))
        
        return "\n\n".join(context_parts)
    
    def _is_individual_entity(self, entity_type: str) -> bool:
        """Determinar si es un tipo de entidad individual"""
        return entity_type.lower() in self.INDIVIDUAL_ENTITY_TYPES
    
    def _is_group_entity(self, entity_type: str) -> bool:
        """Determinar si es un tipo de entidad de grupo/institución"""
        return entity_type.lower() in self.GROUP_ENTITY_TYPES

    def _generate_profile_with_llm(
        self,
        entity_name: str,
        entity_type: str,
        entity_summary: str,
        entity_attributes: Dict[str, Any],
        context: str
    ) -> Dict[str, Any]:
        """
        Usar LLM para generar personificación muy detallada
        
        Distinguir según tipo de entidad:
        - Entidad individual: generar configuración de personaje específica
        - Entidad de grupo/institución: generar configuración de cuenta representativa
        """
        使用LLMGenerarMuy详细的人设

        Basado enEntidadTipo区分：
        - Elementos人Entidad：Generar具体的人物设定
        - 群体/机构Entidad：Generar代表性账号设定
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

        # 尝试多次Generar，HastaÉxitoO达HastaMás大Reintentar次数
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
                    response_Format={"type": "json_object"},
                    temperature=0.7 - (attempt * 0.1),  # 每次Reintentar降低温度
                    # 不Configurarmax_tokens，让LLM自由发挥
                )

                content = response.choices[0].message.content

                # InspecciónSi被截断（finish_reason不Es'stop'）
                finish_reason = response.choices[0].finish_reason
                if finish_reason == "length":
                    logger.warning(
                        f"LLM输出被截断 (attempt {attempt + 1}), 尝试Corrección..."
                    )
                    content = self._fix_truncated_json(content)

                # 尝试AnalizarJSON
                try:
                    result = json.loads(content)

                    # VerificaciónRequeridoCampo
                    if "bio" not in result or not result["bio"]:
                        result["bio"] = (
                            entity_summary[:200]
                            if entity_summary
                            else f"{entity_type}: {entity_name}"
                        )
                    if "persona" not in result or not result["persona"]:
                        result["persona"] = (
                            entity_summary or f"{entity_name}Es一Elementos{entity_type}。"
                        )

                    return result

                except json.JSONDecodeError as je:
                    logger.warning(
                        f"JSONAnalizarFallido (attempt {attempt + 1}): {str(je)[:80]}"
                    )

                    # 尝试CorrecciónJSON
                    result = self._try_fix_json(
                        content, entity_name, entity_type, entity_summary
                    )
                    if result.get("_fixed"):
                        del result["_fixed"]
                        return result

                    last_error = je

            except Exception as e:
                logger.warning(f"LLM调用Fallido (attempt {attempt + 1}): {str(e)[:80]}")
                last_error = e
                import time

                time.sleep(1 * (attempt + 1))  # 指数退避

        logger.warning(
            f"LLMGenerar人设Fallido（{max_attempts}次尝试）: {last_error}, 使用ReglaGenerar"
        )
        return self._generate_profile_rule_based(
            entity_name, entity_type, entity_summary, entity_attributes
        )

    def _fix_truncated_json(self, content: str) -> str:
        """Corrección被截断的JSON（输出被max_tokensLímite截断）"""
        import re

        # SiJSON被截断，尝试闭合它
        content = content.strip()

        # 计算未闭合的括号
        open_braces = content.count("{") - content.count("}")
        open_brackets = content.count("[") - content.count("]")

        # InspecciónSiTener未闭合的字符串
        # 简单Verificar：SiMás后一Elementos引号后没Tener逗号O闭合括号，PosibleEs字符串被截断
        if content and content[-1] not in '",}]':
            # 尝试闭合字符串
            content += '"'

        # 闭合括号
        content += "]" * open_brackets
        content += "}" * open_braces

        return content

    def _try_fix_json(
        self, content: str, entity_name: str, entity_type: str, entity_summary: str = ""
    ) -> Dict[str, Any]:
        """尝试Corrección损坏的JSON"""
        import re

        # 1. 首先尝试Corrección被截断的情况
        content = self._fix_truncated_json(content)

        # 2. 尝试ExtracciónJSON部分
        json_match = re.search(r"\{[\s\S]*\}", content)
        if json_match:
            json_str = json_match.group()

            # 3. Procesar字符串中的换Fila符Problema
            # 找HastaTodos字符串Valor并替换其中的换Fila符
            def fix_string_newlines(match):
                s = match.group(0)
                # Reemplazar字符串内的实际换Fila符为空格
                s = s.replace("\n", " ").replace("\r", " ")
                # Reemplazar多余空格
                s = re.sub(r"\s+", " ", s)
                return s

            # Coincide conJSON字符串Valor
            json_str = re.sub(
                r'"[^"\\]*(?:\\.[^"\\]*)*"', fix_string_newlines, json_str
            )

            # 4. 尝试Analizar
            try:
                result = json.loads(json_str)
                result["_fixed"] = True
                return result
            except json.JSONDecodeError as e:
                # 5. SiTodavíaEsFallido，尝试Más激进的Corrección
                try:
                    # EliminarTodos控制字符
                    json_str = re.sub(r"[\x00-\x1f\x7f-\x9f]", " ", json_str)
                    # ReemplazarTodos连续空白
                    json_str = re.sub(r"\s+", " ", json_str)
                    result = json.loads(json_str)
                    result["_fixed"] = True
                    return result
                except:
                    pass

        # 6. 尝试DesdeContenido中Extracción部分Información
        bio_match = re.search(r'"bio"\s*:\s*"([^"]*)"', content)
        persona_match = re.search(r'"persona"\s*:\s*"([^"]*)', content)  # Posible被截断

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
            else (entity_summary or f"{entity_name}Es一Elementos{entity_type}。")
        )

        # SiExtracciónHasta了Tener意义的Contenido，Marcar为已Corrección
        if bio_match or persona_match:
            logger.info(f"Desde损坏的JSON中Extracción了部分Información")
            return {"bio": bio, "persona": persona, "_fixed": True}

        # 7. 完全Fallido，Volver基础结构
        logger.warning(f"JSONCorrecciónFallido，Volver基础结构")
        return {
            "bio": entity_summary[:200]
            if entity_summary
            else f"{entity_type}: {entity_name}",
            "persona": entity_summary or f"{entity_name}Es一Elementos{entity_type}。",
        }

    def _get_system_prompt(self, is_individual: bool) -> str:
        """ObtenerSistema提示词"""
        base_prompt = "你Es社交媒体用户画像Generar专家。Generar详细、真实的人设用于舆论Simulación,Más大程度Restaurar已Tener现实情况。DebeVolverVálido的JSON格式，Todos字符串Valor不能Contiene未转义的换Fila符。"
        return f"{base_prompt}\n\n{get_language_instruction()}"

    def _build_individual_persona_prompt(
        self,
        entity_name: str,
        entity_type: str,
        entity_summary: str,
        entity_attributes: Dict[str, Any],
        context: str,
    ) -> str:
        """构建Elementos人Entidad的详细人设提示词"""

        attrs_str = (
            json.dumps(entity_attributes, ensure_ascii=False)
            if entity_attributes
            else "Ninguno"
        )
        context_str = context[:3000] if context else "NingunoAdicional上下文"

        return f"""为EntidadGenerar详细的社交媒体用户人设,Más大程度Restaurar已Tener现实情况。

EntidadNombre: {entity_name}
EntidadTipo: {entity_type}
Entidad摘要: {entity_summary}
EntidadAtributo: {attrs_str}

上下文Información:
{context_str}

请GenerarJSON，Contiene以下Campo:

1. bio: 社交媒体Biografía，200字
2. persona: 详细人设Descripción（2000字的纯文本），需Contiene:
   - BásicoInformación（年龄、职业、教育背景、所En地）
   - 人物背景（重要经历、ConEvento的Asociación、社会Relación）
   - 性格特征（MBTITipo、Núcleo性格、情绪表达方式）
   - 社交媒体Fila为（发帖频率、ContenidoPreFerencias、互动风格、语言特点）
   - 立场观点（Para话题的态度、Posible被激怒/感动的Contenido）
   - 独特特征（口头禅、特殊经历、Elementos人爱好）
   - Elementos人记忆（人设的重要部分，要介绍这ElementosElementos体ConEvento的Asociación，以及这ElementosElementos体EnEvento中的已TenerAcciónCon反应）
3. age: 年龄数字（DebeEs整数）
4. gender: 性别，DebeEs英文: "male" O "Female"
5. mbti: MBTITipo（如INTJ、ENFP等）
6. country: 国家（使用中文，如"中国"）
7. proFession: 职业
8. interested_topics: 感兴趣话题数组

重要:
- TodosCampoValorDebeEs字符串O数字，不要使用换Fila符
- personaDebeEs一段连贯的文字Descripción
- {get_language_instruction()} (genderCampoDebe用英文male/Female)
- Contenido要ConEntidadInformación保持一致
- ageDebeEsVálido的整数，genderDebeEs"male"O"Female"
"""

    def _build_group_persona_prompt(
        self,
        entity_name: str,
        entity_type: str,
        entity_summary: str,
        entity_attributes: Dict[str, Any],
        context: str,
    ) -> str:
        """构建群体/机构Entidad的详细人设提示词"""

        attrs_str = (
            json.dumps(entity_attributes, ensure_ascii=False)
            if entity_attributes
            else "Ninguno"
        )
        context_str = context[:3000] if context else "NingunoAdicional上下文"

        return f"""为机构/群体EntidadGenerar详细的社交媒体账号设定,Más大程度Restaurar已Tener现实情况。

EntidadNombre: {entity_name}
EntidadTipo: {entity_type}
Entidad摘要: {entity_summary}
EntidadAtributo: {attrs_str}

上下文Información:
{context_str}

请GenerarJSON，Contiene以下Campo:

1. bio: 官方账号Biografía，200字，专业得体
2. persona: 详细账号设定Descripción（2000字的纯文本），需Contiene:
   - 机构BásicoInformación（正式Nombre、机构性质、成立背景、Principal职能）
   - 账号定位（账号Tipo、目标受众、Núcleo功能）
   - 发言风格（语言特点、常用表达、禁忌话题）
   - PublicarContenido特点（ContenidoTipo、Publicar频率、活跃Tiempo段）
   - 立场态度（ParaNúcleo话题的官方立场、面Para争议的Procesar方式）
   - 特殊Decir明（代表的群体画像、运营习惯）
   - 机构记忆（机构人设的重要部分，要介绍这Elementos机构ConEvento的Asociación，以及这Elementos机构EnEvento中的已TenerAcciónCon反应）
3. age: 固定填30（机构账号的虚拟年龄）
4. gender: 固定填"other"（机构账号使用other表示非Elementos人）
5. mbti: MBTITipo，用于Descripción账号风格，如ISTJ代表严谨保守
6. country: 国家（使用中文，如"中国"）
7. proFession: 机构职能Descripción
8. interested_topics: Seguir领域数组

重要:
- TodosCampoValorDebeEs字符串O数字，不允许nullValor
- personaDebeEs一段连贯的文字Descripción，不要使用换Fila符
- {get_language_instruction()} (genderCampoDebe用英文"other")
- ageDebeEs整数30，genderDebeEs字符串"other"
- 机构账号发言要符合其身份定位"""

    def _generate_profile_rule_based(
        self,
        entity_name: str,
        entity_type: str,
        entity_summary: str,
        entity_attributes: Dict[str, Any],
    ) -> Dict[str, Any]:
        """使用ReglaGenerar基础人设"""

        # Basado enEntidadTipoGenerar不同的人设
        entity_type_lower = entity_type.lower()

        if entity_type_lower in ["student", "alumni"]:
            return {
                "bio": f"{entity_type} with interests in academics and social issues.",
                "persona": f"{entity_name} is a {entity_type.lower()} who is actively engaged in academic and social discussions. They enjoy sharing perspectives and Connecting with peers.",
                "age": random.randint(18, 30),
                "gender": random.choice(["male", "Female"]),
                "mbti": random.choice(self.MBTI_TYPES),
                "country": random.choice(self.COUNTRIES),
                "proFession": "Student",
                "interested_topics": ["Education", "Social Issues", "Technology"],
            }

        elif entity_type_lower in ["publicfigure", "expert", "faculty"]:
            return {
                "bio": f"Expert and thought leader in their field.",
                "persona": f"{entity_name} is a recognized {entity_type.lower()} who shares insights and opinions on important matters. They are known for their expertise and influence in public discourse.",
                "age": random.randint(35, 60),
                "gender": random.choice(["male", "Female"]),
                "mbti": random.choice(["ENTJ", "INTJ", "ENTP", "INTP"]),
                "country": random.choice(self.COUNTRIES),
                "proFession": entity_attributes.get("occupation", "Expert"),
                "interested_topics": ["Politics", "Economics", "Culture & Society"],
            }

        elif entity_type_lower in ["mediaoutlet", "socialmediaplatform"]:
            return {
                "bio": f"Official account for {entity_name}. News and updates.",
                "persona": f"{entity_name} is a media entity that reports news and facilitates public discourse. The account shares timely updates and engages with the audience on current events.",
                "age": 30,  # 机构虚拟年龄
                "gender": "other",  # 机构使用other
                "mbti": "ISTJ",  # 机构风格：严谨保守
                "country": "中国",
                "proFession": "Media",
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
                "age": 30,  # 机构虚拟年龄
                "gender": "other",  # 机构使用other
                "mbti": "ISTJ",  # 机构风格：严谨保守
                "country": "中国",
                "proFession": entity_type,
                "interested_topics": [
                    "Public Policy",
                    "Community",
                    "Official Announcements",
                ],
            }

        else:
            # Por deFecto人设
            return {
                "bio": entity_summary[:150]
                if entity_summary
                else f"{entity_type}: {entity_name}",
                "persona": entity_summary
                or f"{entity_name} is a {entity_type.lower()} participating in social discussions.",
                "age": random.randint(25, 50),
                "gender": random.choice(["male", "Female"]),
                "mbti": random.choice(self.MBTI_TYPES),
                "country": random.choice(self.COUNTRIES),
                "proFession": entity_type,
                "interested_topics": ["General", "Social Issues"],
            }

    def set_graph_id(self, graph_id: str):
        """ConfigurarGrafoID用于Zep检索"""
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
        LoteDesdeEntidadGenerarAgent Profile（SoporteParaleloGenerar）

        Args:
            entities: EntidadLista
            use_llm: Si使用LLMGenerar详细人设
            progress_callback: 进度CallbackFunción (current, total, message)
            graph_id: GrafoID，用于Zep检索ObtenerMás丰富上下文
            parallel_count: ParaleloGenerarCantidad，默认5
            realtime_output_path: 实时Escribir的Archivo路径（Si提供，每Generar一ElementosEntoncesEscribir一次）
            output_platform: 输出Plataforma格式 ("reddit" O "twitter")

        Returns:
            Agent ProfileLista
        """
        import concurrent.futures
        from threading import Lock

        # Configuracióngraph_id用于Zep检索
        if graph_id:
            self.graph_id = graph_id

        total = len(entities)
        profiles = [None] * total  # 预分配Lista保持Secuencial
        completed_count = [0]  # 使用Lista以便En闭包中Modificar
        lock = Lock()

        # 实时EscribirArchivo的辅助Función
        def save_profiles_realtime():
            """实时Guardar已Generar的 profiles HastaArchivo"""
            if not realtime_output_path:
                return

            with lock:
                # Filtrado出已Generar的 profiles
                existing_profiles = [p for p in profiles if p is not None]
                if not existing_profiles:
                    return

                try:
                    if output_platform == "reddit":
                        # Reddit JSON 格式
                        profiles_data = [
                            p.to_reddit_Format() for p in existing_profiles
                        ]
                        with open(realtime_output_path, "w", encoding="utf-8") as f:
                            json.dump(profiles_data, f, ensure_ascii=False, indent=2)
                    else:
                        # Twitter CSV 格式
                        import csv

                        profiles_data = [
                            p.to_twitter_Format() for p in existing_profiles
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
                    logger.warning(f"实时Guardar profiles Fallido: {e}")

        # Capture locale before spawning thread pool workers
        current_locale = get_locale()

        def generate_single_profile(idx: int, entity: EntityNode) -> tuple:
            """Generar单Elementosprofile的工作Función"""
            set_locale(current_locale)
            entity_type = entity.get_entity_type() or "Entity"

            try:
                profile = self.generate_profile_from_entity(
                    entity=entity, user_id=idx, use_llm=use_llm
                )

                # 实时输出Generar的人设HastaConsolaYLog
                self._print_generated_profile(entity.name, entity_type, profile)

                return idx, profile, None

            except Exception as e:
                logger.error(f"GenerarEntidad {entity.name} 的人设Fallido: {str(e)}")
                # Creación一Elementos基础profile
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

        logger.info(f"InicioParaleloGenerar {total} ElementosAgent人设（Paralelo数: {parallel_count}）...")
        print(f"\n{'=' * 60}")
        print(f"InicioGenerando perfiles de agentes - 共 {total} ElementosEntidad，Paralelo数: {parallel_count}")
        print(f"{'=' * 60}\n")

        # 使用HiloPoolParalelo执Fila
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=parallel_count
        ) as executor:
            # ConfirmarTodosTarea
            future_to_entity = {
                executor.submit(generate_single_profile, idx, entity): (idx, entity)
                for idx, entity in enumerate(entities)
            }

            # 收集结果
            for future in concurrent.futures.as_completed(future_to_entity):
                idx, entity = future_to_entity[future]
                entity_type = entity.get_entity_type() or "Entity"

                try:
                    result_idx, profile, error = future.result()
                    profiles[result_idx] = profile

                    with lock:
                        completed_count[0] += 1
                        current = completed_count[0]

                    # 实时EscribirArchivo
                    save_profiles_realtime()

                    if progress_callback:
                        progress_callback(
                            current,
                            total,
                            f"Completado {current}/{total}: {entity.name}（{entity_type}）",
                        )

                    if error:
                        logger.warning(
                            f"[{current}/{total}] {entity.name} 使用备用人设: {error}"
                        )
                    else:
                        logger.info(
                            f"[{current}/{total}] ÉxitoGenerar人设: {entity.name} ({entity_type})"
                        )

                except Exception as e:
                    logger.error(f"ProcesarEntidad {entity.name} 时发生Excepción: {str(e)}")
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
                    # 实时EscribirArchivo（Incluso siEs备用人设）
                    save_profiles_realtime()

        print(f"\n{'=' * 60}")
        print(f"人设GenerarCompletado！共Generar {len([p for p in profiles if p])} ElementosAgent")
        print(f"{'=' * 60}\n")

        return profiles

    def _print_generated_profile(
        self, entity_name: str, entity_type: str, profile: OasisAgentProfile
    ):
        """实时输出Generar的人设HastaConsola（完整Contenido，不截断）"""
        separator = "-" * 70

        # 构建完整输出Contenido（不截断）
        topics_str = (
            ", ".join(profile.interested_topics) if profile.interested_topics else "Ninguno"
        )

        output_lines = [
            f"\n{separator}",
            t("progress.profileGenerated", name=entity_name, type=entity_type),
            f"{separator}",
            f"用户名: {profile.user_name}",
            f"",
            f"【Biografía】",
            f"{profile.bio}",
            f"",
            f"【详细人设】",
            f"{profile.persona}",
            f"",
            f"【BásicoAtributo】",
            f"年龄: {profile.age} | 性别: {profile.gender} | MBTI: {profile.mbti}",
            f"职业: {profile.proFession} | 国家: {profile.country}",
            f"兴趣话题: {topics_str}",
            separator,
        ]

        output = "\n".join(output_lines)

        # Solo输出HastaConsola（避免重复，logger不再输出完整Contenido）
        print(output)

    def save_profiles(
        self,
        profiles: List[OasisAgentProfile],
        file_path: str,
        platform: str = "reddit",
    ):
        """
        GuardarProfileHastaArchivo（Basado enPlataformaSelección正确格式）

        OASISPlataforma格式Requisito：
        - Twitter: CSV格式
        - Reddit: JSON格式

        Args:
            profiles: ProfileLista
            file_path: Archivo路径
            platform: PlataformaTipo ("reddit" O "twitter")
        """
        if platform == "twitter":
            self._save_twitter_csv(profiles, file_path)
        else:
            self._save_reddit_json(profiles, file_path)

    def _save_twitter_csv(self, profiles: List[OasisAgentProfile], file_path: str):
        """
        GuardarTwitter Profile为CSV格式（符合OASIS官方Requisito）

        OASIS TwitterRequisito的CSVCampo：
        - user_id: 用户ID（Basado enCSVSecuencialDesde0Inicio）
        - name: 用户真实姓名
        - username: Sistema中的用户名
        - user_char: 详细人设Descripción（注入HastaLLMSistema提示中，指导AgentFila为）
        - description: 简短的PúblicoBiografía（MostrarEn用户资料页面）

        user_char vs description 区别：
        - user_char: 内部使用，LLMSistema提示，决定AgentCómo思考YFila动
        - description: 外部Mostrar，Otro用户可见的Biografía
        """
        import csv

        # 确保ArchivoExtender名Es.csv
        if not file_path.endswith(".csv"):
            file_path = file_path.replace(".json", ".csv")

        with open(file_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            # EscrituraOASISRequisito的Encabezado de tabla
            headers = ["user_id", "name", "username", "user_char", "description"]
            writer.writerow(headers)

            # EscrituraDatosFila
            for idx, profile in enumerate(profiles):
                # user_char: 完整人设（bio + persona），用于LLMSistema提示
                user_char = profile.bio
                if profile.persona and profile.persona != profile.bio:
                    user_char = f"{profile.bio} {profile.persona}"
                # Manejar换Fila符（CSV中用空格替代）
                user_char = user_char.replace("\n", " ").replace("\r", " ")

                # description: 简短Biografía，用于外部Mostrar
                description = profile.bio.replace("\n", " ").replace("\r", " ")

                row = [
                    idx,  # user_id: Desde0Inicio的SecuencialID
                    profile.name,  # name: 真实姓名
                    profile.user_name,  # username: 用户名
                    user_char,  # user_char: 完整人设（内部LLM使用）
                    description,  # description: 简短Biografía（外部Mostrar）
                ]
                writer.writerow(row)

        logger.info(
            f"已Guardar {len(profiles)} ElementosTwitter ProfileHasta {file_path} (OASIS CSV格式)"
        )

    def _Normalize_gender(self, gender: Optional[str]) -> str:
        """
        EstandarizargenderCampo为OASISRequisito的英文格式

        OASISRequisito: male, Female, other
        """
        if not gender:
            return "other"

        gender_lower = gender.lower().strip()

        # 中文Mapeo
        gender_map = {
            "Masculino": "male",
            "Femenino": "Female",
            "机构": "other",
            "Otro": "other",
            # 英文已Tener
            "male": "male",
            "Female": "Female",
            "other": "other",
        }

        return gender_map.get(gender_lower, "other")

    def _save_reddit_json(self, profiles: List[OasisAgentProfile], file_path: str):
        """
        GuardarReddit Profile为JSON格式

        使用Con to_reddit_Format() 一致的格式，确保 OASIS 能正确Leer。
        DebeContiene user_id Campo，这Es OASIS agent_graph.get_agent() 匹配的关Clave！

        RequeridoCampo：
        - user_id: 用户ID（整数，用于匹配 initial_posts 中的 poster_agent_id）
        - username: 用户名
        - name: MostrarNombre
        - bio: Biografía
        - persona: 详细人设
        - age: 年龄（整数）
        - gender: "male", "Female", O "other"
        - mbti: MBTITipo
        - country: 国家
        """
        data = []
        for idx, profile in enumerate(profiles):
            # 使用Con to_reddit_Format() 一致的格式
            item = {
                "user_id": profile.user_id
                if profile.user_id is not None
                else idx,  # 关Clave：DebeContiene user_id
                "username": profile.user_name,
                "name": profile.name,
                "bio": profile.bio[:150] if profile.bio else f"{profile.name}",
                "persona": profile.persona
                or f"{profile.name} is a participant in social discussions.",
                "karma": profile.karma if profile.karma else 1000,
                "created_at": profile.created_at,
                # OASISRequeridoCampo - 确保TodosTener默认Valor
                "age": profile.age if profile.age else 30,
                "gender": self._Normalize_gender(profile.gender),
                "mbti": profile.mbti if profile.mbti else "ISTJ",
                "country": profile.country if profile.country else "中国",
            }

            # OpcionalCampo
            if profile.proFession:
                item["proFession"] = profile.proFession
            if profile.interested_topics:
                item["interested_topics"] = profile.interested_topics

            data.append(item)

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        logger.info(
            f"已Guardar {len(profiles)} ElementosReddit ProfileHasta {file_path} (JSON格式，Contieneuser_idCampo)"
        )

    # 保留旧Método名Como别名，保持Hacia后兼容
    def save_profiles_to_json(
        self,
        profiles: List[OasisAgentProfile],
        file_path: str,
        platform: str = "reddit",
    ):
        """[已废弃] 请使用 save_profiles() Método"""
        logger.warning("save_profiles_to_json已废弃，请使用save_profilesMétodo")
        self.save_profiles(profiles, file_path, platform)
