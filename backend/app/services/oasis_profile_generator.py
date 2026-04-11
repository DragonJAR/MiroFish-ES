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
        self.zep_api_key = zep_api_key or Config.ZEP_API_KEY
        self.graph_id = graph_id

        if not self.api_key:
            raise ValueError("LLM_API_KEY no configurada")

        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        logger.info(f"OasisProfileGenerator inicializado con modelo {self.model_name}")

    def generate_profile_from_entity(
        self,
        entity: "EntityNode",
        user_id: int,
        use_llm: bool = True,
    ) -> OasisAgentProfile:
        """
        Generar un solo Agent Profile desde una entidad.

        Args:
            entity: EntityNode con los datos de la entidad
            user_id: ID de usuario para el profile
            use_llm: Si usar LLM para generación (sino usar reglas)

        Returns:
            OasisAgentProfile
        """
        entity_name = entity.name
        entity_type = entity.get_entity_type() or "Entity"
        entity_summary = entity.summary or ""
        entity_attributes = entity.attributes or {}

        # Determinar si es tipo individual o grupal
        entity_type_lower = entity_type.lower()
        is_individual = entity_type_lower not in self.GROUP_ENTITY_TYPES

        # Intentar generar múltiples veces hasta éxito o alcanzar máximo de reintentos
        max_attempts = 3
        last_error = None

        for attempt in range(max_attempts):
            try:
                # Construir prompt según tipo de entidad
                if is_individual:
                    prompt = self._build_individual_persona_prompt(
                        entity_name=entity_name,
                        entity_type=entity_type,
                        entity_summary=entity_summary,
                        entity_attributes=entity_attributes,
                        context="",
                    )
                else:
                    prompt = self._build_group_persona_prompt(
                        entity_name=entity_name,
                        entity_type=entity_type,
                        entity_summary=entity_summary,
                        entity_attributes=entity_attributes,
                        context="",
                    )

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
                    temperature=0.7 - (attempt * 0.1),
                )

                content = response.choices[0].message.content

                # Verificación si fue truncado
                finish_reason = response.choices[0].finish_reason
                if finish_reason == "length":
                    logger.warning(
                        f"LLM salida truncada (intent {attempt + 1}), intentando corrección..."
                    )
                    content = self._fix_truncated_json(content)

                # Intentar analizar JSON
                try:
                    result = json.loads(content)

                    # Verificación de campos requeridos
                    if "bio" not in result or not result["bio"]:
                        result["bio"] = (
                            entity_summary[:200]
                            if entity_summary
                            else f"{entity_type}: {entity_name}"
                        )
                    if "persona" not in result or not result["persona"]:
                        result["persona"] = (
                            entity_summary
                            or f"{entity_name} es un elemento de {entity_type}."
                        )

                    # Construir OasisAgentProfile
                    profile = OasisAgentProfile(
                        user_id=user_id,
                        user_name=self._generate_username(entity_name),
                        name=entity_name,
                        bio=result.get("bio", ""),
                        persona=result.get("persona", ""),
                        age=result.get("age"),
                        gender=result.get("gender"),
                        mbti=result.get("mbti"),
                        country=result.get("country"),
                        proFession=result.get("proFession"),
                        interested_topics=result.get("interested_topics", []),
                        source_entity_uuid=entity.uuid,
                        source_entity_type=entity_type,
                    )
                    return profile

                except json.JSONDecodeError as je:
                    logger.warning(
                        f"JSON análisis fallido (intent {attempt + 1}): {str(je)[:80]}"
                    )

                    # Intentar corrección JSON
                    fixed = self._try_fix_json(
                        content, entity_name, entity_type, entity_summary
                    )
                    if fixed and fixed.get("_fixed"):
                        del fixed["_fixed"]
                        profile = OasisAgentProfile(
                            user_id=user_id,
                            user_name=self._generate_username(entity_name),
                            name=entity_name,
                            bio=fixed.get("bio", ""),
                            persona=fixed.get("persona", ""),
                            age=fixed.get("age"),
                            gender=fixed.get("gender"),
                            mbti=fixed.get("mbti"),
                            country=fixed.get("country"),
                            proFession=fixed.get("proFession"),
                            interested_topics=fixed.get("interested_topics", []),
                            source_entity_uuid=entity.uuid,
                            source_entity_type=entity_type,
                        )
                        return profile

                    last_error = je

            except Exception as e:
                logger.warning(
                    f"LLM llamada fallida (intent {attempt + 1}): {str(e)[:80]}"
                )
                last_error = e
                import time

                time.sleep(1 * (attempt + 1))

        # Fallback a generación por reglas
        logger.warning(
            f"LLM generación de personificación fallida ({max_attempts} intentos): {last_error}, usando reglas"
        )
        rule_result = self._generate_profile_rule_based(
            entity_name, entity_type, entity_summary, entity_attributes
        )
        return OasisAgentProfile(
            user_id=user_id,
            user_name=self._generate_username(entity_name),
            name=entity_name,
            bio=rule_result.get("bio", ""),
            persona=rule_result.get("persona", ""),
            age=rule_result.get("age"),
            gender=rule_result.get("gender"),
            mbti=rule_result.get("mbti"),
            country=rule_result.get("country"),
            proFession=rule_result.get("proFession"),
            interested_topics=rule_result.get("interested_topics", []),
            source_entity_uuid=entity.uuid,
            source_entity_type=entity_type,
        )

    def _fix_truncated_json(self, content: str) -> str:
        """Corregir JSON truncado (salida truncada por límite de max_tokens)"""
        import re

        # Si JSON está truncado, intentar cerrarlo
        content = content.strip()

        # Calcular paréntesis sin cerrar
        open_braces = content.count("{") - content.count("}")
        open_brackets = content.count("[") - content.count("]")

        # Inspección si hay cadenas sin cerrar
        # Verificación simple: si después de la última comilla de elemento no hay coma o paréntesis de cierre, es posible que la cadena esté truncada
        if content and content[-1] not in '",}]':
            # Intentar cerrar la cadena
            content += '"'

        # Cerrar paréntesis
        content += "]" * open_brackets
        content += "}" * open_braces

        return content

    def _try_fix_json(
        self, content: str, entity_name: str, entity_type: str, entity_summary: str = ""
    ) -> Dict[str, Any]:
        """Intentar corregir JSON dañado"""
        import re

        # 1. Primero intentar corregir situación truncada
        content = self._fix_truncated_json(content)

        # 2. Intentar extraer parte JSON
        json_match = re.search(r"\{[\s\S]*\}", content)
        if json_match:
            json_str = json_match.group()

            # 3. Procesar problema de saltos de línea en cadenas
            # Encontrar todos los valores de cadena y reemplazar saltos de línea dentro
            def fix_string_newlines(match):
                s = match.group(0)
                # Reemplazar saltos de línea reales dentro de cadenas por espacios
                s = s.replace("\n", " ").replace("\r", " ")
                # Reemplazar espacios excesivos
                s = re.sub(r"\s+", " ", s)
                return s

            # Coincidir con valores de cadena JSON
            json_str = re.sub(
                r'"[^"\\]*(?:\\.[^"\\]*)*"', fix_string_newlines, json_str
            )

            # 4. Intentar analizar
            try:
                result = json.loads(json_str)
                result["_fixed"] = True
                return result
            except json.JSONDecodeError as e:
                # 5. Si todavía falla, intentar corrección más agresiva
                try:
                    # Eliminar todos los caracteres de control
                    json_str = re.sub(r"[\x00-\x1f\x7f-\x9f]", " ", json_str)
                    # Reemplazar todos los espacios en blanco continuos
                    json_str = re.sub(r"\s+", " ", json_str)
                    result = json.loads(json_str)
                    result["_fixed"] = True
                    return result
                except:
                    pass

        # 6. Intentar extraer información parcial desde contenido
        bio_match = re.search(r'"bio"\s*:\s*"([^"]*)"', content)
        persona_match = re.search(
            r'"persona"\s*:\s*"([^"]*)', content
        )  # Posible truncado

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
            else (entity_summary or f"{entity_name} es un elemento de {entity_type}.")
        )

        # Si se extrajo contenido con significado, marcar como corregido
        if bio_match or persona_match:
            logger.info(f"Extraída información parcial desde JSON dañado")
            return {"bio": bio, "persona": persona, "_fixed": True}

        # 7. Falló completamente, volver a estructura básica
        logger.warning(f"Corrección JSON fallida, volver a estructura básica")
        return {
            "bio": entity_summary[:200]
            if entity_summary
            else f"{entity_type}: {entity_name}",
            "persona": entity_summary
            or f"{entity_name} es un elemento de {entity_type}.",
        }

    def _get_system_prompt(self, is_individual: bool) -> str:
        """Obtener prompt del sistema"""
        base_prompt = "Eres experto en generación de perfiles de usuarios de redes sociales. Generar personificaciones detalladas y realistas para simulación de opinión pública, restaurando en gran medida las situaciones ya existentes. Debes devolver formato JSON válido, todos los valores de cadena no pueden contener saltos de línea sin escapar."
        return f"{base_prompt}\n\n{get_language_instruction()}"

    def _build_individual_persona_prompt(
        self,
        entity_name: str,
        entity_type: str,
        entity_summary: str,
        entity_attributes: Dict[str, Any],
        context: str,
    ) -> str:
        """Construir prompt detallado de personificación para entidad individual"""

        attrs_str = (
            json.dumps(entity_attributes, ensure_ascii=False)
            if entity_attributes
            else "Ninguno"
        )
        context_str = context[:3000] if context else "Ningún contexto adicional"

        return f"""Generar personificación detallada de usuario de redes sociales para la entidad, restaurando en gran medida las situaciones ya existentes.

Nombre de entidad: {entity_name}
Tipo de entidad: {entity_type}
Resumen de entidad: {entity_summary}
Atributo de entidad: {attrs_str}

Información contextual:
{context_str}

Por favor generar JSON que contenga los siguientes campos:

1. bio: Biografía de redes sociales, 200 caracteres
2. persona: Descripción detallada de personificación (texto plano de 2000 caracteres), debe incluir:
    - Información básica (edad, profesión, antecedentes educativos, ubicación)
    - Antecedentes del personaje (experiencias importantes, asociaciones con eventos, relaciones sociales)
    - Características de personalidad (tipo MBTI, rasgos nucleares de personalidad, estilo de expresión emocional)
    - Comportamiento en redes sociales (frecuencia de publicación, preferencias de contenido, estilo de interacción, características del lenguaje)
    - Puntos de vista (actitud hacia temas, contenido que podría provocar ira/movimiento)
    - Características únicas (frases recurrentes, experiencias especiales, pasatiempos personales)
    - Memorias personales (parte importante de la personificación, describir estas asociaciones específicas de eventos, y las reacciones de acción ya ocurridas en estos eventos específicos)
3. age: Número de edad (debe ser entero)
4. gender: Sexo, debe ser en inglés: "male" o "Female"
5. mbti: Tipo MBTI (ej. INTJ, ENFP, etc.)
6. country: País (usar español, ej. "China")
7. proFession: Profesión
8. interested_topics: Arreglo de temas de interés

Importante:
- Todos los valores de campos deben ser cadenas o números, no usar saltos de línea
- persona debe ser una descripción textual coherente
- {get_language_instruction()} (el campo gender debe usar inglés male/Female)
- El contenido debe mantenerse consistente con la información de entidad
- age debe ser entero válido, gender debe ser "male" o "Female"
"""

    def _build_group_persona_prompt(
        self,
        entity_name: str,
        entity_type: str,
        entity_summary: str,
        entity_attributes: Dict[str, Any],
        context: str,
    ) -> str:
        """Construir prompt detallado de personificación para entidad institucional/grupal"""

        attrs_str = (
            json.dumps(entity_attributes, ensure_ascii=False)
            if entity_attributes
            else "Ninguno"
        )
        context_str = context[:3000] if context else "Ningún contexto adicional"

        return f"""Generar configuración detallada de cuenta de redes sociales para entidad institucional/grupal, restaurando en gran medida las situaciones ya existentes.

Nombre de entidad: {entity_name}
Tipo de entidad: {entity_type}
Resumen de entidad: {entity_summary}
Atributo de entidad: {attrs_str}

Información contextual:
{context_str}

Por favor generar JSON que contenga los siguientes campos:

1. bio: Biografía de cuenta oficial, 200 caracteres, profesional y apropiada
2. persona: Descripción detallada de configuración de cuenta (texto plano de 2000 caracteres), debe incluir:
    - Información básica institucional (nombre formal, naturaleza de institución, antecedentes de establecimiento, función principal)
    - Posicionamiento de cuenta (tipo de cuenta, audiencia objetivo, función nuclear)
    - Estilo de expresión (características del lenguaje, expresiones comunes, temas tabú)
    - Características de contenido publicado (tipo de contenido, frecuencia de publicación, período de actividad)
    - Actitud frente a temas (posición oficial sobre temas nucleares, manera de manejar controversias)
    - Declaraciones especiales (perfil de grupo representado, hábitos operativos)
    - Memoria institucional (parte importante de la personificación institucional, describir estas asociaciones específicas de eventos institucionales, y las reacciones de acción ya ocurridas en estos eventos específicos)
3. age: Fijo en 30 (edad virtual para cuenta institucional)
4. gender: Fijo en "other" (cuenta institucional usa other para indicar no-individual)
5. mbti: Tipo MBTI, usado para describir estilo de cuenta, ej. ISTJ representa riguroso y conservador
6. country: País (usar español, ej. "China")
7. proFession: Descripción de función institucional
8. interested_topics: Arreglo de áreas de seguimiento

Importante:
- Todos los valores de campos deben ser cadenas o números, no se permiten valores nulos
- persona debe ser una descripción textual coherente, no usar saltos de línea
- {get_language_instruction()} (el campo gender debe usar inglés "other")
- age debe ser entero 30, gender debe ser cadena "other"
- La expresión de la cuenta institucional debe corresponder con su posicionamiento de identidad"""

    def _generate_profile_rule_based(
        self,
        entity_name: str,
        entity_type: str,
        entity_summary: str,
        entity_attributes: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Usar reglas para generar personificación básica"""

        # Basado en tipo de entidad generar diferentes personificaciones
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
                "age": 30,  # Edad virtual institucional
                "gender": "other",  # Institución usa other
                "mbti": "ISTJ",  # Estilo institucional: riguroso y conservador
                "country": "China",
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
                "age": 30,  # Edad virtual institucional
                "gender": "other",  # Institución usa other
                "mbti": "ISTJ",  # Estilo institucional: riguroso y conservador
                "country": "China",
                "proFession": entity_type,
                "interested_topics": [
                    "Public Policy",
                    "Community",
                    "Official Announcements",
                ],
            }

        else:
            # Personificación por defecto
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
        """Configurar graph_id para recuperación Zep"""
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
        Generar Agent Profiles desde entidades en lote (soporta generación paralela)

        Args:
            entities: Lista de entidades
            use_llm: Si usar LLM para generar personificación detallada
            progress_callback: Función callback de progreso (current, total, message)
            graph_id: ID de grafo, para recuperación Zep obtener más contexto rico
            parallel_count: Cantidad de generación paralela, por defecto 5
            realtime_output_path: Ruta de archivo para escribir en tiempo real (si se proporciona, escribir cada vez que se genera uno)
            output_platform: Formato de plataforma de salida ("reddit" o "twitter")

        Returns:
            Lista de Agent Profile
        """
        import concurrent.futures
        from threading import Lock

        # Configurar graph_id para recuperación Zep
        if graph_id:
            self.graph_id = graph_id

        total = len(entities)
        profiles = [None] * total  # Preasignar lista para mantener secuencia
        completed_count = [0]  # Usar lista para modificar en cierre
        lock = Lock()

        # Función auxiliar para escribir archivo en tiempo real
        def save_profiles_realtime():
            """Guardar en tiempo real los profiles generados en archivo"""
            if not realtime_output_path:
                return

            with lock:
                # Filtrar los profiles generados
                existing_profiles = [p for p in profiles if p is not None]
                if not existing_profiles:
                    return

                try:
                    if output_platform == "reddit":
                        # Formato Reddit JSON
                        profiles_data = [
                            p.to_reddit_Format() for p in existing_profiles
                        ]
                        with open(realtime_output_path, "w", encoding="utf-8") as f:
                            json.dump(profiles_data, f, ensure_ascii=False, indent=2)
                    else:
                        # Formato CSV de Twitter
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
                    logger.warning(f"Guardar en tiempo real profiles fallido: {e}")

        # Capturar locale antes de generar workers de thread pool
        current_locale = get_locale()

        def generate_single_profile(idx: int, entity: EntityNode) -> tuple:
            """Función de trabajo para generar un solo profile"""
            set_locale(current_locale)
            entity_type = entity.get_entity_type() or "Entity"

            try:
                profile = self.generate_profile_from_entity(
                    entity=entity, user_id=idx, use_llm=use_llm
                )

                # Salida en tiempo real de personificación generada a consola y log
                self._print_generated_profile(entity.name, entity_type, profile)

                return idx, profile, None

            except Exception as e:
                logger.error(
                    f"Generación de personificación de entidad {entity.name} fallida: {str(e)}"
                )
                # Crear un profile básico de respaldo
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
            f"Inicio de generación paralela de {total} perfiles de agente (cuenta paralela: {parallel_count})..."
        )
        print(f"\n{'=' * 60}")
        print(
            f"InicioGenerando perfiles de agentes - total {total} entidades, cuenta paralela: {parallel_count}"
        )
        print(f"{'=' * 60}\n")

        # Usar ThreadPool para ejecución paralela
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=parallel_count
        ) as executor:
            # Confirmar todas las tareas
            future_to_entity = {
                executor.submit(generate_single_profile, idx, entity): (idx, entity)
                for idx, entity in enumerate(entities)
            }

            # Recopilar resultados
            for future in concurrent.futures.as_completed(future_to_entity):
                idx, entity = future_to_entity[future]
                entity_type = entity.get_entity_type() or "Entity"

                try:
                    result_idx, profile, error = future.result()
                    profiles[result_idx] = profile

                    with lock:
                        completed_count[0] += 1
                        current = completed_count[0]

                    # Escribir archivo en tiempo real
                    save_profiles_realtime()

                    if progress_callback:
                        progress_callback(
                            current,
                            total,
                            f"Completado {current}/{total}: {entity.name} ({entity_type})",
                        )

                    if error:
                        logger.warning(
                            f"[{current}/{total}] {entity.name} usando personificación de respaldo: {error}"
                        )
                    else:
                        logger.info(
                            f"[{current}/{total}] Éxito generación de personificación: {entity.name} ({entity_type})"
                        )

                except Exception as e:
                    logger.error(
                        f"Procesar entidad {entity.name} ocurrió excepción: {str(e)}"
                    )
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
                    # Escribir archivo en tiempo real (incluso si es personificación de respaldo)
                    save_profiles_realtime()

        print(f"\n{'=' * 60}")
        print(
            f"¡Generación de personificación completada! Total {len([p for p in profiles if p])} agentes generados"
        )
        print(f"{'=' * 60}\n")

        return profiles

    def _print_generated_profile(
        self, entity_name: str, entity_type: str, profile: OasisAgentProfile
    ):
        """Salida en tiempo real de personificación generada a consola (contenido completo, no truncar)"""
        separator = "-" * 70

        # Construir contenido de salida completo (no truncar)
        topics_str = (
            ", ".join(profile.interested_topics)
            if profile.interested_topics
            else "Ninguno"
        )

        output_lines = [
            f"\n{separator}",
            t("progress.profileGenerated", name=entity_name, type=entity_type),
            f"{separator}",
            f"Usuario: {profile.user_name}",
            f"",
            f"[Biografía]",
            f"{profile.bio}",
            f"",
            f"[Personificación detallada]",
            f"{profile.persona}",
            f"",
            f"[Atributos básicos]",
            f"Edad: {profile.age} | Sexo: {profile.gender} | MBTI: {profile.mbti}",
            f"Profesión: {profile.proFession} | País: {profile.country}",
            f"Temas de interés: {topics_str}",
            separator,
        ]

        output = "\n".join(output_lines)

        # Solo salida a consola (evitar duplicación, logger no vuelve a emitir contenido completo)
        print(output)

    def save_profiles(
        self,
        profiles: List[OasisAgentProfile],
        file_path: str,
        platform: str = "reddit",
    ):
        """
        Guardar perfiles en archivo (basado en selección de plataforma formato correcto)

        Requisitos de formato de plataforma OASIS:
        - Twitter: Formato CSV
        - Reddit: Formato JSON

        Args:
            profiles: Lista de perfiles
            file_path: Ruta de archivo
            platform: Tipo de plataforma ("reddit" o "twitter")
        """
        if platform == "twitter":
            self._save_twitter_csv(profiles, file_path)
        else:
            self._save_reddit_json(profiles, file_path)

    def _save_twitter_csv(self, profiles: List[OasisAgentProfile], file_path: str):
        """
        Guardar Perfiles de Twitter como formato CSV (cumple requisito oficial de OASIS)

        Campos CSV requeridos por OASIS Twitter:
        - user_id: ID de usuario (basado en secuencia CSV comenzando desde 0)
        - name: Nombre real del usuario
        - username: Nombre de usuario en el sistema
        - user_char: Descripción detallada de personificación (inyectada en prompt del sistema LLM, guía comportamiento del agente)
        - description: Biografía pública corta (mostrada en página de perfil del usuario)

        Diferencia entre user_char y description:
        - user_char: uso interno, prompt del sistema LLM, determina cómo piensa y actúa el agente
        - description: visualización externa, biografía visible para otros usuarios
        """
        import csv

        # Asegurar que nombre de archivo termine en .csv
        if not file_path.endswith(".csv"):
            file_path = file_path.replace(".json", ".csv")

        with open(file_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            # Escribir encabezado de tabla requerido por OASIS
            headers = ["user_id", "name", "username", "user_char", "description"]
            writer.writerow(headers)

            # Escribir filas de datos
            for idx, profile in enumerate(profiles):
                # user_char: personificación completa (bio + persona), para prompt del sistema LLM
                user_char = profile.bio
                if profile.persona and profile.persona != profile.bio:
                    user_char = f"{profile.bio} {profile.persona}"
                # Manejar saltos de línea (reemplazar con espacios en CSV)
                user_char = user_char.replace("\n", " ").replace("\r", " ")

                # description: biografía corta, para visualización externa
                description = profile.bio.replace("\n", " ").replace("\r", " ")

                row = [
                    idx,  # user_id: ID secuencial comenzando desde 0
                    profile.name,  # name: nombre real
                    profile.user_name,  # username: nombre de usuario
                    user_char,  # user_char: personificación completa (uso interno LLM)
                    description,  # description: biografía corta (visualización externa)
                ]
                writer.writerow(row)

        logger.info(
            f"Guardados {len(profiles)} perfiles de Twitter en {file_path} (formato CSV OASIS)"
        )

    def _Normalize_gender(self, gender: Optional[str]) -> str:
        """
        Estandarizar gender campo al formato en inglés requerido por OASIS

        Requisito de OASIS: male, Female, other
        """
        if not gender:
            return "other"

        gender_lower = gender.lower().strip()

        # Mapeo de género
        gender_map = {
            "Masculino": "male",
            "Femenino": "Female",
            "institución": "other",
            "Otro": "other",
            # Inglés ya presente
            "male": "male",
            "Female": "Female",
            "other": "other",
        }

        return gender_map.get(gender_lower, "other")

    def _save_reddit_json(self, profiles: List[OasisAgentProfile], file_path: str):
        """
        Guardar Perfiles de Reddit como formato JSON

        Usar formato consistente con to_reddit_Format(), asegurar que OASIS pueda leer correctamente.
        Debe contener campo user_id, ¡esta es la clave para coincidencia con OASIS agent_graph.get_agent()!

        Campos requeridos:
        - user_id: ID de usuario (entero, para coincidir con poster_agent_id en initial_posts)
        - username: Nombre de usuario
        - name: Nombre para mostrar
        - bio: Biografía
        - persona: Personificación detallada
        - age: Edad (entero)
        - gender: "male", "Female", o "other"
        - mbti: Tipo MBTI
        - country: País
        """
        data = []
        for idx, profile in enumerate(profiles):
            # Usar formato consistente con to_reddit_Format()
            item = {
                "user_id": profile.user_id
                if profile.user_id is not None
                else idx,  # Clave: debe contener user_id
                "username": profile.user_name,
                "name": profile.name,
                "bio": profile.bio[:150] if profile.bio else f"{profile.name}",
                "persona": profile.persona
                or f"{profile.name} is a participant in social discussions.",
                "karma": profile.karma if profile.karma else 1000,
                "created_at": profile.created_at,
                # Campo requerido por OASIS - asegurar que todos tengan valor por defecto
                "age": profile.age if profile.age else 30,
                "gender": self._Normalize_gender(profile.gender),
                "mbti": profile.mbti if profile.mbti else "ISTJ",
                "country": profile.country if profile.country else "China",
            }

            # Campo opcional
            if profile.proFession:
                item["proFession"] = profile.proFession
            if profile.interested_topics:
                item["interested_topics"] = profile.interested_topics

            data.append(item)

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        logger.info(
            f"Guardados {len(profiles)} perfiles de Reddit en {file_path} (formato JSON, contiene campo user_id)"
        )

    # Mantener nombre de método antiguo como alias, mantener compatibilidad hacia atrás
    def save_profiles_to_json(
        self,
        profiles: List[OasisAgentProfile],
        file_path: str,
        platform: str = "reddit",
    ):
        """[Obsoleto] Por favor usar método save_profiles()"""
        logger.warning(
            "save_profiles_to_json obsoleto, por favor usar método save_profiles"
        )
        self.save_profiles(profiles, file_path, platform)
