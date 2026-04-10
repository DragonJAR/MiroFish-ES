"""
Generación de ontologíaServicio
Interfaz1：Análisis de textoContenido，Generar tipo de entidad y relación adecuado para simulación social
"""

import json
import logging
import re
from typing import Dict, Any, List, Optional
from ..utils.llm_client import LLMClient
from ..utils.locale import get_language_instruction

logger = logging.getLogger(__name__)


def _to_pascal_case(name: str) -> str:
    """Convertir nombres de cualquier formato a PascalCase（e.g., 'works_for' -> 'WorksFor', 'person' -> 'Person'）"""
    # Dividir por caracteres no alfanuméricos
    parts = re.split(r"[^a-zA-Z0-9]+", name)
    # Luego dividir por límites de camelCase（e.g., 'camelCase' -> ['camel', 'Case']）
    words = []
    for part in parts:
        words.extend(re.sub(r"([a-z])([A-Z])", r"\1_\2", part).split("_"))
    # Cada palabra con la primera letra en mayúscula, filtrar cadenas vacías
    result = "".join(word.capitalize() for word in words if word)
    return result if result else "Unknown"


# Generación de ontología del sistemaPrompt
ONTOLOGY_SYSTEM_PROMPT = """Eres un experto en diseño de ontología de grafos de conocimientos profesionales. Tu tarea es analizar el contenido de texto y los requisitos de simulación dados, diseñando tipos de entidad y relación adecuados para **simulación de opinión pública en redes sociales**.

**IMPORTANTE：Debes generar datos en formato JSON válido, sin generar ningún otro contenido.**

## Contexto de la tarea

Estamos construyendo un **sistema de simulación de opinión pública en redes sociales**。En este sistema：
- Cada entidad representa una "cuenta" o "sujeto" que puede emitir voz, interactuar y difundir información en las redes sociales
- Las entidades se influencian mutuamente, reenvían, comentan y responden entre sí
- Necesitamos simular las reacciones de todas las partes y las rutas de difusión de información en eventos de opinión pública

Por lo tanto，**las entidades deben ser sujetos que existen realmente en el mundo real, capaces de emitir voz e interactuar en las redes sociales**：

**Pueden ser**：
- Personas concretas（figuras públicas, personas relacionadas con eventos, líderes de opinión, expertos, académicos, personas comunes）
- Compañías, negocios（incluyendo sus cuentas oficiales）
- Organizaciones e instituciones（universidades, asociaciones, ONGs, sindicatos, etc.）
- Departamentos gubernamentales, organismos reguladores
- Medios de comunicación（periódicos, canales de televisión, medios de comunicación auto-gestionados, sitios web）
- Plataformas de redes sociales mismas
- Representantes de grupos específicos（asociaciones de exalumnos, clubes de fans, grupos de defensa de derechos, etc.）

**No pueden ser**：
- Conceptos abstractos（como "opinión pública", "emoción", "tendencia"）
- Temas/temáticas（como "integridad académica", "reforma educativa"）
- Puntos de vista/actitudes（como "partidarios", "opositores"）

## Formato de salida

Por favor, genera en formato JSON con la siguiente estructura：

```json
{
    "entity_types": [
        {
            "name": "Nombre del tipo de entidad（inglés, PascalCase）",
            "description": "Descripción breve（inglés, no más de 100 caracteres）",
            "attributes": [
                {
                    "name": "Nombre del atributo（inglés, snake_case）",
                    "type": "text",
                    "description": "Descripción del atributo"
                }
            ],
            "examples": ["Ejemplo de entidad 1", "Ejemplo de entidad 2"]
        }
    ],
    "edge_types": [
        {
            "name": "Nombre del tipo de relación（inglés, UPPER_SNAKE_CASE）",
            "description": "Descripción breve（inglés, no más de 100 caracteres）",
            "source_targets": [
                {"source": "Tipo de entidad origen", "target": "Tipo de entidad destino"}
            ],
            "attributes": []
        }
    ],
    "analysis_summary": "Análisis breve del contenido de texto para explicar"
}
```

## Guía de diseño（¡extremadamente importante!）

### 1. Diseño de tipos de entidad - Debe seguir estrictamente

**Requisito de cantidad：Debe tener exactamente 10 tipos de entidad**

**Requisito de estructura jerárquica（Debe incluir simultáneamente tipos específicos y tipos de respaldo）**：

Tus 10 tipos de entidad deben incluir las siguientes jerarquías：

A. **Tipos de respaldo（Debe incluir, colocar en los últimos 2 elementos de la lista）**：
   - `Person`: Tipo de respaldo para cualquier persona natural. Cuando una persona no pertenece a otros tipos de persona más específicos, clasificar en esta categoría。
   - `Organization`: Tipo de respaldo para cualquier organización. Cuando una organización no pertenece a otros tipos de organización más específicos, clasificar en esta categoría。

B. **Tipos específicos（8 elementos, diseñados basados en el contenido de texto）**：
   - Diseñar tipos más específicos dirigidos a los roles principales que aparecen en el texto
   - Por ejemplo：si el texto involucra eventos académicos, puede tener `Student`, `Professor`, `University`
   - Por ejemplo：si el texto involucra eventos comerciales, puede tener `Company`, `CEO`, `Employee`

**Por qué necesitamos tipos de respaldo**：
- El texto incluirá varias personas, como "profesor de escuela primaria", "persona anónima", "cierto usuario de red"
- Si no hay un tipo especializado para coincidir, deberían ser clasificados en `Person`
- De manera similar, pequeñas organizaciones, grupos temporales, etc. deberían ser clasificados en `Organization`

**Principios de diseño de tipos específicos**：
- Identificar en el texto los tipos de personajes que aparecen con alta frecuencia o son clave
- Cada tipo específico debe tener límites claros para evitar superposiciones
- La descripción debe explicar claramente la diferencia entre este tipo y los tipos de respaldo

### 2. Diseño de tipos de relación

- Cantidad：6-10 elementos
- Las relaciones deben reflejar conexiones reales en la interacción en redes sociales
- Asegurar que los source_targets de las relaciones cubran los tipos de entidad que definiste

### 3. Diseño de atributos

- 1-3 atributos clave por cada tipo de entidad
- **Nota**：Los nombres de atributos no pueden usar `name`, `uuid`, `group_id`, `created_at`, `summary`（Estos son palabras reservadas del sistema）
- Recomendación de uso：`full_name`, `title`, `role`, `position`, `location`, `description`, etc.

## Referencia de tipos de entidad

**Categoría de personas（específicos）**：
- Student: Estudiante
- Professor: Profesor/académico
- Journalist: Periodista
- Celebrity: Celebridad/influencer
- Executive: Ejecutivo de alto nivel
- Official: Funcionario gubernamental
- Lawyer: Abogado
- Doctor: Médico

**Categoría de personas（respaldo）**：
- Person: Cualquier persona natural（usar cuando no pertenece a los tipos específicos anteriores）

**Categoría de organización（específicos）**：
- University: Institución de educación superior
- Company: Compañía/negocio
- GovernmentAgency: Institución gubernamental
- MediaOutlet: Medios de comunicación
- Hospital: Hospital
- School: Escuela primaria/secundaria
- NGO: Organización no gubernamental

**Categoría de organización（respaldo）**：
- Organization: Cualquier organización（usar cuando no pertenece a los tipos específicos anteriores）

## Referencia de tipos de relación

- WORKS_FOR: Trabaja en
- STUDIES_AT: Estudia en
- AFFILIATED_WITH: Afiliado con
- REPRESENTS: Representa
- REGULATES: Regula
- REPORTS_ON: Reporta sobre
- COMMENTS_ON: Comenta sobre
- RESPONDS_TO: Responde a
- SUPPORTS: Apoya
- OPPOSES: Se opone a
- COLLABORATES_WITH: Colabora con
- COMPETES_WITH: Compete con
"""


class OntologyGenerator:
    """
    Generador de ontologías
    Analiza contenido de texto, genera definiciones de tipos de entidad y relación
    """

    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm_client = llm_client or LLMClient()

    def generate(
        self,
        document_texts: List[str],
        simulation_requirement: str,
        additional_context: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generar definiciones de ontología

        Args:
            document_texts: Lista de textos de documentos
            simulation_requirement: Descripción del requisito de simulación
            additional_context: Contexto adicional

        Returns:
            Definiciones de ontología（entity_types, edge_types, etc.）
        """
        # Construir mensaje de usuario
        user_message = self._build_user_message(
            document_texts, simulation_requirement, additional_context
        )

        lang_instruction = get_language_instruction()
        system_prompt = f"{ONTOLOGY_SYSTEM_PROMPT}\n\n{lang_instruction}\nIMPORTANT: Entity type names MUST be in English PascalCase (e.g., 'PersonEntity', 'MediaOrganization'). Relationship type names MUST be in English UPPER_SNAKE_CASE (e.g., 'WORKS_FOR'). Attribute names MUST be in English snake_case. Only description fields and analysis_summary should use the specified language above."
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]

        # Invocar LLM
        result = self.llm_client.chat_json(
            messages=messages, temperature=0.3, max_tokens=4096
        )

        # Verificar y procesar
        result = self._validate_and_process(result)

        return result

    # Longitud máxima del texto a pasar al LLM（50000 caracteres）
    MAX_TEXT_LENGTH_FOR_LLM = 50000

    def _build_user_message(
        self,
        document_texts: List[str],
        simulation_requirement: str,
        additional_context: Optional[str],
    ) -> str:
        """Construir mensaje de usuario"""

        # Combinar textos
        combined_text = "\n\n---\n\n".join(document_texts)
        original_length = len(combined_text)

        # Si el texto excede 50000 caracteres, truncar（solo afecta el contenido pasado al LLM, no afecta la construcción del grafo）
        if len(combined_text) > self.MAX_TEXT_LENGTH_FOR_LLM:
            combined_text = combined_text[: self.MAX_TEXT_LENGTH_FOR_LLM]
            combined_text += f"\n\n...(texto original tiene {original_length} caracteres, se han tomado los primeros {self.MAX_TEXT_LENGTH_FOR_LLM} caracteres para el análisis de ontología)..."

        message = f"""## Requisito de simulación

{simulation_requirement}

## DocumentaciónContenido

{combined_text}
"""

        if additional_context:
            message += f"""
## Explicación adicional

{additional_context}
"""

        message += """
Por favor, basándose en el contenido anterior, diseñe tipos de entidad y relación adecuados para la simulación de opinión pública social.

**Reglas que deben seguirse**：
1. Debe generar exactamente 10 tipos de entidad
2. Los últimos 2 elementos deben ser tipos de respaldo：Person（respaldo para personas）y Organization（respaldo para organizaciones）
3. Los primeros 8 elementos son tipos específicos diseñados basados en el contenido de texto
4. Todos los tipos de entidad deben ser sujetos que pueden emitir voz en el mundo real, no pueden ser conceptos abstractos
5. Los nombres de atributos no pueden usar name, uuid, group_id y otras palabras reservadas, usar full_name, org_name, etc. como alternativas
"""

        return message

    def _validate_and_process(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Verificar y procesar resultados"""

        # Asegurar que los campos necesarios existan
        if "entity_types" not in result:
            result["entity_types"] = []
        if "edge_types" not in result:
            result["edge_types"] = []
        if "analysis_summary" not in result:
            result["analysis_summary"] = ""

        # Verificación de tipos de entidad
        # Registrar mapeo desde nombre original hasta PascalCase, usado para corregir posteriormente las referencias source_targets de edge
        entity_name_map = {}
        for entity in result["entity_types"]:
            # Forzar conversión de entity name a PascalCase（requisito de Zep API）
            if "name" in entity:
                original_name = entity["name"]
                entity["name"] = _to_pascal_case(original_name)
                if entity["name"] != original_name:
                    logger.warning(
                        f"Entity type name '{original_name}' auto-converted to '{entity['name']}'"
                    )
                entity_name_map[original_name] = entity["name"]
            if "attributes" not in entity:
                entity["attributes"] = []
            if "examples" not in entity:
                entity["examples"] = []
            # Asegurar que la descripción no exceda 100 caracteres
            if len(entity.get("description", "")) > 100:
                entity["description"] = entity["description"][:97] + "..."

        # Verificación de tipos de relación
        for edge in result["edge_types"]:
            # Forzar conversión de edge name a SCREAMING_SNAKE_CASE（requisito de Zep API）
            if "name" in edge:
                original_name = edge["name"]
                edge["name"] = original_name.upper()
                if edge["name"] != original_name:
                    logger.warning(
                        f"Edge type name '{original_name}' auto-converted to '{edge['name']}'"
                    )
            # Corregir referencias de nombre de entidad en source_targets para mantener consistencia con PascalCase convertido
            for st in edge.get("source_targets", []):
                if st.get("source") in entity_name_map:
                    st["source"] = entity_name_map[st["source"]]
                if st.get("target") in entity_name_map:
                    st["target"] = entity_name_map[st["target"]]
            if "source_targets" not in edge:
                edge["source_targets"] = []
            if "attributes" not in edge:
                edge["attributes"] = []
            if len(edge.get("description", "")) > 100:
                edge["description"] = edge["description"][:97] + "..."

        # Límite de Zep API：máximo 10 tipos de entidad personalizados, máximo 10 tipos de borde personalizados
        MAX_ENTITY_TYPES = 10
        MAX_EDGE_TYPES = 10

        # Eliminación de duplicados：eliminar duplicados por nombre, mantener la primera aparición
        seen_names = set()
        deduped = []
        for entity in result["entity_types"]:
            name = entity.get("name", "")
            if name and name not in seen_names:
                seen_names.add(name)
                deduped.append(entity)
            elif name in seen_names:
                logger.warning(
                    f"Duplicate entity type '{name}' removed during validation"
                )
        result["entity_types"] = deduped

        # Definición de tipos de respaldo
        person_fallback = {
            "name": "Person",
            "description": "Any individual person not fitting other specific person types.",
            "attributes": [
                {
                    "name": "full_name",
                    "type": "text",
                    "description": "Full name of the person",
                },
                {"name": "role", "type": "text", "description": "Role or occupation"},
            ],
            "examples": ["ordinary citizen", "anonymous netizen"],
        }

        organization_fallback = {
            "name": "Organization",
            "description": "Any organization not fitting other specific organization types.",
            "attributes": [
                {
                    "name": "org_name",
                    "type": "text",
                    "description": "Name of the organization",
                },
                {
                    "name": "org_type",
                    "type": "text",
                    "description": "Type of organization",
                },
            ],
            "examples": ["small business", "community group"],
        }

        # Inspección si ya tiene tipos de respaldo
        entity_names = {e["name"] for e in result["entity_types"]}
        has_person = "Person" in entity_names
        has_organization = "Organization" in entity_names

        # Tipos de respaldo necesarios a agregar
        fallbacks_to_add = []
        if not has_person:
            fallbacks_to_add.append(person_fallback)
        if not has_organization:
            fallbacks_to_add.append(organization_fallback)

        if fallbacks_to_add:
            current_count = len(result["entity_types"])
            needed_slots = len(fallbacks_to_add)

            # Si al agregar excede 10 elementos, necesita eliminar algunos tipos existentes
            if current_count + needed_slots > MAX_ENTITY_TYPES:
                # Calcular cuántos elementos necesita eliminar
                to_remove = current_count + needed_slots - MAX_ENTITY_TYPES
                # Eliminar desde el final（mantener los tipos específicos más importantes al frente）
                result["entity_types"] = result["entity_types"][:-to_remove]

            # Agregar tipos de respaldo
            result["entity_types"].extend(fallbacks_to_add)

        # Finalmente asegurar no exceder el límite（programación defensiva）
        if len(result["entity_types"]) > MAX_ENTITY_TYPES:
            result["entity_types"] = result["entity_types"][:MAX_ENTITY_TYPES]

        if len(result["edge_types"]) > MAX_EDGE_TYPES:
            result["edge_types"] = result["edge_types"][:MAX_EDGE_TYPES]

        return result

    def generate_python_code(self, ontology: Dict[str, Any]) -> str:
        """
        Convertir definiciones de ontología a código Python（similar a ontology.py）

        Args:
            ontology: Definiciones de ontología

        Returns:
            Cadena de código Python
        """
        code_lines = [
            '"""',
            "Definiciones de tipos de entidad personalizadas",
            "Generado automáticamente por MiroFish, para simulación de opinión pública social",
            '"""',
            "",
            "from pydantic import Field",
            "from zep_cloud.external_clients.ontology import EntityModel, EntityText, EdgeModel",
            "",
            "",
            "# ============== Definiciones de tipos de entidad ==============",
            "",
        ]

        # Generación de tipos de entidad
        for entity in ontology.get("entity_types", []):
            name = entity["name"]
            desc = entity.get("description", f"A {name} entity.")

            code_lines.append(f"class {name}(EntityModel):")
            code_lines.append(f'    """{desc}"""')

            attrs = entity.get("attributes", [])
            if attrs:
                for attr in attrs:
                    attr_name = attr["name"]
                    attr_desc = attr.get("description", attr_name)
                    code_lines.append(f"    {attr_name}: EntityText = Field(")
                    code_lines.append(f'        description="{attr_desc}",')
                    code_lines.append(f"        default=None")
                    code_lines.append(f"    )")
            else:
                code_lines.append("    pass")

            code_lines.append("")
            code_lines.append("")

        code_lines.append(
            "# ============== Definiciones de tipos de relación =============="
        )
        code_lines.append("")

        # Generación de tipos de relación
        for edge in ontology.get("edge_types", []):
            name = edge["name"]
            # Transformación a nombre de clase PascalCase
            class_name = "".join(word.capitalize() for word in name.split("_"))
            desc = edge.get("description", f"A {name} relationship.")

            code_lines.append(f"class {class_name}(EdgeModel):")
            code_lines.append(f'    """{desc}"""')

            attrs = edge.get("attributes", [])
            if attrs:
                for attr in attrs:
                    attr_name = attr["name"]
                    attr_desc = attr.get("description", attr_name)
                    code_lines.append(f"    {attr_name}: EntityText = Field(")
                    code_lines.append(f'        description="{attr_desc}",')
                    code_lines.append(f"        default=None")
                    code_lines.append(f"    )")
            else:
                code_lines.append("    pass")

            code_lines.append("")
            code_lines.append("")

        # Generación de diccionario de tipos
        code_lines.append("# ============== Configuración de tipos ==============")
        code_lines.append("")
        code_lines.append("ENTITY_TYPES = {")
        for entity in ontology.get("entity_types", []):
            name = entity["name"]
            code_lines.append(f'    "{name}": {name},')
        code_lines.append("}")
        code_lines.append("")
        code_lines.append("EDGE_TYPES = {")
        for edge in ontology.get("edge_types", []):
            name = edge["name"]
            class_name = "".join(word.capitalize() for word in name.split("_"))
            code_lines.append(f'    "{name}": {class_name},')
        code_lines.append("}")
        code_lines.append("")

        # Generación de mapeo de source_targets de borde
        code_lines.append("EDGE_SOURCE_TARGETS = {")
        for edge in ontology.get("edge_types", []):
            name = edge["name"]
            source_targets = edge.get("source_targets", [])
            if source_targets:
                st_list = ", ".join(
                    [
                        f'{{"source": "{st.get("source", "Entity")}", "target": "{st.get("target", "Entity")}"}}'
                        for st in source_targets
                    ]
                )
                code_lines.append(f'    "{name}": [{st_list}],')
        code_lines.append("}")

        return "\n".join(code_lines)
