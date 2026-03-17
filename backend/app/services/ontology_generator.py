"""
Servicio de generación de ontología
Interfaz 1: Analizar contenido de texto, generar tipos de entidades y relaciones definidos apropiados para simulación social
"""

import json
from typing import Dict, Any, List, Optional
from ..utils.llm_client import LLMClient

# Intentar importar el sistema de prompts i18n
try:
    from ..prompts import get_prompt as _get_prompt

    _PROMPTS_AVAILABLE = True
except ImportError:
    _PROMPTS_AVAILABLE = False
    _get_prompt = None


def _get_ontology_system_prompt() -> str:
    """Obtiene el prompt del sistema para ontología (i18n)."""
    if _PROMPTS_AVAILABLE:
        return _get_prompt("ontology", "ontology_system", "")
    else:
        # Fallback al original en chino
        return ONTOLOGY_SYSTEM_PROMPT_CHINESE


# Prompt del sistema para generación de ontología (fallback cuando i18n no está disponible)
ONTOLOGY_SYSTEM_PROMPT_CHINESE = """Eres un experto profesional en diseño de ontología de grafos de conocimiento. Tu tarea es analizar el contenido del texto dado y los requisitos de simulación, diseñar tipos de entidades y tipos de relaciones apropiados para **simulación de opinión en redes sociales**.

**Importante: Debes generar datos en formato JSON válido, no generes ningún otro contenido.**

## Contexto de la tarea principal

Estamos construyendo un **sistema de simulación de opinión en redes sociales**. En este sistema:
- Cada entidad es una "cuenta" o "sujeto" que puede publicar, interactuar y diseminar información en redes sociales
- Las entidades se influyen mutuamente, republican, comentan y responden
- Necesitamos simular las reacciones de todas las partes y las rutas de diseminación de información en eventos de opinión

Por lo tanto, **las entidades deben ser sujetos que realmente existan en la realidad y puedan expresarse e interactuar en redes sociales**:

**PUEDEN SER**:
- Individuos específicos (figuras públicas, partes involucradas, líderes de opinión, expertos académicos, personas comunes)
- Empresas, corporaciones (incluyendo sus cuentas oficiales)
- Organizaciones (universidades, asociaciones, ONGs, sindicatos, etc.)
- Agencias gubernamentales, reguladores
- Medios de comunicación (periódicos, TVs, autofonts, sitios web)
- Plataformas de redes sociales en sí
- Representantes de grupos específicos (asociaciones de ex-alumnos, fandom, grupos de derechos, etc.)

**NO PUEDEN SER**:
- Conceptos abstractos (como "opinión", "emoción", "tendencia")
- Temas/tópicos (como "integridad académica", "reforma educativa")
- Actitudes/puntos de vista (como "partidarios", "opositores")

## Formato de salida

Por favor genera en formato JSON, con la siguiente estructura:

```json
{
    "entity_types": [
        {
            "name": "Nombre del tipo de entidad (inglés, PascalCase)",
            "description": "Descripción breve (inglés, máximo 100 caracteres)",
            "attributes": [
                {
                    "name": "Nombre del atributo (inglés, snake_case)",
                    "type": "text",
                    "description": "Descripción del atributo"
                }
            ],
            "examples": ["Entidad de ejemplo 1", "Entidad de ejemplo 2"]
        }
    ],
    "edge_types": [
        {
            "name": "Nombre del tipo de relación (inglés, UPPER_SNAKE_CASE)",
            "description": "Descripción breve (inglés, máximo 100 caracteres)",
            "source_targets": [
                {"source": "Tipo de entidad origen", "target": "Tipo de entidad destino"}
            ],
            "attributes": []
        }
    ],
    "analysis_summary": "Breve análisis y explicación del contenido del texto (chino)"
}
```

## Guía de diseño (¡extremadamente importante!)

### 1. Diseño de tipos de entidad - Debe

**Requisito de cantidad: Exactamente 10 tipos de entidad**

**Requisito de estructura jerárquica (debe incluir tipos específicos Y tipos de respaldo al mismo tiempo)**:

Tus 10 tipos de entidad deben incluir la siguiente jerarquía:

A. **Tipos de respaldo (debe incluir, colocar últimos 2 en la lista)**:
   - `Person`: Tipo de respaldo para cualquier individuo natural. Cuando una persona no pertenezca a otros tipos más específicos de personas, se clasifica aquí.
   - `Organization`: Tipo de respaldo para cualquier organización. Cuando una organización no pertenezca a otros tipos más específicos de organización, se clasifica aquí.

B. **Tipos específicos (8, diseñados según contenido del texto)**:
   - Diseñar tipos más específicos para los roles principales que aparecen en el texto
   - Por ejemplo: si el texto involucra eventos académicos, puede haber `Student`, `Professor`, `University`
   - Por ejemplo: si el texto involucra eventos comerciales, puede haber `Company`, `CEO`, `Employee`

**Por qué se necesitan tipos de respaldo**:
- El texto puede contener varias personas como "profesor de escuela primaria", " transeúnte", "algunos"
- Si no hay un tipo específico para ellos, deben clasificarse en `Person`
- De manera similar, organizaciones pequeñas, grupos temporales deben clasificarse en `Organization`

**Principios de diseño de tipos específicos**:
- Identificar los tipos de rol que aparecen con mayor frecuencia o son clave en el texto
- Cada tipo específico debe tener límites claros, evitar superposición
- La descripción debe explicar claramente la diferencia entre este tipo y el tipo de respaldo

### 2. Diseño de tipos de relaciones

- Cantidad: 6-10
- Las relaciones deben reflejar conexiones reales en interacciones de redes sociales
- Asegurar que los source_targets de las relaciones cubran los tipos de entidad que definiste

### 3. Diseño de atributos

- 1-3 atributos clave por tipo de entidad
- **Nota**: Los nombres de atributos no pueden usar `name`, `uuid`, `group_id`, `created_at`, `summary` (estos son palabras reservadas del sistema)
- Recomendado usar: `full_name`, `title`, `role`, `position`, `location`, `description`, etc.

## Referencia de tipos de entidad

**Personas (específicas)**:
- Student: Estudiante
- Professor: Profesor/académico
- Journalist: Periodista
- Celebrity: Estrella/famoso de internet
- Executive: Ejecutivo
- Official: Funcionario gubernamental
- Lawyer: Abogado
- Doctor: Médico

**Personas (respaldo)**:
- Person: Cualquier persona natural (usar cuando no pertenezca a tipos específicos anteriores)

**Organizaciones (específicas)**:
- University: Universidad
- Company: Empresa
- GovernmentAgency: Agencia gubernamental
- MediaOutlet: Medio de comunicación
- Hospital: Hospital
- School: Escuela primaria/secundaria
- NGO: Organización no gubernamental

**Organizaciones (respaldo)**:
- Organization: Cualquier organización (usar cuando no pertenezca a tipos específicos anteriores)

## Referencia de tipos de relaciones

- WORKS_FOR: Trabaja en
- STUDIES_AT: Estudia en
- AFFILIATED_WITH: Afiliado a
- REPRESENTS: Representa
- REGULATES: Regula
- REPORTS_ON: Reporta sobre
- COMMENTS_ON: Comenta sobre
- RESPONDS_TO: Responde a
- SUPPORTS: Apoya
- OPPOSES: Se opone a
- COLLABORATES_WITH: Colabora con
- COMPETES_WITH: Compite con
"""


class OntologyGenerator:
    """
    Generador de ontología
    Analizar contenido de texto, generar definiciones de tipos de entidades y relaciones
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
        Generar definición de ontología

        Args:
            document_texts: Lista de textos de documentos
            simulation_requirement: Descripción de requisitos de simulación
            additional_context: Contexto adicional

        Returns:
            Definición de ontología (entity_types, edge_types, etc.)
        """
        # Construir mensaje de usuario
        user_message = self._build_user_message(
            document_texts, simulation_requirement, additional_context
        )

        messages = [
            {"role": "system", "content": _get_ontology_system_prompt()},
            {"role": "user", "content": user_message},
        ]

        # Llamar LLM
        result = self.llm_client.chat_json(
            messages=messages, temperature=0.3, max_tokens=4096
        )

        # Validar y post-procesar
        result = self._validate_and_process(result)

        return result

    # Longitud máxima de texto para LLM (50 mil caracteres)
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

        # Si el texto excede 50 mil caracteres, truncar (solo afecta lo pasado al LLM, no afecta construcción del grafo)
        if len(combined_text) > self.MAX_TEXT_LENGTH_FOR_LLM:
            combined_text = combined_text[: self.MAX_TEXT_LENGTH_FOR_LLM]
            combined_text += f"\n\n...(El texto original tiene {original_length} caracteres, se han tomado los primeros {self.MAX_TEXT_LENGTH_FOR_LLM} para análisis de ontología)..."

        message = f"""## Requisitos de simulación

{simulation_requirement}

## Contenido del documento

{combined_text}
"""

        if additional_context:
            message += f"""
## Explicación adicional

{additional_context}
"""

        message += """
Por favor, basándote en el contenido anterior, diseñar tipos de entidades y tipos de relaciones apropiados para simulación de opinión social.

**Reglas que deben cumplirse**:
1. Debe haber exactamente 10 tipos de entidades
2. Los últimos 2 deben ser tipos de respaldo: Person (respaldo de individuo) y Organization (respaldo de organización)
3. Los primeros 8 son tipos específicos diseñados según el contenido del texto
4. Todos los tipos de entidades deben ser sujetos que puedan expresarse en la realidad, no conceptos abstractos
5. Los nombres de atributos no pueden usar palabras reservadas como name, uuid, group_id, usar full_name, org_name, etc. en su lugar
"""

        return message

    def _validate_and_process(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Validar y post-procesar resultados"""

        # Asegurar que existan campos necesarios
        if "entity_types" not in result:
            result["entity_types"] = []
        if "edge_types" not in result:
            result["edge_types"] = []
        if "analysis_summary" not in result:
            result["analysis_summary"] = ""

        # Validar tipos de entidades
        for entity in result["entity_types"]:
            if "attributes" not in entity:
                entity["attributes"] = []
            if "examples" not in entity:
                entity["examples"] = []
            # Asegurar que description no exceda 100 caracteres
            if len(entity.get("description", "")) > 100:
                entity["description"] = entity["description"][:97] + "..."

        # Validar tipos de relaciones
        for edge in result["edge_types"]:
            if "source_targets" not in edge:
                edge["source_targets"] = []
            if "attributes" not in edge:
                edge["attributes"] = []
            if len(edge.get("description", "")) > 100:
                edge["description"] = edge["description"][:97] + "..."

        # Restricciones de API de Zep: máximo 10 tipos de entidades personalizadas, máximo 10 tipos de bordes personalizados
        MAX_ENTITY_TYPES = 10
        MAX_EDGE_TYPES = 10

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

        # Verificar si ya existen tipos de respaldo
        entity_names = {e["name"] for e in result["entity_types"]}
        has_person = "Person" in entity_names
        has_organization = "Organization" in entity_names

        # Tipos de respaldo a añadir
        fallbacks_to_add = []
        if not has_person:
            fallbacks_to_add.append(person_fallback)
        if not has_organization:
            fallbacks_to_add.append(organization_fallback)

        if fallbacks_to_add:
            current_count = len(result["entity_types"])
            needed_slots = len(fallbacks_to_add)

            # Si añadir excedería 10, eliminar algunos tipos existentes
            if current_count + needed_slots > MAX_ENTITY_TYPES:
                # Calcular cuántos eliminar
                to_remove = current_count + needed_slots - MAX_ENTITY_TYPES
                # Eliminar desde el final (conservar tipos específicos más importantes al frente)
                result["entity_types"] = result["entity_types"][:-to_remove]

            # Añadir tipos de respaldo
            result["entity_types"].extend(fallbacks_to_add)

        # Asegurar finalmente no exceder límites (programación defensiva)
        if len(result["entity_types"]) > MAX_ENTITY_TYPES:
            result["entity_types"] = result["entity_types"][:MAX_ENTITY_TYPES]

        if len(result["edge_types"]) > MAX_EDGE_TYPES:
            result["edge_types"] = result["edge_types"][:MAX_EDGE_TYPES]

        return result

    def generate_python_code(self, ontology: Dict[str, Any]) -> str:
        """
        Convertir definición de ontología a código Python (similar a ontology.py)

        Args:
            ontology: Definición de ontología

        Returns:
            String de código Python
        """
        code_lines = [
            '"""',
            "Definiciones de tipos de entidades personalizadas",
            "Generado automáticamente por MiroFish para simulación de opinión social",
            '"""',
            "",
            "from pydantic import Field",
            "from zep_cloud.external_clients.ontology import EntityModel, EntityText, EdgeModel",
            "",
            "",
            "# ============== Definiciones de tipos de entidades ==============",
            "",
        ]

        # Generar tipos de entidades
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
            "# ============== Definiciones de tipos de relaciones =============="
        )
        code_lines.append("")

        # Generar tipos de relaciones
        for edge in ontology.get("edge_types", []):
            name = edge["name"]
            # Convertir a nombre de clase PascalCase
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

        # Generar diccionario de tipos
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

        # Generar mapeo de source_targets de bordes
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
