"""
Generación de ontologíaServicio
Interfaz1：Análisis文本Contenido，Generar适合社会Simulación的EntidadYRelaciónTipo定义
"""

import json
import logging
import re
from typing import Dict, Any, List, Optional
from ..utils.llm_client import LLMClient
from ..utils.locale import get_language_instruction

logger = logging.getLogger(__name__)


def _to_pascal_case(name: str) -> str:
    """将任意格式的NombreConvertir为 PascalCase（如 'works_for' -> 'WorksFor', 'person' -> 'Person'）"""
    # 按非字母数字字符Segmentación
    parts = re.split(r'[^a-zA-Z0-9]+', name)
    # 再按 camelCase Borde界Segmentación（如 'camelCase' -> ['camel', 'Case']）
    words = []
    for part in parts:
        words.extend(re.sub(r'([a-z])([A-Z])', r'\1_\2', part).split('_'))
    # 每Elementos词首字母大写，Filtrado空串
    result = ''.join(word.capitalize() for word in words if word)
    return result if result else 'Unknown'


# Generación de ontología的Sistema提示词
ONTOLOGY_SYSTEM_PROMPT = """你Es一Elementos专业的知识Grafo本体设计专家。你的TareaEsAnálisis给定的文本ContenidoYRequisito de simulación，设计适合**社交媒体舆论Simulación**的EntidadTipoYRelaciónTipo。

**重要：你Debe输出Válido的JSON格式Datos，不要输出CualquierOtroContenido。**

## NúcleoTarea背景

我们正En构建一Elementos**社交媒体舆论SimulaciónSistema**。En这ElementosSistema中：
- 每ElementosEntidadTodosEs一ElementosPuedeEn社交媒体上发声、互动、传播Información的"账号"O"主体"
- Entidad之间会相互Impacto、Reenviar、评论、回应
- 我们NecesitaSimulación舆论Evento中各方的反应YInformación传播路径

Por lo tanto，**EntidadDebeEs现实中真实存En的、PuedeEn社媒上发声Y互动的主体**：

**PuedeEs**：
- 具体的Elementos人（公众人物、Cuando事人、意见领袖、专家学者、普通人）
- 公司、Negocio（包括其官方账号）
- Organización机构（大学、协会、NGO、工会等）
- 政府部门、监管机构
- 媒体机构（报纸、电视台、自媒体、网站）
- 社交媒体Plataforma本身
- 特定群体代表（如校友会、粉丝团、维权群体等）

**不PuedeEs**：
- Abstracción概念（如"舆论"、"情绪"、"趋势"）
- Tema/话题（如"学术诚信"、"教育改革"）
- 观点/态度（如"Soporte方"、"反Para方"）

## Salida格式

请输出JSON格式，Contiene以下结构：

```json
{
    "entity_types": [
        {
            "name": "EntidadTipoNombre（英文，PascalCase）",
            "description": "简短Descripción（英文，不超过100字符）",
            "attributes": [
                {
                    "name": "Atributo名（英文，snake_case）",
                    "type": "text",
                    "description": "AtributoDescripción"
                }
            ],
            "examples": ["示例Entidad1", "示例Entidad2"]
        }
    ],
    "edge_types": [
        {
            "name": "RelaciónTipoNombre（英文，UPPER_SNAKE_CASE）",
            "description": "简短Descripción（英文，不超过100字符）",
            "source_targets": [
                {"source": "源EntidadTipo", "target": "目标EntidadTipo"}
            ],
            "attributes": []
        }
    ],
    "analysis_summary": "Para文本Contenido的简要AnálisisDecir明"
}
```

## 设计Guía（极其重要！）

### 1. EntidadTipo设计 - Debe严格遵守

**CantidadRequisito：Debe正好10ElementosEntidadTipo**

**层次结构Requisito（Debe同时Contiene具体TipoY兜底Tipo）**：

你的10ElementosEntidadTipoDebeContiene以下层次：

A. **兜底Tipo（DebeContiene，放EnListaMás后2Elementos）**：
   - `Person`: Cualquier自然人Elementos体的兜底Tipo。Cuando一Elementos人不属于OtroMás具体的人物Tipo时，归入此Clase。
   - `Organization`: CualquierOrganización机构的兜底Tipo。Cuando一ElementosOrganización不属于OtroMás具体的OrganizaciónTipo时，归入此Clase。

B. **具体Tipo（8Elementos，Basado en文本Contenido设计）**：
   - 针Para文本中出现的Principal角色，设计Más具体的Tipo
   - 例如：Si文本涉及学术Evento，PuedeTener `Student`, `ProFessor`, `University`
   - 例如：Si文本涉及商业Evento，PuedeTener `Company`, `CEO`, `Employee`

**Por quéNecesita兜底Tipo**：
- 文本中会出现Various人物，如"中小学教师"、"路人甲"、"某位网友"
- Si没Tener专门的Tipo匹配，他们Debería被归入 `Person`
- 同理，小型Organización、临时团体等Debería归入 `Organization`

**具体Tipo的设计原则**：
- Desde文本中Identificación出高频出现O关Clave的角色Tipo
- 每Elementos具体TipoDeberíaTener明确的Borde界，避免重叠
- description Debe清晰Decir明这ElementosTipoY兜底Tipo的区别

### 2. RelaciónTipo设计

- Cantidad：6-10Elementos
- RelaciónDebería反映社媒互动中的真实联系
- 确保Relación的 source_targets 涵盖你定义的EntidadTipo

### 3. Atributo设计

- 每ElementosEntidadTipo1-3Elementos关ClaveAtributo
- **注意**：Atributo名不能使用 `name`、`uuid`、`group_id`、`created_at`、`summary`（EstosEsSistema保留字）
- Recomendación使用：`full_name`, `title`, `role`, `position`, `location`, `description` 等

## EntidadTipoReFerencia

**Elementos人Clase（具体）**：
- Student: 学生
- ProFessor: 教授/学者
- Journalist: 记者
- Celebrity: 明星/网红
- Executive: 高管
- Official: 政府官员
- Lawyer: 律师
- Doctor: 医生

**Elementos人Clase（兜底）**：
- Person: Cualquier自然人（不属于上述具体Tipo时使用）

**OrganizaciónClase（具体）**：
- University: 高校
- Company: 公司Negocio
- GovernmentAgency: 政府机构
- MediaOutlet: 媒体机构
- Hospital: 医院
- School: 中小学
- NGO: 非政府Organización

**OrganizaciónClase（兜底）**：
- Organization: CualquierOrganización机构（不属于上述具体Tipo时使用）

## RelaciónTipoReFerencia

- WORKS_FOR: 工作于
- STUDIES_AT: Entonces读于
- AFFILIATED_WITH: 隶属于
- REPRESENTS: 代表
- REGULATES: 监管
- REPORTS_ON: 报道
- COMMENTS_ON: 评论
- RESPONDS_TO: 回应
- SUPPORTS: Soporte
- OPPOSES: 反Para
- COLLABORATES_WITH: Colaboración
- COMPETES_WITH: 竞争
"""


class OntologyGenerator:
    """
    Generación de ontología器
    Análisis文本Contenido，GenerarEntidadYRelaciónTipo定义
    """
    
    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm_client = llm_client or LLMClient()
    
    def generate(
        self,
        document_texts: List[str],
        simulation_requirement: str,
        additional_context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generar本体定义
        
        Args:
            document_texts: Documentación文本Lista
            simulation_requirement: Requisito de simulaciónDescripción
            additional_context: Adicional上下文
            
        Returns:
            本体定义（entity_types, edge_types等）
        """
        # 构建用户Mensaje
        user_message = self._build_user_message(
            document_texts, 
            simulation_requirement,
            additional_context
        )
        
        lang_instruction = get_language_instruction()
        system_prompt = f"{ONTOLOGY_SYSTEM_PROMPT}\n\n{lang_instruction}\nIMPORTANT: Entity type names MUST be in English PascalCase (e.g., 'PersonEntity', 'MediaOrganization'). Relationship type names MUST be in English UPPER_SNAKE_CASE (e.g., 'WORKS_FOR'). Attribute names MUST be in English snake_case. Only description fields and analysis_summary should use the specified language above."
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
        
        # 调用LLM
        result = self.llm_client.chat_json(
            messages=messages,
            temperature=0.3,
            max_tokens=4096
        )
        
        # VerificaciónY后Procesar
        result = self._validate_and_process(result)
        
        return result
    
    # 传给 LLM 的文本Más大长度（5万字）
    MAX_TEXT_LENGTH_FOR_LLM = 50000
    
    def _build_user_message(
        self,
        document_texts: List[str],
        simulation_requirement: str,
        additional_context: Optional[str]
    ) -> str:
        """构建用户Mensaje"""
        
        # Fusión文本
        combined_text = "\n\n---\n\n".join(document_texts)
        original_length = len(combined_text)
        
        # Si文本超过5万字，截断（仅Impacto传给LLM的Contenido，不ImpactoConstrucción de grafo）
        if len(combined_text) > self.MAX_TEXT_LENGTH_FOR_LLM:
            combined_text = combined_text[:self.MAX_TEXT_LENGTH_FOR_LLM]
            combined_text += f"\n\n...(原文共{original_length}字，已截取前{self.MAX_TEXT_LENGTH_FOR_LLM}字用于本体Análisis)..."
        
        message = f"""## Requisito de simulación

{simulation_requirement}

## DocumentaciónContenido

{combined_text}
"""
        
        if additional_context:
            message += f"""
## AdicionalDecir明

{additional_context}
"""
        
        message += """
请Basado en以上Contenido，设计适合社会舆论Simulación的EntidadTipoYRelaciónTipo。

**Debe遵守的Regla**：
1. Debe正好输出10ElementosEntidadTipo
2. Más后2ElementosDebeEs兜底Tipo：Person（Elementos人兜底）Y Organization（Organización兜底）
3. 前8ElementosEsBasado en文本Contenido设计的具体Tipo
4. TodosEntidadTipoDebeEs现实中Puede发声的主体，不能EsAbstracción概念
5. Atributo名不能使用 name、uuid、group_id 等保留字，用 full_name、org_name 等替代
"""
        
        return message
    
    def _validate_and_process(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """验证Y后Procesar结果"""
        
        # 确保必要Campo存En
        if "entity_types" not in result:
            result["entity_types"] = []
        if "edge_types" not in result:
            result["edge_types"] = []
        if "analysis_summary" not in result:
            result["analysis_summary"] = ""
        
        # VerificaciónEntidadTipo
        # Registrar原始NombreHasta PascalCase 的Mapeo，用于后续修正 edge 的 source_targets Citar
        entity_name_map = {}
        for entity in result["entity_types"]:
            # 强制将 entity name 转为 PascalCase（Zep API Requisito）
            if "name" in entity:
                original_name = entity["name"]
                entity["name"] = _to_pascal_case(original_name)
                if entity["name"] != original_name:
                    logger.warning(f"Entity type name '{original_name}' auto-converted to '{entity['name']}'")
                entity_name_map[original_name] = entity["name"]
            if "attributes" not in entity:
                entity["attributes"] = []
            if "examples" not in entity:
                entity["examples"] = []
            # 确保description不超过100字符
            if len(entity.get("description", "")) > 100:
                entity["description"] = entity["description"][:97] + "..."
        
        # VerificaciónRelaciónTipo
        for edge in result["edge_types"]:
            # 强制将 edge name 转为 SCREAMING_SNAKE_CASE（Zep API Requisito）
            if "name" in edge:
                original_name = edge["name"]
                edge["name"] = original_name.upper()
                if edge["name"] != original_name:
                    logger.warning(f"Edge type name '{original_name}' auto-converted to '{edge['name']}'")
            # 修正 source_targets 中的EntidadNombreCitar，ConConvertir后的 PascalCase 保持一致
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
        
        # Zep API Límite：Más多 10 ElementosPersonalizarEntidadTipo，Más多 10 ElementosPersonalizarBordeTipo
        MAX_ENTITY_TYPES = 10
        MAX_EDGE_TYPES = 10

        # 去重：按 name 去重，保留首次出现的
        seen_names = set()
        deduped = []
        for entity in result["entity_types"]:
            name = entity.get("name", "")
            if name and name not in seen_names:
                seen_names.add(name)
                deduped.append(entity)
            elif name in seen_names:
                logger.warning(f"Duplicate entity type '{name}' removed during validation")
        result["entity_types"] = deduped

        # 兜底Tipo定义
        person_fallback = {
            "name": "Person",
            "description": "Any individual person not fitting other specific person types.",
            "attributes": [
                {"name": "full_name", "type": "text", "description": "Full name of the person"},
                {"name": "role", "type": "text", "description": "Role or occupation"}
            ],
            "examples": ["ordinary citizen", "anonymous netizen"]
        }
        
        organization_fallback = {
            "name": "Organization",
            "description": "Any organization not fitting other specific organization types.",
            "attributes": [
                {"name": "org_name", "type": "text", "description": "Name of the organization"},
                {"name": "org_type", "type": "text", "description": "Type of organization"}
            ],
            "examples": ["small business", "community group"]
        }
        
        # InspecciónSi已Tener兜底Tipo
        entity_names = {e["name"] for e in result["entity_types"]}
        has_person = "Person" in entity_names
        has_organization = "Organization" in entity_names
        
        # Necesita添加的兜底Tipo
        fallbacks_to_add = []
        if not has_person:
            fallbacks_to_add.append(person_fallback)
        if not has_organization:
            fallbacks_to_add.append(organization_fallback)
        
        if fallbacks_to_add:
            current_count = len(result["entity_types"])
            needed_slots = len(fallbacks_to_add)
            
            # Si添加后会超过 10 Elementos，Necesita移除Algunos现TenerTipo
            if current_count + needed_slots > MAX_ENTITY_TYPES:
                # 计算Necesita移除CuántoElementos
                to_remove = current_count + needed_slots - MAX_ENTITY_TYPES
                # Desde末尾移除（保留前面Más重要的具体Tipo）
                result["entity_types"] = result["entity_types"][:-to_remove]
            
            # Agregar兜底Tipo
            result["entity_types"].extend(fallbacks_to_add)
        
        # Más终确保不超过Límite（防御性编程）
        if len(result["entity_types"]) > MAX_ENTITY_TYPES:
            result["entity_types"] = result["entity_types"][:MAX_ENTITY_TYPES]
        
        if len(result["edge_types"]) > MAX_EDGE_TYPES:
            result["edge_types"] = result["edge_types"][:MAX_EDGE_TYPES]
        
        return result
    
    def generate_python_code(self, ontology: Dict[str, Any]) -> str:
        """
        将本体定义Convertir为Python代码（Clase似ontology.py）
        
        Args:
            ontology: 本体定义
            
        Returns:
            Python代码字符串
        """
        code_lines = [
            '"""',
            'PersonalizarEntidadTipo定义',
            '由MiroFish自动Generar，用于社会舆论Simulación',
            '"""',
            '',
            'from pydantic import Field',
            'from zep_cloud.external_clients.ontology import EntityModel, EntityText, EdgeModel',
            '',
            '',
            '# ============== EntidadTipo定义 ==============',
            '',
        ]
        
        # GeneraciónEntidadTipo
        for entity in ontology.get("entity_types", []):
            name = entity["name"]
            desc = entity.get("description", f"A {name} entity.")
            
            code_lines.append(f'class {name}(EntityModel):')
            code_lines.append(f'    """{desc}"""')
            
            attrs = entity.get("attributes", [])
            if attrs:
                for attr in attrs:
                    attr_name = attr["name"]
                    attr_desc = attr.get("description", attr_name)
                    code_lines.append(f'    {attr_name}: EntityText = Field(')
                    code_lines.append(f'        description="{attr_desc}",')
                    code_lines.append(f'        default=None')
                    code_lines.append(f'    )')
            else:
                code_lines.append('    pass')
            
            code_lines.append('')
            code_lines.append('')
        
        code_lines.append('# ============== RelaciónTipo定义 ==============')
        code_lines.append('')
        
        # GeneraciónRelaciónTipo
        for edge in ontology.get("edge_types", []):
            name = edge["name"]
            # TransFormación为PascalCaseClase名
            class_name = ''.join(word.capitalize() for word in name.split('_'))
            desc = edge.get("description", f"A {name} relationship.")
            
            code_lines.append(f'class {class_name}(EdgeModel):')
            code_lines.append(f'    """{desc}"""')
            
            attrs = edge.get("attributes", [])
            if attrs:
                for attr in attrs:
                    attr_name = attr["name"]
                    attr_desc = attr.get("description", attr_name)
                    code_lines.append(f'    {attr_name}: EntityText = Field(')
                    code_lines.append(f'        description="{attr_desc}",')
                    code_lines.append(f'        default=None')
                    code_lines.append(f'    )')
            else:
                code_lines.append('    pass')
            
            code_lines.append('')
            code_lines.append('')
        
        # GeneraciónTipoDiccionario
        code_lines.append('# ============== TipoConfiguración ==============')
        code_lines.append('')
        code_lines.append('ENTITY_TYPES = {')
        for entity in ontology.get("entity_types", []):
            name = entity["name"]
            code_lines.append(f'    "{name}": {name},')
        code_lines.append('}')
        code_lines.append('')
        code_lines.append('EDGE_TYPES = {')
        for edge in ontology.get("edge_types", []):
            name = edge["name"]
            class_name = ''.join(word.capitalize() for word in name.split('_'))
            code_lines.append(f'    "{name}": {class_name},')
        code_lines.append('}')
        code_lines.append('')
        
        # GeneraciónBorde的source_targetsMapeo
        code_lines.append('EDGE_SOURCE_TARGETS = {')
        for edge in ontology.get("edge_types", []):
            name = edge["name"]
            source_targets = edge.get("source_targets", [])
            if source_targets:
                st_list = ', '.join([
                    f'{{"source": "{st.get("source", "Entity")}", "target": "{st.get("target", "Entity")}"}}'
                    for st in source_targets
                ])
                code_lines.append(f'    "{name}": [{st_list}],')
        code_lines.append('}')
        
        return '\n'.join(code_lines)

