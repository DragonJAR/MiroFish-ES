/**
 * Composable para traducir logs del backend
 * Usa vue-i18n para mantener consistencia con el sistema de traducciones
 */
import { useI18n } from 'vue-i18n'

// Mapeo de textos chinos a claves de i18n
const logKeyMap = {
  // Estados y progreso
  '正在生成': 'logs.generating',
  '正在分析': 'logs.analyzing',
  '规划完成': 'logs.planningComplete',
  '大纲': 'logs.outline',
  '章节': 'logs.section',
  '个章节': 'logs.sections',
  '报告生成': 'logs.reportGeneration',
  '报告': 'logs.report',
  '开始规划': 'logs.startPlanning',
  '开始生成': 'logs.startGeneration',
  '生成完成': 'logs.generationComplete',
  '已完成': 'logs.completed',
  '正在': 'logs.now',
  '获取': 'logs.getting',
  '搜索': 'logs.search',
  '完成': 'logs.complete',
  '失败': 'logs.failed',
  '重试': 'logs.retrying',
  '尝试': 'logs.attempt',
  '次尝试': 'logs.attempts',
  '后仍失败': 'logs.failedAfterAttempts',
  '秒后': 'logs.secondsAfter',
  
  // Elementos del grafo
  '图谱': 'logs.graph',
  '图谱搜索': 'logs.searchInGraph',
  '双栏': 'logs.doubleColumn',
  '工作台': 'logs.workbench',
  '深度互动': 'logs.deepInteraction',
  '图谱数据加载中': 'logs.graphDataLoading',
  '节点': 'logs.nodes',
  '边': 'logs.edges',
  '实体': 'logs.entities',
  '事实': 'logs.facts',
  '关系': 'logs.relationships',
  '个节点': 'logs.nodeCount',
  '条边': 'logs.edgeCount',
  '条事实': 'logs.factCount',
  '个实体': 'logs.entityCount',
  '获取图谱': 'logs.getting',
  '获取到': 'logs.found',
  
  // Acciones
  '执行工具': 'logs.executingTool',
  '参数': 'logs.parameters',
  '调用': 'logs.calling',
  '加载': 'logs.loading',
  '保存': 'logs.saving',
  '保存到': 'logs.savedTo',
  '文件': 'logs.file',
  '报告已保存': 'logs.reportSaved',
  '大纲已保存': 'logs.outlineSaved',
  
  // Búsqueda
  '搜索完成': 'logs.searchComplete',
  '找到': 'logs.found',
  '相关': 'logs.related',
  '检索': 'logs.retrieval',
  '深度洞察': 'logs.deepInsight',
  
  // Agentes
  '人设': 'logs.profile',
  '个Agent': 'logs.agentCount',
  '加载到': 'logs.loaded',
  '进行采访': 'logs.interviewing',
  '选择了': 'logs.selected',
  '采访': 'logs.interview',
  '深度采访': 'logs.deepInterview',
  
  // Herramientas
  '与': 'logs.with',
  '对话': 'logs.chat',
  '与世界中任意个体对话': 'logs.chatWithAnyAgent',
  '报告生成智能体的快速对话版本': 'logs.reportAgentQuickVersion',
  '可调用': 'logs.canCall',
  '种专业工具': 'logs.professionalTools',
  '拥有': 'logs.has',
  '的完整记忆': 'logs.fullMemory',
  '深度归因': 'logs.deepAttribution',
  '对齐现实世界种子数据与模拟环境状态': 'logs.alignSeedData',
  '结合': 'logs.combine',
  '机制': 'logs.mechanism',
  '提供跨时空的深度归因分析': 'logs.crossTemporalAnalysis',
  '全景追踪': 'logs.panoramaTracking',
  '基于图结构的广度遍历算法': 'logs.breadthFirstTraversal',
  '重构事件传播路径': 'logs重构事件传播路径',
  '捕获全量信息流动的拓扑结构': 'logs.captureTopology',
  '快速检索': 'logs.quickSearch',
  '基于': 'logs.basedOn',
  '的即时查询接口': 'logs.instantQueryInterface',
  '优化索引效率': 'logs.optimizeIndex',
  '用于快速提取具体的节点属性与离散事实': 'logs.extractNodeFacts',
  '虚拟访谈': 'logs.virtualInterview',
  '自主式访谈': 'logs.autonomousInterview',
  '能够并行与模拟世界中个体进行多轮对话': 'logs.multiRoundDialogue',
  '采集非结构化的观点数据与心理状态': 'logs.collectOpinions',
  
  // Misceláneos
  '当前': 'logs.current',
  '最新': 'logs.latest',
  '核心实体': 'logs.mainEntities',
  '关系链': 'logs.relationshipChain',
  '当前关键记忆': 'logs.currentKeyMemory',
  '关键事实': 'logs.keyFacts',
  '时序记忆': 'logs.temporalMemory',
  '记忆': 'logs.memory',
  '关联': 'logs.associated',
  '统计信息': 'logs.statistics',
  '所有': 'logs.all',
  '展开': 'logs.expand',
  '全部': 'logs.allItems',
  '共': 'logs.total',
  '已保存': 'logs.saved',
  '分析': 'logs.analysis',
  '模拟': 'logs.simulation',
  '需求': 'logs.requirement',
  '上下文': 'logs.context',
  'API失败': 'logs.apiFailed',
  '降级为': 'logs.degradingTo',
  '本地': 'logs.local',
  '使用': 'logs.using',
  '深度': 'logs.deep',
  '子问题': 'logs.subQuestions',
  '时序': 'logs.temporal',
  '对话': 'logs.dialogue',
  '等待图谱数据': 'logs.waitingGraphData',
  '开始规划报告大纲': 'logs.startPlanning',
  '获取模拟上下文': 'logs.getContext',
  '执行工具': 'logs.executingTool',
  '参数': 'logs.parameters',
  '深度洞察检索': 'logs.deepInsightSearch',
  '搜索完成': 'logs.searchComplete',
  '找到': 'logs.found',
  '获取节点详情': 'logs.nodeDetails',
  '完成:': 'logs.completed',
  '条事实': 'logs.factCount',
  '个实体': 'logs.entityCount',
  '条关系': 'logs.relationships',
  '深度采访': 'logs.deepInterview',
  '真实API': 'logs.realAPI',
  '从': 'logs.from',
  '加载了': 'logs.loaded',
  '个人设': 'logs.profile',
  '加载到': 'logs.loaded',
  '选择了': 'logs.selected',
  '进行采访': 'logs.interviewing',
  '生成了': 'logs.generated',
  '个采访问题': 'logs.interviewQuestions',
  '调用批量采访API': 'logs.callingBatchAPI',
  '双平台': 'logs.bothPlatforms',
  '采访API调用异常': 'logs.interviewAPIError',
  '等待命令响应超时': 'logs.waitingTimeout',
  '秒': 'logs.seconds',
  '广度搜索': 'logs.broadSearch',
  '条有效': 'logs.effectiveCount',
  '条历史': 'logs.historicalCount',
  '简单搜索': 'logs.simpleSearch',
  '章节': 'logs.section',
  '生成完成': 'logs.generationComplete',
  '章节已保存': 'logs.sectionSaved',
  '报告生成任务开始': 'logs.reportGenerationStarted',
  '开始规划报告大纲': 'logs.startPlanning',
  '获取模拟上下文信息': 'logs.getContextInfo',
  '大纲规划完成': 'logs.outlineComplete',
  '开始生成章节': 'logs.startGeneratingSection',
  'ReACT 第': 'logs.reactRound',
  '轮思考': 'logs.roundThinking',
  '调用工具': 'logs.callingTool',
  '工具': 'logs.tool',
  '返回结果': 'logs.returnedResult',
  'LLM 响应': 'logs.llmResponse',
  '工具调用': 'logs.toolCall',
  '最终答案': 'logs.finalAnswer',
  '内容生成完成': 'logs.contentGenerated',
  '生成完成': 'logs.generationComplete',
  '报告生成完成': 'logs.reportGenerationComplete',
  '发生错误': 'logs.errorOccurred',
}

/**
 * Hook para traducir logs del backend
 * @returns {function} - Función translateLog(log)
 */
export function useTranslateLog() {
  const { t } = useI18n()
  
  const translateLog = (log) => {
    if (!log || typeof log !== 'string') return log || ''
    
    let result = log
    
    // Reemplazar cada texto chino con su traducción
    for (const [cn, i18nKey] of Object.entries(logKeyMap)) {
      if (result.includes(cn)) {
        const translated = t(i18nKey)
        // Solo reemplazar si la traducción existe y es diferente
        if (translated && translated !== i18nKey) {
          result = result.split(cn).join(translated)
        }
      }
    }
    
    return result
  }
  
  return { translateLog, t }
}
