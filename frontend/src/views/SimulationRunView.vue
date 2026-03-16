<template>
  <div class="main-view">
    <!-- Header -->
    <header class="app-header">
      <div class="header-left">
        <div class="brand" @click="router.push('/')">MIROFISH</div>
      </div>
      
      <div class="header-center">
        <div class="view-switcher">
          <button
            v-for="mode in ['graph', 'split', 'workbench']"
            :key="mode"
            class="switch-btn"
            :class="{ active: viewMode === mode }"
            @click="viewMode = mode"
          >
            {{ viewModeLabels[mode] }}
          </button>
        </div>
      </div>

      <div class="header-right">
        <div class="workflow-step">
          <span class="step-num">{{ $t('mainView.step3') }}</span>
          <span class="step-name">{{ t('mainView.step3') }}</span>
        </div>
        <div class="step-divider"></div>
        <span class="status-indicator" :class="statusClass">
          <span class="dot"></span>
          {{ statusText }}
        </span>
      </div>
    </header>

    <!-- Main Content Area -->
    <main class="content-area">
      <!-- Left Panel: Graph -->
      <div class="panel-wrapper left" :style="leftPanelStyle">
        <GraphPanel 
          :graphData="graphData"
          :loading="graphLoading"
          :currentPhase="3"
          :isSimulating="isSimulating"
          @refresh="refreshGraph"
          @toggle-maximize="toggleMaximize('graph')"
        />
      </div>

      <!-- Right Panel: Step3 开始模拟 -->
      <div class="panel-wrapper right" :style="rightPanelStyle">
        <Step3Simulation
          :simulationId="currentSimulationId"
          :maxRounds="maxRounds"
          :minutesPerRound="minutesPerRound"
          :projectData="projectData"
          :graphData="graphData"
          :systemLogs="systemLogs"
          @go-back="handleGoBack"
          @next-step="handleNextStep"
          @add-log="addLog"
          @update-status="updateStatus"
        />
      </div>
    </main>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, onUnmounted, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { useI18n } from 'vue-i18n'
import GraphPanel from '../components/GraphPanel.vue'
import Step3Simulation from '../components/Step3Simulation.vue'
import { getProject, getGraphData } from '../api/graph'
import { getSimulation, getSimulationConfig, stopSimulation, closeSimulationEnv, getEnvStatus } from '../api/simulation'

const route = useRoute()
const router = useRouter()
const { t } = useI18n()

// Props
const props = defineProps({
  simulationId: String
})

// Layout State
const viewMode = ref('split')

// Data State
const currentSimulationId = ref(route.params.simulationId)
// 直接在初始化时从 query 参数获取 maxRounds，确保子组件能立即获取到值
const maxRounds = ref(route.query.maxRounds ? parseInt(route.query.maxRounds) : null)
const minutesPerRound = ref(30) // 默认每轮30分钟
const projectData = ref(null)
const graphData = ref(null)
const graphLoading = ref(false)
const systemLogs = ref([])
const currentStatus = ref('processing') // processing | completed | error

// --- Computed Layout Styles ---
const viewModeLabels = computed(() => ({
  graph: t('mainView.graphView'),
  split: t('mainView.splitView'),
  workbench: t('mainView.workbenchView')
}))

const leftPanelStyle = computed(() => {
  if (viewMode.value === 'graph') return { width: '100%', opacity: 1, transform: 'translateX(0)' }
  if (viewMode.value === 'workbench') return { width: '0%', opacity: 0, transform: 'translateX(-20px)' }
  return { width: '50%', opacity: 1, transform: 'translateX(0)' }
})

const rightPanelStyle = computed(() => {
  if (viewMode.value === 'workbench') return { width: '100%', opacity: 1, transform: 'translateX(0)' }
  if (viewMode.value === 'graph') return { width: '0%', opacity: 0, transform: 'translateX(20px)' }
  return { width: '50%', opacity: 1, transform: 'translateX(0)' }
})

// --- Status Computed ---
const statusClass = computed(() => {
  return currentStatus.value
})

const statusText = computed(() => {
  if (currentStatus.value === 'error') return t('common.error')
  if (currentStatus.value === 'completed') return t('common.completed')
  return t('common.running')
})

const isSimulating = computed(() => currentStatus.value === 'processing')

// --- Helpers ---
const addLog = (msg) => {
  const time = new Date().toLocaleTimeString('en-US', { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' }) + '.' + new Date().getMilliseconds().toString().padStart(3, '0')
  systemLogs.value.push({ time, msg })
  if (systemLogs.value.length > 200) {
    systemLogs.value.shift()
  }
}

const updateStatus = (status) => {
  currentStatus.value = status
}

// --- Layout Methods ---
const toggleMaximize = (target) => {
  if (viewMode.value === target) {
    viewMode.value = 'split'
  } else {
    viewMode.value = target
  }
}

const handleGoBack = async () => {
  // 在返回 Step 2 之前，先关闭正在运行的模拟
  addLog(t('logs.preparingReturnStep2'))

  // 停止轮询
  stopGraphRefresh()

  try {
    // 先尝试优雅关闭模拟环境
    const envStatusRes = await getEnvStatus({ simulation_id: currentSimulationId.value })

    if (envStatusRes.success && envStatusRes.data?.env_alive) {
      addLog(t('logs.closingSimEnv'))
      try {
        await closeSimulationEnv({
          simulation_id: currentSimulationId.value,
          timeout: 10
        })
        addLog(t('logs.simEnvClosed'))
      } catch (closeErr) {
        addLog(t('logs.closingSimFailed'))
        try {
          await stopSimulation({ simulation_id: currentSimulationId.value })
          addLog(t('logs.simForceStopped'))
        } catch (stopErr) {
          addLog(`${t('logs.forceStopFailed')}: ${stopErr.message}`)
        }
      }
    } else {
      // 环境未运行，检查是否需要停止进程
      if (isSimulating.value) {
        addLog(t('logs.stoppingSimProcess'))
        try {
          await stopSimulation({ simulation_id: currentSimulationId.value })
          addLog(t('logs.simStopped'))
        } catch (err) {
          addLog(`${t('logs.stopSimFailed')}: ${err.message}`)
        }
      }
    }
  } catch (err) {
    addLog(`${t('logs.statusCheckFailed')}: ${err.message}`)
  }

  // 返回到 Step 2 (环境搭建)
  router.push({ name: 'Simulation', params: { simulationId: currentSimulationId.value } })
}

const handleNextStep = () => {
  // Step3Simulation 组件会直接处理报告生成和路由跳转
  // 这个方法仅作为备用
  addLog(t('logs.enteringStep4'))
}

// --- Data Logic ---
const loadSimulationData = async () => {
  try {
    addLog(`${t('logs.simDataLoaded')}: ${currentSimulationId.value}`)

    // 获取 simulation 信息
    const simRes = await getSimulation(currentSimulationId.value)
    if (simRes.success && simRes.data) {
      const simData = simRes.data

      // 获取 simulation config 以获取 minutes_per_round
      try {
        const configRes = await getSimulationConfig(currentSimulationId.value)
        if (configRes.success && configRes.data?.time_config?.minutes_per_round) {
          minutesPerRound.value = configRes.data.time_config.minutes_per_round
          addLog(`${t('logs.timeConfigSet')} ${minutesPerRound.value} ${t('step2.minutes')}`)
        }
      } catch (configErr) {
        addLog(`${t('logs.timeConfigError')}: ${minutesPerRound.value}${t('logs.defaultTimeConfig')}`)
      }

      // 获取 project 信息
      if (simData.project_id) {
        const projRes = await getProject(simData.project_id)
        if (projRes.success && projRes.data) {
          projectData.value = projRes.data
          addLog(`${t('logs.projectLoadedSuccessfully')}: ${projRes.data.project_id}`)

          // 获取 graph 数据
          if (projRes.data.graph_id) {
            await loadGraph(projRes.data.graph_id)
          }
        }
      }
    } else {
      addLog(`${t('logs.simDataLoadFailed')}: ${simRes.error || t('common.error')}`)
    }
  } catch (err) {
    addLog(`${t('logs.simDataException')}: ${err.message}`)
  }
}

const loadGraph = async (graphId) => {
  // 当正在模拟时，自动刷新不显示全屏 loading，以免闪烁
  // 手动刷新或初始加载时显示 loading
  if (!isSimulating.value) {
    graphLoading.value = true
  }

  try {
    const res = await getGraphData(graphId)
    if (res.success) {
      graphData.value = res.data
      if (!isSimulating.value) {
        addLog(t('logs.graphDataLoaded'))
      }
    }
  } catch (err) {
    addLog(`${t('logs.graphLoadFailed')}: ${err.message}`)
  } finally {
    graphLoading.value = false
  }
}

const refreshGraph = () => {
  if (projectData.value?.graph_id) {
    loadGraph(projectData.value.graph_id)
  }
}

// --- Auto Refresh Logic ---
let graphRefreshTimer = null

const startGraphRefresh = () => {
  if (graphRefreshTimer) return
  addLog(t('logs.graphRealtimeRefreshEnabled'))
  // 立即刷新一次，然后每30秒刷新
  graphRefreshTimer = setInterval(refreshGraph, 30000)
}

const stopGraphRefresh = () => {
  if (graphRefreshTimer) {
    clearInterval(graphRefreshTimer)
    graphRefreshTimer = null
    addLog(t('logs.graphRealtimeRefreshStopped'))
  }
}

watch(isSimulating, (newValue) => {
  if (newValue) {
    startGraphRefresh()
  } else {
    stopGraphRefresh()
  }
}, { immediate: true })

onMounted(() => {
  addLog(t('logs.simRunViewInit'))

  // 记录 maxRounds 配置（值已在初始化时从 query 参数获取）
  if (maxRounds.value) {
    addLog(`${t('logs.customSimRounds')}: ${maxRounds.value}`)
  }

  loadSimulationData()
})

onUnmounted(() => {
  stopGraphRefresh()
})
</script>

<style scoped>
.main-view {
  height: 100vh;
  display: flex;
  flex-direction: column;
  background: #FFF;
  overflow: hidden;
  font-family: 'Space Grotesk', 'Noto Sans SC', system-ui, sans-serif;
}

/* Header */
.app-header {
  height: 60px;
  border-bottom: 1px solid #EAEAEA;
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0 24px;
  background: #FFF;
  z-index: 100;
  position: relative;
}

.header-center {
  position: absolute;
  left: 50%;
  transform: translateX(-50%);
}

.brand {
  font-family: 'JetBrains Mono', monospace;
  font-weight: 800;
  font-size: 18px;
  letter-spacing: 1px;
  cursor: pointer;
}

.view-switcher {
  display: flex;
  background: #F5F5F5;
  padding: 4px;
  border-radius: 6px;
  gap: 4px;
}

.switch-btn {
  border: none;
  background: transparent;
  padding: 6px 16px;
  font-size: 12px;
  font-weight: 600;
  color: #666;
  border-radius: 4px;
  cursor: pointer;
  transition: all 0.2s;
}

.switch-btn.active {
  background: #FFF;
  color: #000;
  box-shadow: 0 2px 4px rgba(0,0,0,0.05);
}

.header-right {
  display: flex;
  align-items: center;
  gap: 16px;
}

.workflow-step {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 14px;
}

.step-num {
  font-family: 'JetBrains Mono', monospace;
  font-weight: 700;
  color: #999;
}

.step-name {
  font-weight: 700;
  color: #000;
}

.step-divider {
  width: 1px;
  height: 14px;
  background-color: #E0E0E0;
}

.status-indicator {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 12px;
  color: #666;
  font-weight: 500;
}

.dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: #CCC;
}

.status-indicator.processing .dot { background: #FF5722; animation: pulse 1s infinite; }
.status-indicator.completed .dot { background: #4CAF50; }
.status-indicator.error .dot { background: #F44336; }

@keyframes pulse { 50% { opacity: 0.5; } }

/* Content */
.content-area {
  flex: 1;
  display: flex;
  position: relative;
  overflow: hidden;
}

.panel-wrapper {
  height: 100%;
  overflow: hidden;
  transition: width 0.4s cubic-bezier(0.25, 0.8, 0.25, 1), opacity 0.3s ease, transform 0.3s ease;
  will-change: width, opacity, transform;
}

.panel-wrapper.left {
  border-right: 1px solid #EAEAEA;
}
</style>

