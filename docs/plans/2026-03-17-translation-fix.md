# Translation Fix Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use executing-plans to implement this plan task-by-task.

**Goal:** Fix all hardcoded Chinese/Spanish/English texts in Vue components by adding i18n keys and updating components to use $t().

**Architecture:** 
- Add missing i18n keys to `frontend/src/i18n/es.json`
- Update Vue components to use `$t()` for all user-facing text
- Verify no Chinese text visible in browser

**Tech Stack:** Vue 3, vue-i18n, Vite

---

## Pre-requisites

Before starting, verify current state:
```bash
# Check current es.json exists
ls -la frontend/src/i18n/es.json

# Check useTranslateLog exists (for backend log translation)
ls -la frontend/src/composables/useTranslateLog.js
```

---

## Task 1: Add missing i18n keys to es.json

**Files:**
- Modify: `frontend/src/i18n/es.json`

**Step 1: Read current es.json structure**

```bash
head -50 frontend/src/i18n/es.json
```

**Step 2: Add Step4Report keys (add after existing step4 keys)**

Add these keys to es.json under the step4 section:

```json
"step4": {
  "predictionScenario": "Escenario de Predicción",
  "currentKeyMemories": "Memorias Clave Actuales ({count})",
  "coreEntities": "Entidades Core ({count})",
  "relationChains": "Cadenas de Relación ({count})",
  "subQuestions": "Sub-preguntas ({count})",
  "latestKeyFacts": "Hechos Clave Recientes en Memorias Temporales",
  "totalFacts": "Total: {n}",
  "collapse": "Contraer ▲",
  "expandAll": "Expandir todo {n} ▼",
  "totalEntities": "Total: {n}",
  "totalRelations": "Total: {n}",
  "totalSubQueries": "Total: {n}",
  "noKeyMemories": "No hay memorias clave actuales",
  "noEntities": "No hay entidades core",
  "noRelations": "No hay cadenas de relación",
  "activeMemories": "Memorias Activas ({count})",
  "historicalMemories": "Memorias Históricas ({count})",
  "involvedEntities": "Entidades Involucradas ({count})",
  "noActiveMemories": "No hay memorias activas",
  "noHistoricalMemories": "No hay memorias históricas",
  "noInvolvedEntities": "No hay entidades involucradas",
  "subQueryAnalysis": "Análisis de Sub-preguntas de Consulta Drift"
}
```

**Step 3: Add Process and MainView keys**

Add under existing keys:

```json
"process": {
  "graphBuildTaskStarted": "Tarea de Construcción de Grafo iniciada...",
  "envSetupInDevelopment": "Función de configuración de entorno en desarrollo..."
},
"common": {
  "error": "Error",
  "ready": "Listo",
  "buildingGraph": "Construyendo Grafo",
  "generatingOntology": "Generando Ontología",
  "initializing": "Inicializando",
  "projectFailed": "Proyecto falló"
}
```

**Step 4: Verify JSON is valid**

```bash
cd frontend && npm run build 2>&1 | head -20
```

Expected: No JSON errors

---

## Task 2: Fix Step4Report.vue Chinese texts

**Files:**
- Modify: `frontend/src/components/Step4Report.vue:980-1271`

**Step 1: Read the file around lines 980-1271 to find exact Chinese strings**

```bash
grep -n "预测场景\|当前关键记忆\|核心实体\|关系链\|子问题\|收起\|展开\|暂无\|涉及实体\|历史记忆" frontend/src/components/Step4Report.vue
```

**Step 2: Replace Chinese strings with $t() calls**

The Chinese strings are inside `h()` calls in render functions. Need to change from:
```javascript
h('div', {}, '预测场景: ')
```

To:
```javascript
h('div', {}, t('step4.predictionScenario') + ': ')
```

Or use computed properties with t() for dynamic values like:
```javascript
h('span', {}, t('step4.currentKeyMemories', { count: data.length }))
```

**Step 3: Test build**

```bash
cd frontend && npm run build 2>&1 | tail -10
```

Expected: Build succeeds

---

## Task 3: Fix Process.vue Chinese text

**Files:**
- Modify: `frontend/src/views/Process.vue:685`

**Step 1: Find the exact line**

```bash
grep -n "图谱构建任务已启动" frontend/src/views/Process.vue
```

**Step 2: Replace with i18n**

From:
```javascript
buildProgress.value.message = '图谱构建任务已启动...'
```

To:
```javascript
buildProgress.value.message = t('process.graphBuildTaskStarted')
```

**Step 3: Find and fix Spanish hardcoded text (line 481)**

```bash
grep -n "Función de configuración" frontend/src/views/Process.vue
```

Replace with:
```javascript
t('process.envSetupInDevelopment')
```

**Step 4: Verify build**

```bash
cd frontend && npm run build 2>&1 | tail -10
```

---

## Task 4: Fix MainView.vue hardcoded English

**Files:**
- Modify: `frontend/src/views/MainView.vue:141-145,275`

**Step 1: Find the statusText computed property**

```bash
grep -n "statusText\|Error\|Ready\|Building Graph" frontend/src/views/MainView.vue | head -20
```

**Step 2: Replace hardcoded strings with t() calls**

From:
```javascript
const statusText = computed(() => {
  if (props.status === 'error') return 'Error'
  if (props.status === 'ready') return 'Ready'
  // etc...
})
```

To:
```javascript
const statusText = computed(() => {
  if (props.status === 'error') return t('common.error')
  if (props.status === 'ready') return t('common.ready')
  // etc...
})
```

**Step 3: Find "Project failed" in switch case**

```bash
grep -n "Project failed" frontend/src/views/MainView.vue
```

Replace with: `t('common.projectFailed')`

**Step 4: Verify build**

```bash
cd frontend && npm run build 2>&1 | tail -10
```

---

## Task 5: Final verification

**Step 1: Search for any remaining Chinese characters in Vue files**

```bash
grep -r "[\u4e00-\u9fff]" frontend/src/components/*.vue frontend/src/views/*.vue 2>/dev/null | grep -v "i18n\|//\|/\*"
```

Expected: No output (or only false positives like variable names)

**Step 2: Build production**

```bash
cd frontend && npm run build
```

Expected: Build succeeds

**Step 3: Test in browser (manual)**

```bash
echo "Start dev server: npm run frontend"
echo "Open http://localhost:3000"
echo "Navigate through all steps and verify no Chinese text visible"
```

---

## Rollback Plan (if issues)

If build fails or errors appear:

```bash
git checkout -- frontend/src/i18n/es.json frontend/src/components/Step4Report.vue frontend/src/views/Process.vue frontend/src/views/MainView.vue
```

---

## Success Criteria

1. ✅ All Chinese text in Vue components replaced with $t() calls
2. ✅ All hardcoded English status messages use i18n
3. ✅ Production build succeeds
4. ✅ No Chinese text visible in browser UI
