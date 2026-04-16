import { createApp } from 'vue'
import App from './App.vue'
import router from './router'
import i18n from './i18n'

// Suppress vue-i18n "Not found parent scope" warning — false positive when
// globalInjection: true is set. The fallback to global scope is intentional and correct.
// This warning is a known issue in vue-i18n 9.x Composition API mode.
const originalWarn = console.warn
console.warn = (...args) => {
  if (args[0] && typeof args[0] === 'string' && args[0].includes('Not found parent scope')) {
    return
  }
  originalWarn.apply(console, args)
}

const app = createApp(App)

app.use(router)
app.use(i18n)

app.mount('#app')
