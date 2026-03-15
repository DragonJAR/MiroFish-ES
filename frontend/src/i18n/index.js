import { createI18n } from 'vue-i18n'
import es from './es.json'
import zh from './zh.json'

const i18n = createI18n({
  legacy: false,
  locale: 'es',
  fallbackLocale: 'zh',
  messages: { es, zh }
})

export default i18n
