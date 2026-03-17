import { createI18n } from 'vue-i18n'
import es from './es.json'
// zh.json removed - Spanish is the only language
// import zh from './zh.json'

const i18n = createI18n({
  legacy: false,
  locale: 'es',
  fallbackLocale: 'es',  // Fallback to Spanish, not Chinese
  messages: { es }  // Only Spanish messages
})

export default i18n
