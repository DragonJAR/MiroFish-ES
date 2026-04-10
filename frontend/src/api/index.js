import axios from 'axios'
import i18n from '../i18n'

// Crear instancia de axios
const service = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL || 'http://localhost:5001',
  timeout: 300000, // 5minutos超时（Generación de ontología可能需要较长时间）
  headers: {
    'Content-Type': 'application/json'
  }
})

// Interceptador de solicitudes
service.interceptors.request.use(
  config => {
    config.headers['Accept-Language'] = i18n.global.locale.value
    return config
  },
  error => {
    console.error('Request error:', error)
    return Promise.reject(error)
  }
)

// 响应拦截器（容错Reintentar机制）
service.interceptors.response.use(
  response => {
    const res = response.data
    
    // 如果Volver的状态码不是success，则抛出Error
    if (!res.success && res.success !== undefined) {
      console.error('API Error:', res.error || res.message || 'Unknown error')
      return Promise.reject(new Error(res.error || res.message || 'Error'))
    }
    
    return res
  },
  error => {
    console.error('Response error:', error)
    
    // Timeout error
    if (error.code === 'ECONNABORTED' && error.message.includes('timeout')) {
      console.error('Request timeout')
      return Promise.reject(new Error(i18n.global.t('common.timeout', 'Timeout')))
    }
    
    // Network error
    if (error.message === 'Network Error') {
      console.error('Network error - please check your connection')
      return Promise.reject(new Error(i18n.global.t('common.networkError', 'Network Error')))
    }
    
    // Generic HTTP error (4xx/5xx) - extract backend error message if available
    let userMessage = i18n.global.t('common.serverError', 'Server error')
    if (error.response && error.response.data) {
      const resData = error.response.data
      if (resData.error) {
        userMessage = resData.error
      } else if (resData.message) {
        userMessage = resData.message
      }
    }
    
    return Promise.reject(new Error(userMessage))
  }
)

// 带Reintentar的请求函数
export const requestWithRetry = async (requestFn, maxRetries = 3, delay = 1000) => {
  for (let i = 0; i < maxRetries; i++) {
    try {
      return await requestFn()
    } catch (error) {
      if (i === maxRetries - 1) throw error
      
      console.warn(`Request failed, retrying (${i + 1}/${maxRetries})...`)
      await new Promise(resolve => setTimeout(resolve, delay * Math.pow(2, i)))
    }
  }
}

export default service
