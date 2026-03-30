import axios from 'axios'

// 创建axios实例
const service = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL || 'http://localhost:5001',
  timeout: 300000, // 5分钟超时（本体生成可能需要较长时间）
  headers: {
    'Content-Type': 'application/json'
  }
})

// 请求拦截器
service.interceptors.request.use(
  config => {
    return config
  },
  error => {
    console.error('Request error:', error)
    return Promise.reject(error)
  }
)

// 响应拦截器（容错重试机制）
service.interceptors.response.use(
  response => {
    const res = response.data
    
    // 如果返回的状态码不是success，则抛出错误
    if (!res.success && res.success !== undefined) {
      console.error('API Error:', res.error || res.message || 'Unknown error')
      return Promise.reject(new Error(res.error || res.message || 'Error'))
    }
    
    return res
  },
  error => {
    console.error('Response error:', error)
    
    // Caso 2: Error HTTP (4xx/5xx)
    if (error.code === 'ECONNABORTED' && error.message.includes('timeout')) {
      console.error('Request timeout')
      return Promise.reject(new Error('Timeout: El servidor tardó demasiado en responder'))
    }
    
    if (error.message === 'Network Error') {
      console.error('Network error - please check your connection')
      return Promise.reject(new Error('Error de red: Verifica tu conexión'))
    }
    
    // Caso 3: Error HTTP genérico (4xx/5xx)
    // ✅ ANTES de rechazar, intenta extraer mensaje del backend
    let userMessage = 'Error en la solicitud al servidor'
    if (error.response && error.response.data) {
      const resData = error.response.data
      if (resData.error) {
        userMessage = resData.error  // ← USA EL MENSAJE REAL
      } else if (resData.message) {
        userMessage = resData.message
      }
    }
    
    return Promise.reject(new Error(userMessage))
  }
)

// 带重试的请求函数
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
