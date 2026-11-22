import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/analyze': 'http://localhost:8000',
      '/analyze_video': 'http://localhost:8000',
      '/simulate': 'http://localhost:8000',
      '/ml': 'http://localhost:8000',
    }
  }
})
