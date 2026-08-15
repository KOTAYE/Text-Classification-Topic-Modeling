import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    // Talk to the FastAPI process during development so the browser sees one
    // origin and there is nothing to configure in the app itself.
    proxy: {
      '/api': 'http://localhost:8000',
    },
  },
})
