import { defineConfig } from 'vite' // Vite config helper
import react from '@vitejs/plugin-react' // React plugin loader

export default defineConfig({ // Main Vite config
  plugins: [react()], // Enable React support
  server: { // Dev server settings
    port: 5173, // Listen on port 5173
    proxy: { // Proxy configuration
      '/api': { // API route prefix
        target: 'http://localhost:8000', // Backend URL
        changeOrigin: true // Rewrite origin header
      }
    }
  }
})
