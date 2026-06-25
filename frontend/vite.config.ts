import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Local dev (`vite`) serves at root; the production build (`vite build`) is based at
// /<repo>/ for GitHub Pages (override with VITE_BASE if the repo name changes).
export default defineConfig(({ command }) => ({
  base: command === 'build' ? (process.env.VITE_BASE ?? '/License-Plates/') : '/',
  plugins: [react()],
}))
