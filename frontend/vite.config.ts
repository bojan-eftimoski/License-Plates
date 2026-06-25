import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// GitHub Pages serves a project site under /<repo>/, so the static build is based there.
// Override with VITE_BASE=/ for local preview at the root.
export default defineConfig({
  base: process.env.VITE_BASE ?? '/License-Plates/',
  plugins: [react()],
})
