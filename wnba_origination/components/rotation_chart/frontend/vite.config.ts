import { defineConfig } from "vite"
import react from "@vitejs/plugin-react"

export default defineConfig({
  plugins: [react()],
  base: "",          // relative paths so Streamlit can serve from any location
  build: {
    outDir: "build",
    assetsDir: "static",
  },
  server: {
    port: 3001,
  },
})
