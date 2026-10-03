import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// In development the UI runs on :5173 and proxies /api to the FastAPI backend on :8000.
// In production set VITE_API_URL (see .env.example) or serve the built UI from FastAPI (same origin).
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: { "/api": { target: process.env.VITE_DEV_PROXY || "http://127.0.0.1:8000", changeOrigin: true } },
  },
  build: { outDir: "dist", chunkSizeWarningLimit: 1200 },
});
