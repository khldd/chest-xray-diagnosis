import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// In dev, /api/* is proxied to the FastAPI server, so the browser never makes a cross-origin call.
// In production (Docker Compose) nginx does the same proxying.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: process.env.API_URL ?? "http://localhost:8000",
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ""),
      },
    },
  },
});
