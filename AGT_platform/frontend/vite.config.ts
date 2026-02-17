import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig(({ mode }) => {
  // Load environment variables - loadEnv works in config file context
  // import.meta.env only works in client code (React components)
  const env = loadEnv(mode, '.', '');

  // Get VITE_API_BASE from environment variable
  // CORRECT VALUES:
  // - Development: "http://localhost:5000"
  // - Production: "https://dia-ai-grader.com"
  const apiBase = env.VITE_API_BASE || "https://dia-ai-grader.com";

  return {
    plugins: [react()],
    build: {
      outDir: "dist",
      sourcemap: false,
      minify: "esbuild",
    },
    server: {
      host: "0.0.0.0",
      port: 5173,
      proxy: {
        "/api": {
          target: apiBase,
          changeOrigin: true,
          secure: true,
        },
      },
    },
  };
});