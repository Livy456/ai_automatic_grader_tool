import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig(({ mode }) => {
  // Load environment variables - loadEnv works in config file context
  // import.meta.env only works in client code (React components)
  const env = loadEnv(mode, '.', '');

  // Get VITE_API_BASE from environment variable
  // CORRECT VALUES:
  // - Development: "http://localhost:5000"
  // - Production (ALB): "https://api.dia-ai-grader.com"
  // - Legacy same-origin API: "https://dia-ai-grader.com"
  // Proxy /api to local Flask when unset (host SPA + Docker backend). Override with VITE_API_BASE.
  const apiBase = env.VITE_API_BASE || "http://localhost:5000";
  const proxySecure = apiBase.startsWith("https://");

  return {
    plugins: [react()],
    build: {
      outDir: "dist",
      sourcemap: false,
      minify: "esbuild",
    },
    server: {
      host: "0.0.0.0",
      // Host `npm run dev`: use 5174 so Docker Compose can keep publishing frontend on 5173.
      port: 5174,
      strictPort: false,
      proxy: {
        "/api": {
          target: apiBase,
          changeOrigin: true,
          // Local Flask is http:// — secure:true can break the proxy; only verify TLS for https targets.
          secure: proxySecure,
          // See docs/BUG_REPORT_ADMIN_WRITE_401.md — http-proxy-middleware can drop
          // Authorization when changeOrigin is true; re-attach on every proxied request.
          configure(proxy) {
            proxy.on("proxyReq", (proxyReq, req) => {
              const auth = req.headers.authorization;
              if (auth) {
                proxyReq.setHeader("Authorization", auth);
              }
            });
          },
        },
      },
    },
  };
});