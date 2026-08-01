import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
import { loadEnv } from "vite";
import { configDefaults, defineConfig } from "vitest/config";

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), "");

  return {
    plugins: [react(), tailwindcss()],
    server: {
      proxy: {
        "/api": env.VITE_API_PROXY_TARGET ?? "http://localhost:8000",
      },
    },
    build: {
      target: "es2022",
    },
    test: {
      environment: "jsdom",
      setupFiles: "./src/test/setup.ts",
      /* e2e/ holds Playwright specs, whose test API vitest cannot collect. */
      include: ["src/**/*.test.{ts,tsx}"],
      exclude: [...configDefaults.exclude, "e2e/**"],
    },
  };
});
