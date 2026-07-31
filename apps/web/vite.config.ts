import { configDefaults, defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    proxy: {
      "/api": "http://localhost:8000",
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
});
