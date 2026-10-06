import vue from "@vitejs/plugin-vue";
import { defineConfig } from "vitest/config";

export default defineConfig({
  plugins: [vue()],
  // A literal address: no name lookup, so tests also run where DNS is closed (the shell sandbox).
  server: { host: "127.0.0.1" },
  test: { environment: "happy-dom", include: ["src/**/*.test.ts"] },
});
