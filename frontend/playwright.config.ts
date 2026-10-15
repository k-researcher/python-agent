import { defineConfig } from "@playwright/test";

const baseURL = process.env.UI_E2E_BASE_URL ?? "http://127.0.0.1:8097";

export default defineConfig({
  testDir: "./e2e",
  timeout: 30_000,
  fullyParallel: true,
  reporter: [["list"]],
  snapshotPathTemplate:
    "{testDir}/__screenshots__/{projectName}/{testFilePath}/{arg}{ext}",
  use: {
    baseURL,
    browserName: "chromium",
    deviceScaleFactor: 1,
    trace: "retain-on-failure",
    screenshot: "only-on-failure",
    video: "retain-on-failure",
  },
  projects: [
    {
      name: "desktop-light",
      use: { viewport: { width: 1440, height: 900 }, colorScheme: "light" },
    },
    {
      name: "desktop-dark",
      use: { viewport: { width: 1440, height: 900 }, colorScheme: "dark" },
    },
    {
      name: "laptop-light",
      use: { viewport: { width: 1100, height: 800 }, colorScheme: "light" },
    },
    {
      name: "laptop-dark",
      use: { viewport: { width: 1100, height: 800 }, colorScheme: "dark" },
    },
    {
      name: "tablet-light",
      use: { viewport: { width: 760, height: 800 }, colorScheme: "light" },
    },
    {
      name: "tablet-dark",
      use: { viewport: { width: 760, height: 800 }, colorScheme: "dark" },
    },
    {
      name: "mobile-light",
      use: { viewport: { width: 390, height: 844 }, colorScheme: "light" },
    },
    {
      name: "mobile-dark",
      use: { viewport: { width: 390, height: 844 }, colorScheme: "dark" },
    },
  ],
});
