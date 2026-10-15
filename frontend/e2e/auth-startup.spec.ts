import { test, expect } from "@playwright/test";
import { openApp, BASE_URL } from "./helpers";

test.describe("auth startup", () => {
  test("cold start at / shows composer and no login form", async ({ page }, testInfo) => {
    await openApp(page, testInfo, { path: "/" });
    await expect(page).toHaveURL(`${BASE_URL}/`);
    await expect(page.getByTestId("composer-input")).toBeVisible();
    await expect(page.locator("[data-project]")).toHaveCount(1);
    await expect(page.locator('[data-node-key="n:s1"]')).toHaveCount(1);
    await expect(page.locator(".login-container")).toHaveCount(0);
  });

  test("cold start at /s/s1 shows session composer and no login form", async ({ page }, testInfo) => {
    await openApp(page, testInfo, { path: "/s/s1" });
    await expect(page).toHaveURL(`${BASE_URL}/s/s1`);
    await expect(page.getByTestId("composer-input")).toBeVisible();
    await expect(page.locator("[data-project]")).toHaveCount(1);
    await expect(page.locator('[data-node-key="n:s1"]')).toHaveCount(1);
    await expect(page.locator(".login-container")).toHaveCount(0);
  });

  test("reload existing session /s/s1", async ({ page }, testInfo) => {
    await openApp(page, testInfo, { path: "/s/s1" });
    await page.reload();
    await expect(page).toHaveURL(`${BASE_URL}/s/s1`);
    await expect(page.getByTestId("composer-input")).toBeVisible();
    await expect(page.locator("[data-project]")).toHaveCount(1);
    await expect(page.locator('[data-node-key="n:s1"]')).toHaveCount(1);
    await expect(page.locator(".login-container")).toHaveCount(0);
  });
});
