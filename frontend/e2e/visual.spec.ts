import { test, expect } from "@playwright/test";
import { openApp, ensureLeftPanelOpen } from "./helpers";

const VISUAL = process.env.UI_E2E_VISUAL === "1";

test.describe("visual snapshots (UI_E2E_VISUAL=1)", () => {
  test("home view snapshot", async ({ page }, testInfo) => {
    test.skip(!VISUAL, "set UI_E2E_VISUAL=1 to generate/verify references");
    await openApp(page, testInfo);
    await expect(page.getByTestId("composer-input")).toBeVisible();
    await expect(page).toHaveScreenshot("home.png", { animations: "disabled" });
  });

  test("session view snapshot", async ({ page }, testInfo) => {
    test.skip(!VISUAL, "set UI_E2E_VISUAL=1 to generate/verify references");
    await openApp(page, testInfo);
    await ensureLeftPanelOpen(page);
    await page.locator('[data-node-key="n:s1"]').click();
    await expect(page).toHaveURL(/\/s\/s1/);
    await expect(page.getByTestId("session-title")).toBeVisible();
    await expect(page.getByTestId("composer-input")).toBeVisible();
    await expect(page).toHaveScreenshot("session.png", { animations: "disabled" });
  });
});
