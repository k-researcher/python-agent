import { test, expect } from "@playwright/test";
import { openApp } from "./helpers";

test.describe("agent settings keyboard interactions", () => {
  test("opens via Enter and focuses first enabled select", async ({ page }, testInfo) => {
    await openApp(page, testInfo);
    const button = page.getByTestId("agent-settings-button");
    await expect(button).toBeVisible();

    await button.focus();
    await page.keyboard.press("Enter");

    await expect(page.getByTestId("agent-settings-panel")).toBeVisible();
    const firstEnabledSelect = page.getByTestId("agent-settings-panel").locator("select:not(:disabled)").first();
    await expect(firstEnabledSelect).toBeFocused();
  });

  test("closes via Escape and returns focus to button", async ({ page }, testInfo) => {
    await openApp(page, testInfo);
    const button = page.getByTestId("agent-settings-button");

    await button.click();
    await expect(page.getByTestId("agent-settings-panel")).toBeVisible();

    await page.keyboard.press("Escape");
    await expect(page.getByTestId("agent-settings-panel")).not.toBeVisible();
    await expect(button).toBeFocused();
  });

  test("closes via Tab out of popover without intercepting focus", async ({ page }, testInfo) => {
    await openApp(page, testInfo);
    const button = page.getByTestId("agent-settings-button");
    const nextAction = page.getByRole("button", { name: /^Новая идея/ });

    await button.click();
    const panel = page.getByTestId("agent-settings-panel");
    await expect(panel).toBeVisible();

    const lastEnabledSelect = panel.locator("select:not(:disabled)").last();
    await lastEnabledSelect.focus();

    await page.keyboard.press("Tab");

    await expect(panel).not.toBeVisible();
    await expect(nextAction).toBeFocused();
  });
});
