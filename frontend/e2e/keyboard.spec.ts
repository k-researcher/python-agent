import { test, expect } from "@playwright/test";
import {
  openApp,
  ensureRightPanelOpen,
  ensureLeftPanelOpen,
} from "./helpers";

const RIGHT_TABS = ["context", "tools", "agents", "audit", "monitor"];

async function focusFirstRightTab(page: import("@playwright/test").Page): Promise<void> {
  const tabs = page.locator('[role="tab"][data-tab]');
  await expect(tabs).toHaveCount(5);
  await tabs.first().focus();
  await expect(tabs.first()).toBeFocused();
}

async function assertRightTabActive(page: import("@playwright/test").Page, id: string): Promise<void> {
  await expect(page.locator(`[data-tab="${id}"]`)).toHaveAttribute("aria-selected", "true");
  await expect(page.locator(`[data-panel="${id}"]`)).toBeVisible();
  await expect(page.locator(`[data-tab="${id}"]`)).toBeFocused();
}

async function openSessionViaSidebar(
  page: import("@playwright/test").Page,
  info: import("@playwright/test").TestInfo,
  id: string,
): Promise<void> {
  await openApp(page, info);
  await ensureLeftPanelOpen(page);
  await page.locator(`[data-node-key="n:${id}"]`).click();
  await expect(page).toHaveURL(new RegExp(`/s/${id}`));
}

test.describe("keyboard navigation", () => {
  test("right panel tabs advance with ArrowRight and wrap around", async ({ page }, testInfo) => {
    await openApp(page, testInfo);
    await ensureRightPanelOpen(page);
    await focusFirstRightTab(page);
    await assertRightTabActive(page, "context");

    for (let i = 1; i <= RIGHT_TABS.length; i++) {
      await page.keyboard.press("ArrowRight");
      await assertRightTabActive(page, RIGHT_TABS[i % RIGHT_TABS.length]);
    }
  });

  test("right panel tabs move back with ArrowLeft and wrap around", async ({ page }, testInfo) => {
    await openApp(page, testInfo);
    await ensureRightPanelOpen(page);
    await focusFirstRightTab(page);
    await assertRightTabActive(page, "context");

    await page.keyboard.press("ArrowLeft");
    await assertRightTabActive(page, "monitor");
    await page.keyboard.press("ArrowLeft");
    await assertRightTabActive(page, "audit");
    await page.keyboard.press("ArrowLeft");
    await assertRightTabActive(page, "agents");
    await page.keyboard.press("ArrowLeft");
    await assertRightTabActive(page, "tools");
    await page.keyboard.press("ArrowLeft");
    await assertRightTabActive(page, "context");
  });

  test("right panel tabs jump to first and last with Home and End", async ({ page }, testInfo) => {
    await openApp(page, testInfo);
    await ensureRightPanelOpen(page);
    await focusFirstRightTab(page);

    await page.keyboard.press("End");
    await assertRightTabActive(page, "monitor");
    await page.keyboard.press("Home");
    await assertRightTabActive(page, "context");
  });

  test("hidden right panel opens and closes via focused toggle and Enter", async ({ page }, testInfo) => {
    await openApp(page, testInfo);
    const toggle = page.locator('[data-toggle="right"]');
    const panel = page.locator(".app-shell__right");

    if (await toggle.isVisible()) {
      await expect(panel).toBeHidden();
      await toggle.focus();
      await expect(toggle).toBeFocused();

      await page.keyboard.press("Enter");
      await expect(panel).toBeVisible();
      await expect(toggle).toHaveAttribute("aria-expanded", "true");
      await expect(toggle).toBeFocused();

      await page.keyboard.press("Enter");
      await expect(panel).toBeHidden();
      await expect(toggle).toHaveAttribute("aria-expanded", "false");
    } else {
      await expect(panel).toBeVisible();
    }
  });

  test("composer Shift+Enter inserts a newline and does not send the draft", async ({ page }, testInfo) => {
    await openApp(page, testInfo);
    const input = page.getByTestId("composer-input");
    await expect(input).toBeVisible();

    await input.fill("первая строка");
    await input.press("Shift+Enter");
    await input.pressSequentially("вторая строка");

    await expect(input).toHaveValue("первая строка\nвторая строка");
    await expect(page.getByTestId("composer-send")).toBeEnabled();
    await expect(input).toBeFocused();
    await expect(page.locator('[data-testid="home-error"]')).toHaveCount(0);
  });

  test("composer Enter takes the failed-send path and restores the draft", async ({ page }, testInfo) => {
    await openSessionViaSidebar(page, testInfo, "s1");
    const input = page.getByTestId("composer-input");
    const text = "не сохранившееся сообщение";
    await input.fill(text);
    await expect(page.getByTestId("composer-send")).toBeEnabled();

    await input.press("Enter");

    await expect(page.getByTestId("send-error")).toBeVisible();
    await expect(input).toHaveValue(text);
    await expect(page.getByTestId("composer-send")).toBeEnabled();
  });
});
