import { test, expect } from "@playwright/test";
import {
  openApp,
  assertComposerGeometry,
  ensureLeftPanelOpen,
  viewportWidthOf,
} from "./helpers";

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

test.describe("session view", () => {
  test("opens an existing session and shows its composer", async ({ page }, testInfo) => {
    await openSessionViaSidebar(page, testInfo, "s1");
    await expect(page.getByTestId("session-title")).toHaveText("Первая сессия");
    await expect(page.getByTestId("composer-input")).toBeVisible();
    if (viewportWidthOf(testInfo) <= 760) {
      await expect(page.locator(".app-shell__left")).toBeHidden();
    }
    await expect(page.getByTestId("composer-input")).toBeVisible();
    await assertComposerGeometry(page);
  });

  test("session shows its fixed role beside reasoning and status in the header", async ({ page }, testInfo) => {
    await openSessionViaSidebar(page, testInfo, "s1");
    await assertComposerGeometry(page);
    await expect(page.getByTestId("agent-controls").getByTestId("session-mode")).toHaveText("Разработчик");
    await expect(page.getByTestId("agent-controls").getByTestId("session-profile")).toHaveText("default · model-a");
    await expect(page.getByTestId("agent-controls").locator("select")).toHaveCount(1);
    await expect(page.getByTestId("session-status")).toHaveText("Готова");
  });

  test("composer restores the draft after a failed send", async ({ page }, testInfo) => {
    await openSessionViaSidebar(page, testInfo, "s1");
    const input = page.getByTestId("composer-input");
    await expect(input).toBeVisible();
    const text = "не сохранившееся сообщение";
    await input.fill(text);
    await expect(page.getByTestId("composer-send")).toBeEnabled();
    await page.getByTestId("composer-send").click();

    await expect(page.getByTestId("send-error")).toBeVisible();
    await expect(input).toHaveValue(text);
    await expect(page.getByTestId("composer-send")).toBeEnabled();
  });

  test("home composer restores the draft after a failed session creation", async ({ page }, testInfo) => {
    await openApp(page, testInfo);
    const input = page.getByTestId("composer-input");
    await expect(input).toBeVisible();
    const text = "проект не создастся";
    await input.fill(text);
    await expect(page.getByTestId("composer-send")).toBeEnabled();
    await page.getByTestId("composer-send").click();

    await expect(page.getByTestId("home-error")).toBeVisible();
    await expect(input).toHaveValue(text);
  });

  test("settings and message input have predictable keyboard navigation", async ({ page }, testInfo) => {
    await openSessionViaSidebar(page, testInfo, "s1");
    const input = page.getByTestId("composer-input");
    const send = page.getByTestId("composer-send");
    const effort = page.getByTestId("composer-effort");
    const button = page.getByTestId("agent-settings-button");
    await button.focus();
    await page.keyboard.press("Enter");
    await expect(effort).toBeFocused();
    await page.keyboard.press("Escape");
    await expect(button).toBeFocused();
    await button.press("Enter");
    await expect(effort).toBeFocused();
    await page.keyboard.press("Tab");
    await expect(effort).toBeHidden();
    await expect(input).toBeVisible();
    await input.fill("достаточно текста для включения кнопки");
    await input.click();
    await expect(input).toBeFocused();

    await page.keyboard.press("Tab");
    await expect(send).toBeFocused();
  });

  test("left sidebar tree navigates and selects a session by keyboard", async ({ page }, testInfo) => {
    await openApp(page, testInfo);
    await ensureLeftPanelOpen(page);
    const tree = page.locator("[role=tree]");
    await expect(tree).toBeVisible();
    const nodes = tree.locator("[data-node-key]");
    await expect(nodes).toHaveCount(2);
    await nodes.first().focus();
    await expect(nodes.nth(0)).toBeFocused();

    await page.keyboard.press("ArrowDown");
    await expect(nodes.nth(1)).toBeFocused();
    await page.keyboard.press("Enter");
    await expect(page).toHaveURL(/\/s\/s2/);
    await expect(page.getByTestId("session-title")).toHaveText("Вторая сессия");
  });
});
