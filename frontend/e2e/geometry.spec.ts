import { test, expect } from "@playwright/test";
import {
  openApp,
  assertThemeApplied,
  expectNoHorizontalOverflow,
  assertComposerGeometry,
  ensureRightPanelOpen,
  assertRightTabTextsDoNotOverlap,
  assertResponsiveDefaultPanels,
  themeOf,
  viewportWidthOf,
  ensureAgentControlsOpen,
  mockConfig,
} from "./helpers";

test.describe("layout geometry", () => {
  test("document has no horizontal overflow", async ({ page }, testInfo) => {
    await openApp(page, testInfo);
    await expect(page.getByTestId("composer-input")).toBeVisible();
    await expectNoHorizontalOverflow(page);
  });

  test("theme is applied deterministically", async ({ page }, testInfo) => {
    await openApp(page, testInfo);
    await expect(page.getByTestId("composer-input")).toBeVisible();
    await assertThemeApplied(page, themeOf(testInfo));
  });

  test("composer controls do not overlap and stay in their containers", async ({ page }, testInfo) => {
    await openApp(page, testInfo);
    const input = page.locator('[data-testid="composer-input"]');
    await expect(input).toBeVisible();
    await assertComposerGeometry(page);
  });

  test("settings open from the conversation header and leave the global header clear", async ({ page }, testInfo) => {
    await openApp(page, testInfo);
    await expect(page.locator('[data-testid="composer-input"]')).toBeVisible();
    await ensureAgentControlsOpen(page);
    const effort = page.locator('[data-testid="composer-effort"]');
    await expect(effort).toBeVisible();
    const geometry = await page.evaluate(() => {
      const rect = (sel: string) => {
        const el = document.querySelector(sel);
        if (!el) return null;
        const r = el.getBoundingClientRect();
        return { top: r.top, bottom: r.bottom };
      };
      const row = rect(".input-row");
      const effort = rect('[data-testid="composer-effort"]');
      return { rowTop: row?.top ?? 0, effortBottom: effort?.bottom ?? 0 };
    });
    expect(geometry.effortBottom).toBeLessThanOrEqual(geometry.rowTop + 1);
    await expect(page.locator(".home-header").getByTestId("composer-effort")).toHaveCount(1);
    await expect(page.locator(".topbar select")).toHaveCount(0);
  });

  test("agent controls share one toolbar and the composer stays at the bottom", async ({ page }, testInfo) => {
    await openApp(page, testInfo);
    await ensureAgentControlsOpen(page);
    const controls = page.getByTestId("agent-controls");
    await expect(controls.getByTestId("model-select")).toBeVisible();
    await expect(controls.getByTestId("mode-select")).toBeVisible();
    await expect(controls.getByTestId("composer-effort")).toBeVisible();
    await expect(page.locator(".home-header").getByTestId("model-select")).toHaveCount(1);
    await expect(page.getByTestId("composer").locator("select")).toHaveCount(0);
    await expect(page.getByTestId("model-select")).toHaveCount(1);
    const composer = await page.getByTestId("composer").boundingBox();
    const main = await page.locator(".app-shell__main").boundingBox();
    expect(composer).not.toBeNull();
    expect(main).not.toBeNull();
    expect(Math.abs(composer!.y + composer!.height - main!.y - main!.height)).toBeLessThanOrEqual(1);
    const bounds = await controls.boundingBox();
    expect(bounds).not.toBeNull();
    const fields = controls.locator("select");
    const rectangles = [];
    for (const field of await fields.all()) {
      const box = await field.boundingBox();
      expect(box).not.toBeNull();
      expect(box!.x).toBeGreaterThanOrEqual(bounds!.x - 1);
      expect(box!.x + box!.width).toBeLessThanOrEqual(bounds!.x + bounds!.width + 1);
      rectangles.push(box!);
    }
    for (const [index, current] of rectangles.entries()) {
      for (const other of rectangles.slice(index + 1)) {
        const overlap = current.x < other.x + other.width && other.x < current.x + current.width
          && current.y < other.y + other.height && other.y < current.y + current.height;
        expect(overlap).toBe(false);
      }
    }
    await page.getByTestId("composer-input").fill(Array.from({ length: 16 }, () => "Длинная строка для проверки ввода").join("\n"));
    await assertComposerGeometry(page);
    await expectNoHorizontalOverflow(page);
  });

  test("header choices update the request and unsupported reasoning resets", async ({ page }, testInfo) => {
    await openApp(page, testInfo);
    await ensureAgentControlsOpen(page);
    await page.getByTestId("composer-effort").selectOption("high");
    await page.getByTestId("model-select").selectOption("fast");
    await expect(page.getByTestId("composer-effort")).toHaveValue("auto");
    await page.getByTestId("composer-effort").selectOption("medium");
    await page.getByTestId("mode-select").selectOption("ask");
    await expect(page.getByTestId("agent-settings-button")).toHaveText("model-b · Консультант · Средний");
    await page.getByTestId("composer-input").fill("Проверка настройки агента");
    const request = page.waitForRequest((candidate) => candidate.url().endsWith("/api/sessions") && candidate.method() === "POST");
    await page.getByTestId("composer-send").click();
    expect((await request).postDataJSON()).toMatchObject({ llm_profile: "fast", mode: "ask", reasoning_effort: "medium" });
    await expect(page.getByTestId("home-error")).toHaveText("create failed");
    await ensureAgentControlsOpen(page);
    await expect(page.getByTestId("composer-effort")).toHaveValue("medium");
  });

  test("summary opens settings on every width and closes on Escape or outside interaction", async ({ page }, testInfo) => {
    await openApp(page, testInfo);
    const button = page.getByRole("button", { name: "Параметры агента", exact: true });
    await expect(button).toBeVisible();
    await expect(page.getByTestId("agent-controls")).toBeHidden();
    await button.click();
    await expect(button).toHaveAttribute("aria-expanded", "true");
    await page.getByTestId("mode-select").focus();
    await page.keyboard.press("Escape");
    await expect(button).toHaveAttribute("aria-expanded", "false");
    await expect(button).toBeFocused();
    await expect(page.getByTestId("agent-controls")).toBeHidden();
    await button.click();
    await page.getByTestId("composer-input").click();
    await expect(button).toHaveAttribute("aria-expanded", "false");
    await expect(page.getByTestId("composer-input")).toBeFocused();
  });

  test("long model names do not overflow the summary or settings panel", async ({ page }, testInfo) => {
    await openApp(page, testInfo);
    for (const modelName of ["Среднее-название-модели-24", "Очень-длинное-название-модели".repeat(8)]) {
    await page.route("**/api/config/public", async (route) => {
      const config = {
        ...mockConfig,
        llm_model: modelName,
        default_llm_profile: "long",
        llm_profiles: [{ name: "long", model: modelName, destination: "local", configured: true }],
      };
      await route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify(config) });
    });
    await page.reload();
    await expect(page.getByTestId("agent-settings-button")).toHaveAttribute("title", `${modelName} · Разработчик · Авто`);
    await ensureAgentControlsOpen(page);
    await expectNoHorizontalOverflow(page);
    const button = await page.getByTestId("agent-settings-button").boundingBox();
    const panel = await page.getByTestId("agent-settings-panel").boundingBox();
    expect(button).not.toBeNull();
    expect(panel).not.toBeNull();
    for (const bounds of [button!, panel!]) {
      expect(bounds.x).toBeGreaterThanOrEqual(0);
      expect(bounds.x + bounds.width).toBeLessThanOrEqual(viewportWidthOf(testInfo));
    }
    }
  });

  test("right panel tabs open and their text ranges do not overlap", async ({ page }, testInfo) => {
    await openApp(page, testInfo);
    await ensureRightPanelOpen(page);
    await assertRightTabTextsDoNotOverlap(page);
    const panel = await page.locator(".app-shell__right").boundingBox();
    expect(panel).not.toBeNull();
    const tabs = page.locator("[data-tab]");
    await expect(tabs).toHaveCount(5);
    for (const tab of await tabs.all()) {
      await tab.scrollIntoViewIfNeeded();
      const bounds = await tab.boundingBox();
      expect(bounds).not.toBeNull();
      expect(bounds!.x).toBeGreaterThanOrEqual(panel!.x - 1);
      expect(bounds!.x + bounds!.width).toBeLessThanOrEqual(panel!.x + panel!.width + 1);
    }
  });

  test("right panel tab panels switch content", async ({ page }, testInfo) => {
    await openApp(page, testInfo);
    await ensureRightPanelOpen(page);
    const tabs = page.locator("[data-tab]");
    await expect(tabs).toHaveCount(5);

    const ids = ["tools", "agents", "audit", "monitor", "context"];
    for (const id of ids) {
      await page.locator(`[data-tab="${id}"]`).click();
      await expect(page.locator(`[data-panel="${id}"]`)).toBeVisible();
    }
  });

  test("cold default state shows responsive panels without overflow", async ({ page }, testInfo) => {
    await openApp(page, testInfo, { cold: true });
    await expect(page.getByTestId("composer-input")).toBeVisible();
    await assertResponsiveDefaultPanels(page, viewportWidthOf(testInfo));
    await expectNoHorizontalOverflow(page);
  });
});
