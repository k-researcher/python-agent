import { expect, test } from "@playwright/test";
import { openApp, assertThemeApplied, themeOf } from "./helpers";

test.describe("theme", () => {
  test("explicit light overrides a dark system preference", async ({ page }, info) => {
    await page.emulateMedia({ colorScheme: "dark" });
    await page.addInitScript(() => localStorage.setItem("agent:theme", "light"));
    await openApp(page, info, { cold: true });
    await expect(page.getByTestId("composer-input")).toBeVisible();
    await assertThemeApplied(page, "light");
  });

  test("theme toggle persists on reload", async ({ page }, info) => {
    await openApp(page, info, { cold: true });
    await expect(page.getByTestId("composer-input")).toBeVisible();
    const target = themeOf(info) === "light" ? "dark" : "light";
    await page.locator("[data-theme-toggle]").click();
    await assertThemeApplied(page, target);
    await expect.poll(() => page.evaluate(() => localStorage.getItem("agent:theme")))
      .toBe(target);
    await page.reload();
    await expect(page.getByTestId("composer-input")).toBeVisible();
    await assertThemeApplied(page, target);
  });

  test("send button uses contrasting theme tokens", async ({ page }, info) => {
    await openApp(page, info, { cold: true });
    await page.getByTestId("composer-input").fill("проверка кнопки");
    const send = page.getByTestId("composer-send");
    await expect(send).toBeEnabled();
    const colors = await send.evaluate((button) => {
      const style = getComputedStyle(button);
      const channels = (color: string) => color.match(/[\d.]+/g)!.slice(0, 3).map(Number);
      const luminance = (color: string) => channels(color).map((value) => {
        const channel = value / 255;
        return channel <= 0.04045 ? channel / 12.92 : ((channel + 0.055) / 1.055) ** 2.4;
      }).reduce((total, value, index) => total + value * [0.2126, 0.7152, 0.0722][index], 0);
      const foreground = luminance(style.color);
      const background = luminance(style.backgroundColor);
      return (Math.max(foreground, background) + 0.05) /
        (Math.min(foreground, background) + 0.05);
    });
    expect(colors).toBeGreaterThanOrEqual(4.5);
  });
});
