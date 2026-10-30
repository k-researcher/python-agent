import { expect, test, type Page } from "@playwright/test";
import { ensureLeftPanelOpen, expectNoHorizontalOverflow, openApp, viewportWidthOf } from "./helpers";

async function assertComposerInsideMain(page: Page): Promise<void> {
  const composer = page.getByTestId("composer");
  const input = page.getByTestId("composer-input");
  const send = page.getByTestId("composer-send");
  await expect(composer).toBeVisible();
  await expect(input).toBeInViewport({ ratio: 1 });
  await expect(send).toBeInViewport({ ratio: 1 });
  const mainBounds = await page.locator(".app-shell__main").boundingBox();
  expect(mainBounds).not.toBeNull();
  for (const control of [composer, input, send]) {
    const bounds = await control.boundingBox();
    expect(bounds).not.toBeNull();
    expect(bounds!.y).toBeGreaterThanOrEqual(mainBounds!.y - 1);
    expect(bounds!.y + bounds!.height).toBeLessThanOrEqual(mainBounds!.y + mainBounds!.height + 1);
  }
}

for (const view of [{ name: "home", path: "/" }, { name: "session", path: "/s/s1" }]) {
  for (const shortViewport of [false, true]) {
    test(`${view.name}: composer fits with a long draft${shortViewport ? " in short viewport" : ""}`, async ({ page }, testInfo) => {
      if (shortViewport) {
        await page.setViewportSize({ width: viewportWidthOf(testInfo), height: 480 });
      }
      await openApp(page, testInfo);
      await ensureLeftPanelOpen(page);
      if (view.name === "session") {
        await page.locator('[data-node-key="n:s1"]').click();
      } else {
        await page.getByRole("button", { name: "Новая сессия", exact: true }).click();
      }
      await expect(page).toHaveURL(new URL(view.path, page.url()).href);
      await expect(page.getByTestId("composer-input")).toBeVisible();
      if (viewportWidthOf(testInfo) <= 760) {
        await page.locator('[data-toggle="left"]').click();
      }
      await assertComposerInsideMain(page);
      await page.getByTestId("composer-input").fill(
        Array.from({ length: 16 }, () => "Длинная строка для проверки ввода").join("\n"),
      );
      await assertComposerInsideMain(page);
    });
  }
}

test("settings open the model screen", async ({ page }, testInfo) => {
  await openApp(page, testInfo);
  await ensureLeftPanelOpen(page);
  await page.getByRole("button", { name: "Настройки", exact: true }).click();
  await expect(page).toHaveURL(/\/settings\/models$/);
  await expect(page.getByTestId("models-settings")).toBeVisible();
  await expect(page.locator('[data-model="model-a"]')).toBeVisible();
  await expectNoHorizontalOverflow(page);
});
