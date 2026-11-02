import { expect, test } from "@playwright/test";
import { ensureLeftPanelOpen, expectNoHorizontalOverflow, openApp } from "./helpers";

// A fake microphone lets the browser record without a real device or a permission prompt.
test.use({
  launchOptions: { args: ["--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream"] },
  permissions: ["microphone"],
});

test("voice input puts recognized text into the composer for editing", async ({ page }, testInfo) => {
  await openApp(page, testInfo);
  await ensureLeftPanelOpen(page);
  await page.locator('[data-node-key="n:s1"]').click();
  const input = page.getByTestId("composer-input");
  await expect(input).toBeVisible();
  let sent = 0;
  page.on("request", (request) => {
    if (request.method() === "POST" && request.url().endsWith("/messages")) sent += 1;
  });

  await page.getByTestId("voice-start").click();
  await expect(page.getByTestId("voice-status")).toContainText("Запись");
  await expectNoHorizontalOverflow(page);
  await page.getByTestId("voice-finish").click();

  await expect(input).toHaveValue("распознанный текст");
  await expect(page.getByTestId("composer-send")).toBeEnabled();
  expect(sent).toBe(0);
});

test("settings open the voice screen from the models screen", async ({ page }, testInfo) => {
  await openApp(page, testInfo);
  await ensureLeftPanelOpen(page);
  await page.getByRole("button", { name: "Настройки", exact: true }).click();
  await page.getByTestId("settings-speech-link").click();
  await expect(page).toHaveURL(/\/settings\/speech$/);
  await expect(page.locator('[data-profile="local"]')).toContainText("Основной · Русский");
  await expectNoHorizontalOverflow(page);
});
