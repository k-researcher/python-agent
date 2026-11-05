import { expect, test } from "@playwright/test";
import { ensureLeftPanelOpen, expectNoHorizontalOverflow, openApp } from "./helpers";

const longName = "очень-длинное-имя-файла-с-заметками-по-квартире-и-договору-найма.md";

test("text files are attached, sent with the message and shown as chips", async ({ page }, testInfo) => {
  let body: { content: string; attachments: { name: string; text: string }[] } | null = null;
  await openApp(page, testInfo);
  await page.route("**/api/sessions/s1/messages", async (route) => {
    if (route.request().method() !== "POST") return route.fallback();
    body = route.request().postDataJSON();
    await route.fulfill({
      status: 201,
      json: {
        id: 900,
        session_id: "s1",
        role: "user",
        content: `Посмотри\n\n<attachment name="${longName}">\n# Заметки\n</attachment>\n\n<attachment name="run.py">\nprint(1)\n</attachment>`,
        tool_call_id: null,
        tool_calls: null,
        token_count: null,
        skipped: false,
        kind: "normal",
        reasoning_content: null,
        reasoning_effort: null,
        attachments: {
          text_end: 8,
          files: [
            { name: longName, size: 16, chars: 9 },
            { name: "run.py", size: 8, chars: 8 },
          ],
        },
        created_at: "2026-11-05T10:00:00Z",
      },
    });
  });
  await ensureLeftPanelOpen(page);
  await page.locator('[data-node-key="n:s1"]').click();
  await expect(page.getByTestId("composer-input")).toBeVisible();

  await page.getByTestId("composer-file-input").setInputFiles([
    { name: longName, mimeType: "text/markdown", buffer: Buffer.from("# Заметки") },
    { name: "run.py", mimeType: "text/x-python", buffer: Buffer.from("print(1)") },
  ]);
  const chips = page.getByTestId("composer-files").locator(".file");
  await expect(chips).toHaveCount(2);
  const composer = await page.getByTestId("composer").boundingBox();
  for (const chip of await chips.all()) {
    const box = await chip.boundingBox();
    expect(box && composer && box.x + box.width <= composer.x + composer.width + 1).toBe(true);
  }
  await expectNoHorizontalOverflow(page);

  await page.getByTestId("composer-input").fill("Посмотри");
  await page.getByTestId("composer-send").click();
  await expect(page.getByTestId("composer-files")).toHaveCount(0);
  expect(body).toEqual({
    content: "Посмотри",
    reasoning_effort: null,
    attachments: [
      { name: longName, text: "# Заметки" },
      { name: "run.py", text: "print(1)" },
    ],
  });

  const sent = page.getByTestId("message-files").last();
  await expect(sent.locator(".file")).toHaveCount(2);
  await expect(page.getByText("<attachment", { exact: false })).toHaveCount(0);
  await expectNoHorizontalOverflow(page);
});

test("refused files show an inline notice with the reason, inside the composer", async ({ page }, testInfo) => {
  await openApp(page, testInfo);
  await ensureLeftPanelOpen(page);
  await page.locator('[data-node-key="n:s1"]').click();
  await expect(page.getByTestId("composer-input")).toBeVisible();

  await page.getByTestId("composer-file-input").setInputFiles([
    { name: "photo.png", mimeType: "image/png", buffer: Buffer.from([0x89, 0x50, 0x00, 0x47]) },
    { name: longName, mimeType: "text/plain", buffer: Buffer.alloc(3 * 1024 * 1024, 0x61) },
    { name: "notes.md", mimeType: "text/markdown", buffer: Buffer.from("# ok") },
  ]);
  const notice = page.getByTestId("composer-file-error");
  await expect(notice).toBeVisible();
  await expect(notice.locator("li")).toHaveText([
    "photo.png: Это не текстовый файл. Сейчас можно приложить только текст: код, Markdown, логи, CSV.",
    `${longName}: Размер 3 МБ, а можно до 1 МБ. Разделите файл на части.`,
  ]);
  await expect(page.getByTestId("composer-files").locator(".file")).toHaveCount(1);
  const composer = await page.getByTestId("composer").boundingBox();
  const box = await notice.boundingBox();
  expect(box && composer && box.x >= composer.x && box.x + box.width <= composer.x + composer.width + 1).toBe(true);
  await expectNoHorizontalOverflow(page);
  await page.getByTestId("composer").screenshot({ path: testInfo.outputPath("notice.png") });

  await page.getByRole("button", { name: "Скрыть сообщение" }).click();
  await expect(notice).toHaveCount(0);
});
