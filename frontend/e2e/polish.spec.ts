import { test, expect } from "@playwright/test";
import {
  openApp,
  ensureLeftPanelOpen,
  ensureRightPanelOpen,
  expectNoHorizontalOverflow,
  mockMessages,
  mockSessions,
  mockContext,
  routeSnapshot,
} from "./helpers";

test.describe("geometry polish", () => {
  for (const [stage, label] of [
    ["stage_1", "Сокращены результаты инструментов"],
    ["stage_2", "Свёрнуты вызовы инструментов"],
    ["summary", "Старые сообщения заменены сводкой"],
    ["summarize", "Старые сообщения заменены сводкой"],
    ["deterministic", "История сокращена без модели"],
  ]) {
    test(`compression metadata keeps its complete label and value (${stage})`, async ({ page }, testInfo) => {
      await openApp(page, testInfo);
      await routeSnapshot(page, "s1", "context", () => ({ ...mockContext, stage }));
      await page.goto("/s/s1");
      await ensureRightPanelOpen(page);
      const row = page.locator("[data-context-stage]").locator("..");
      await expect(row.locator("dt")).toHaveText("Сжатие контекста");
      await expect(row.locator("dd")).toHaveText(label);
      await expect.poll(() => row.evaluate((element) => {
        const term = element.querySelector("dt");
        const value = element.querySelector("dd");
        if (!term || !value) throw new Error("Compression metadata is missing");
        const bounds = element.getBoundingClientRect();
        const termBounds = term.getBoundingClientRect();
        const valueBounds = value.getBoundingClientRect();
        const termText = document.createRange();
        termText.selectNodeContents(term);
        const valueText = document.createRange();
        valueText.selectNodeContents(value);
        const valueStyle = getComputedStyle(value);
        const below = valueBounds.top >= termBounds.bottom - 1;
        return {
          labelLines: termText.getClientRects().length,
          labelNowrap: getComputedStyle(term).whiteSpace === "nowrap",
          pairWraps: getComputedStyle(element).flexWrap === "wrap",
          valueWraps: valueStyle.whiteSpace === "normal",
          noEllipsis: valueStyle.textOverflow !== "ellipsis",
          noClipping: valueStyle.overflowX === "visible" && valueStyle.overflowY === "visible",
          labelFits: termBounds.left >= bounds.left - 1 && termBounds.right <= bounds.right + 1,
          valueFits: valueBounds.left >= bounds.left - 1 && valueBounds.right <= bounds.right + 1
            && value.scrollWidth <= value.clientWidth + 1 && value.scrollHeight <= value.clientHeight + 1,
          textFits: Array.from(valueText.getClientRects()).every((rect) =>
            rect.left >= valueBounds.left - 1 && rect.right <= valueBounds.right + 1
            && rect.top >= valueBounds.top - 1 && rect.bottom <= valueBounds.bottom + 1),
          pairAligned: below ? Math.abs(valueBounds.left - bounds.left) <= 1
            : valueBounds.left >= termBounds.right + 9,
        };
      })).toEqual({
        labelLines: 1,
        labelNowrap: true,
        pairWraps: true,
        valueWraps: true,
        noEllipsis: true,
        noClipping: true,
        labelFits: true,
        valueFits: true,
        textFits: true,
        pairAligned: true,
      });
      const otherRows = page.locator(".context-block__meta > div:not(.context-block__compression)");
      await expect(otherRows).toHaveCount(3);
      for (const otherRow of await otherRows.all()) {
        await expect(otherRow).toHaveCSS("flex-wrap", "nowrap");
        await expect(otherRow.locator("dd")).toHaveCSS("white-space", "nowrap");
        await expect(otherRow.locator("dd")).toHaveCSS("text-overflow", "ellipsis");
      }
      await expectNoHorizontalOverflow(page);
      if (stage === "stage_1") {
        await testInfo.attach("compression-layout", {
          body: await page.screenshot(),
          contentType: "image/png",
        });
      }
    });
  }

  test("sidebar headings keep sentence case and context compression is clear", async ({ page }, testInfo) => {
    await openApp(page, testInfo);
    await ensureLeftPanelOpen(page);
    const headings = page.locator(".sidebar-heading__title");
    await expect(headings).toHaveText(["Проекты", "Сессии"]);
    await expect(headings.nth(0)).toHaveCSS("text-transform", "none");
    await expect(headings.nth(1)).toHaveCSS("text-transform", "none");
    await page.goto("/s/s1");
    await ensureRightPanelOpen(page);
    await expect(page.getByText("Сжатие контекста", { exact: true })).toBeVisible();
    await expect(page.locator("[data-context-stage]")).toHaveText("Сокращены результаты инструментов");
  });

  test("long session title keeps its tooltip and stays within the heading", async ({ page }, testInfo) => {
    const longTitle = "Очень длинный заголовок существующей сессии ".repeat(20);
    await openApp(page, testInfo);
    await routeSnapshot(page, "s1", "session", () => ({ ...mockSessions[0], title: longTitle }));
    await ensureLeftPanelOpen(page);
    await page.locator('[data-node-key="n:s1"]').click();
    const title = page.getByTestId("session-title");
    await expect(title).toBeVisible();
    await expect(title).toHaveAttribute("title", longTitle);
    const geometry = await title.evaluate((element) => {
      const heading = element.closest(".session-heading");
      if (!heading) throw new Error("Session heading is missing");
      const bounds = element.getBoundingClientRect();
      const container = heading.getBoundingClientRect();
      const style = getComputedStyle(element);
      return {
        width: bounds.width,
        left: bounds.left,
        right: bounds.right,
        containerLeft: container.left,
        containerRight: container.right,
        clientWidth: element.clientWidth,
        scrollWidth: element.scrollWidth,
        textOverflow: style.textOverflow,
        whiteSpace: style.whiteSpace,
      };
    });
    expect(geometry.width).toBeGreaterThan(0);
    expect(geometry.left).toBeGreaterThanOrEqual(geometry.containerLeft - 1);
    expect(geometry.right).toBeLessThanOrEqual(geometry.containerRight + 1);
    expect(geometry.scrollWidth).toBeGreaterThan(geometry.clientWidth);
    expect(geometry.textOverflow).toBe("ellipsis");
    expect(geometry.whiteSpace).toBe("nowrap");
    await expectNoHorizontalOverflow(page);
    await testInfo.attach("long-title", {
      body: await page.screenshot(),
      contentType: "image/png",
    });
  });

  test("long code wraps inside the block with its toolbar inside the corner", async ({ page }, testInfo) => {
    const longCode = "const value = '" + "a".repeat(1000) + "';";
    await openApp(page, testInfo);
    await routeSnapshot(page, "s1", "messages", () => ([{
        ...mockMessages[0],
        id: 30,
        role: "assistant",
        kind: "normal",
        content: "```text\n" + longCode + "\n```",
      }]));
    await page.goto("/s/s1");
    const block = page.locator(".code-block").first();
    const pre = block.locator("pre");
    await expect(pre).toBeVisible();
    await expect(pre.locator("code")).toHaveText(longCode);
    await block.hover();
    await expect(block.locator(".copy-btn")).toBeVisible();
    const geometry = () => pre.evaluate((element) => {
      const button = element.parentElement?.querySelector(".copy-btn");
      if (!button) return null;
      const box = element.getBoundingClientRect();
      const btn = button.getBoundingClientRect();
      return {
        overflowing: element.scrollWidth > element.clientWidth + 1,
        buttonInside: btn.top >= box.top && btn.right <= box.right + 1 && btn.bottom <= box.bottom,
        whiteSpace: getComputedStyle(element).whiteSpace,
      };
    });
    await expect.poll(geometry).toEqual({ overflowing: false, buttonInside: true, whiteSpace: "pre-wrap" });
    await block.locator(".wrap-btn").click();
    await expect.poll(geometry).toEqual({ overflowing: true, buttonInside: true, whiteSpace: "pre" });
    await expectNoHorizontalOverflow(page);
    await testInfo.attach("long-code", {
      body: await page.screenshot(),
      contentType: "image/png",
    });
  });
});
