import { expect, test } from "@playwright/test";
import { expectNoHorizontalOverflow, mockSessions, openApp, routeSnapshot } from "./helpers";

test("child approval is visible in the parent and supports a round trip", async ({ page }, info) => {
  await openApp(page, info);
  const child = { ...mockSessions[1], parent_id: "s1", status: "awaiting_confirmation",
    title: "Первый пост о Python, Rust и машинном обучении ".repeat(4) };
  await routeSnapshot(page, "s1", "session", () => ({ ...mockSessions[0], status: "running" }));
  await routeSnapshot(page, "s1", "children", () => [child]);
  await routeSnapshot(page, "s2", "session", () => child);
  await page.goto("/s/s1");
  const banner = page.getByTestId("child-waiting");
  await expect(banner).toContainText("ждут вашего подтверждения");
  const link = banner.getByRole("button");
  await link.focus();
  await expect(link).toBeFocused();
  await expectNoHorizontalOverflow(page);
  const input = await page.getByTestId("composer-input").boundingBox();
  const warning = await banner.boundingBox();
  expect(input).not.toBeNull();
  expect(warning).not.toBeNull();
  expect(warning!.y + warning!.height).toBeLessThanOrEqual(input!.y);
  expect(input!.y + input!.height).toBeLessThanOrEqual(page.viewportSize()!.height);
  await info.attach("child-approval", {
    body: await page.screenshot({ path: `/private/tmp/python-agent-child-approval-${info.project.name}.png` }),
    contentType: "image/png",
  });
  await link.press("Enter");
  await expect(page).toHaveURL(/\/s\/s2$/);
  const back = page.getByTestId("parent-session");
  await expect(back).toHaveText("К родительской сессии");
  await back.focus();
  await back.press("Enter");
  await expect(page).toHaveURL(/\/s\/s1$/);
  await expect(banner).toBeVisible();
});
