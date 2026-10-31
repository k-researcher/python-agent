import { expect, test } from "@playwright/test";
import type { MonitorRow } from "../src/api/types";
import { ensureRightPanelOpen, expectNoHorizontalOverflow, mockProjects, mockSessions, openApp, routeSnapshot } from "./helpers";

test("global activity shows steps, elapsed time, Stop and cross-project navigation", async ({ page }, info) => {
  await openApp(page, info);
  let hits = 0;
  let stopped = false;
  const row: MonitorRow = { session_id: "s1", parent_id: null, project_id: "p1", project: "Demo project",
    title: "Проверка проекта и длинного заголовка без обрезанных действий ".repeat(3),
    status: "running", model: "default", mode: "dev", action_kind: "tool", action_text: "read_file",
    duration_seconds: 10, idle_seconds: 1, cost: "0.03", has_unknown_cost: false };
  await page.route("**/api/monitor", route => {
    hits += 1;
    return route.fulfill({ json: [
      { ...row, status: stopped ? "stopped" : "running", duration_seconds: hits > 1 ? 15 : 10 },
      { ...row, session_id: "s2", project_id: "p2", project: "Второй проект", title: "Подготовка ответа",
        action_kind: "assistant", cost: "0", status: "awaiting_confirmation" },
    ] });
  });
  await page.route("**/api/projects", route => route.fulfill({ json: [
    ...mockProjects, { id: "p2", name: "Второй проект", root_path: "/tmp/other" },
  ] }));
  await routeSnapshot(page, "s2", "session", () => ({ ...mockSessions[1], project_id: "p2" }));
  const stops: string[] = [];
  await page.route("**/api/sessions/s1/stop", route => {
    stops.push(route.request().method());
    stopped = true;
    return route.fulfill({ json: { stopped: true } });
  });
  await ensureRightPanelOpen(page);
  expect(hits).toBe(0);
  const tab = page.locator('[data-tab="monitor"]');
  await tab.click();
  const tabGeometry = await page.locator('.right-panel__tabs').evaluate(element => ({
    width: element.clientWidth, content: element.scrollWidth,
  }));
  expect(tabGeometry.content).toBeLessThanOrEqual(tabGeometry.width + 1);
  const panelGeometry = await page.locator('[data-panel="monitor"]').evaluate(element => ({
    width: element.clientWidth, content: element.scrollWidth,
  }));
  expect(panelGeometry.content).toBeLessThanOrEqual(panelGeometry.width + 1);
  const first = page.locator('[data-monitor-id="s1"]');
  const second = page.locator('[data-monitor-id="s2"]');
  await expect(first).toContainText("Шаг: read_file");
  await expect(first).toContainText("Стоимость: $0.03");
  await expect(second).toContainText("Второй проект");
  await expect(second).toContainText("Шаг: отвечает");
  await expect(second.locator(".monitor-cost")).toHaveCount(0);
  await expect(first.locator(".monitor-time")).toHaveText("Время: 15 с", { timeout: 7000 });
  await expectNoHorizontalOverflow(page);
  const stop = first.getByRole("button", { name: /^Остановить:/ });
  await stop.focus();
  await expect(stop).toBeFocused();
  await info.attach("working-panel", {
    body: await page.screenshot({ path: info.outputPath("working-panel.png") }), contentType: "image/png",
  });
  await stop.press("Enter");
  await expect(first).toContainText("Остановлена");
  expect(stops).toEqual(["POST"]);
  await expect(page).not.toHaveURL(/\/s\/s1$/);
  await second.locator(".monitor-open").click();
  await expect(page).toHaveURL(/\/s\/s2$/);
});
