import { expect, test, type Page, type TestInfo } from "@playwright/test";
import {
  ensureLeftPanelOpen,
  expectNoHorizontalOverflow,
  installRoutes,
  installWsMock,
  mockProjects,
  mockSessions,
  panelStateForWidth,
  seedApp,
  themeOf,
  viewportWidthOf,
} from "./helpers";

const child = { ...mockSessions[1], id: "s3", parent_id: "s2", title: "Дочерняя сессия", status: "stopped" };
const archived = { ...mockSessions[1], id: "s4", title: "Старая сессия", archived: true };

interface Server {
  requests: string[];
  active: typeof mockSessions;
  archive: typeof mockSessions;
  projects: typeof mockProjects;
}

async function openSidebar(page: Page, info: TestInfo): Promise<Server> {
  const server: Server = {
    requests: [],
    active: [...mockSessions, child],
    archive: [archived],
    projects: [...mockProjects],
  };
  await seedApp(page, themeOf(info), panelStateForWidth(viewportWidthOf(info)));
  await installRoutes(page);
  await installWsMock(page);
  await page.route("**/api/projects**", async (route) => {
    const request = route.request();
    if (request.method() === "DELETE") {
      server.requests.push(`DELETE ${new URL(request.url()).pathname}${new URL(request.url()).search}`);
      server.projects = [];
      server.active = [];
      server.archive = [];
      return route.fulfill({ status: 204 });
    }
    return route.fulfill({ json: server.projects });
  });
  await page.route("**/api/sessions**", async (route) => {
    const request = route.request();
    const url = new URL(request.url());
    if (url.pathname === "/api/sessions" && request.method() === "GET") {
      return route.fulfill({ json: url.searchParams.get("archived") === "true" ? server.archive : server.active });
    }
    if (request.method() === "DELETE") {
      server.requests.push(`DELETE ${url.pathname}${url.search}`);
      server.active = server.active.filter((item) => item.id !== "s2" && item.id !== "s3");
      return route.fulfill({ status: 204 });
    }
    if (url.pathname.endsWith("/unarchive")) {
      server.requests.push(`POST ${url.pathname}`);
      server.active = [...server.active, { ...archived, archived: false }];
      server.archive = [];
      return route.fulfill({ json: { archived: false } });
    }
    return route.fallback();
  });
  await page.goto("/");
  await ensureLeftPanelOpen(page);
  return server;
}

function row(page: Page, title: string) {
  return page.locator(".tree-row").filter({ hasText: title });
}

test("a stopped session is deleted with its child after a confirmation", async ({ page }, info) => {
  const server = await openSidebar(page, info);
  await row(page, "Вторая сессия").locator("[data-session-menu]").click();
  const menu = page.getByRole("menu");
  await expect(menu).toBeVisible();
  const box = await menu.boundingBox();
  expect(box && box.x >= 0 && box.x + box.width <= viewportWidthOf(info)).toBeTruthy();
  await expectNoHorizontalOverflow(page);

  let question = "";
  page.once("dialog", (dialog) => {
    question = dialog.message();
    void dialog.accept();
  });
  await menu.locator("[data-session-delete]").click();
  await expect(row(page, "Вторая сессия")).toHaveCount(0);
  await expect(row(page, "Дочерняя сессия")).toHaveCount(0);
  expect(question).toContain("дочерние сессии: 1");
  expect(server.requests).toEqual(["DELETE /api/sessions/s2?with_children=true"]);
  await expect(row(page, "Первая сессия")).toHaveCount(1);
});

test("a cancelled confirmation keeps the session", async ({ page }, info) => {
  const server = await openSidebar(page, info);
  await row(page, "Вторая сессия").locator("[data-session-menu]").click();
  page.once("dialog", (dialog) => void dialog.dismiss());
  await page.getByRole("menu").locator("[data-session-delete]").click();
  await expect(row(page, "Вторая сессия")).toHaveCount(1);
  expect(server.requests).toEqual([]);
});

test("an archived session comes back from the archive group", async ({ page }, info) => {
  const server = await openSidebar(page, info);
  await row(page, "Архив").click();
  await row(page, "Старая сессия").locator("[data-session-menu]").click();
  await page.getByRole("menu").getByText("Вернуть из архива").click();
  await expect(row(page, "Архив")).toHaveCount(0);
  await expect(row(page, "Старая сессия")).toHaveCount(1);
  expect(server.requests).toEqual(["POST /api/sessions/s4/unarchive"]);
});

test("a project is deleted only after its name is typed", async ({ page }, info) => {
  const server = await openSidebar(page, info);
  await page.locator('[data-project-item="p1"] [data-project-menu]').click();
  await page.locator("[data-project-delete]").click();
  const form = page.locator("[data-delete-project-form]");
  await expect(form).toContainText("Файлы в папке проекта останутся");
  const submit = form.locator("[data-delete-project-submit]");
  await expect(submit).toBeDisabled();
  await form.locator("[data-delete-project-name]").fill("Demo project");
  await expectNoHorizontalOverflow(page);
  await submit.click();
  await expect(page.locator(".left-sidebar__hint", { hasText: "Нет проектов" })).toBeVisible();
  await expect(form).toHaveCount(0);
  expect(server.requests).toEqual(["DELETE /api/projects/p1?name=Demo%20project"]);
});
