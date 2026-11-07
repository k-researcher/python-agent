import { expect, test, type Page } from "@playwright/test";
import type { McpList, McpServerItem } from "../src/api/mcp";
import { expectNoHorizontalOverflow, openApp } from "./helpers";

// The MCP page: cards with long commands fit, the editor fits a phone, and removal asks first.

const server = (id: string, extra: Partial<McpServerItem> = {}): McpServerItem => ({
  id,
  command: ["npx", "-y", `@example/${id}-server-with-a-long-package-name`, "--stdio"],
  env_keys: [],
  project_ids: [],
  allowed_modes: ["dev"],
  allow_network: true,
  read_paths: ["/opt/homebrew"],
  enabled: true,
  last_check: null,
  updated_at: null,
  ...extra,
});

function initial(): McpList {
  return {
    sandbox: true,
    file_sources: ["files"],
    servers: [
      server("docs", {
        env_keys: ["DOCS_TOKEN", "DOCS_ORGANIZATION_ID"],
        allowed_modes: ["dev", "ask"],
        last_check: {
          ok: true,
          server: { name: "docs", version: "1.4.0" },
          tools: [
            { name: "search_documentation", description: "Ищет по документации и возвращает подходящие разделы с адресами." },
            { name: "read_page" },
          ],
          tool_count: 2,
        },
      }),
      server("broken", {
        command: ["/Applications/Some Tools.app/Contents/MacOS/broken-server"],
        allow_network: false,
        enabled: false,
        last_check: { ok: false, error: "Сервер завершился до ответа", stderr: "Error: Cannot find module 'x'\n    at load (node:internal)" },
      }),
    ],
  };
}

async function openMcp(page: Page, list: McpList = initial()): Promise<string[]> {
  const sent: string[] = [];
  await page.route("**/api/mcp/servers**", async (route) => {
    const request = route.request();
    if (request.method() !== "GET") sent.push(`${request.method()} ${new URL(request.url()).pathname} ${request.postData() ?? ""}`);
    if (request.method() === "POST" && new URL(request.url()).pathname === "/api/mcp/servers") {
      const body = JSON.parse(request.postData() ?? "{}");
      list = { ...list, servers: [...list.servers, server(body.id, { command: body.command, env_keys: Object.keys(body.env) })] };
      return route.fulfill({ status: 201, json: list });
    }
    if (request.method() === "DELETE") {
      const id = decodeURIComponent(new URL(request.url()).pathname.split("/").pop() ?? "");
      list = { ...list, servers: list.servers.filter((item) => item.id !== id) };
    }
    return route.fulfill({ json: list });
  });
  await page.goto("/settings/mcp");
  await expect(page.getByTestId("mcp-settings")).toBeVisible();
  return sent;
}

test("MCP page fits the screen with long commands and opened details", async ({ page }, info) => {
  await openApp(page, info);
  await openMcp(page);
  await expect(page.locator("[data-server]")).toHaveCount(2);
  await expect(page.getByTestId("settings-mcp-link")).toHaveAttribute("aria-current", "page");
  await page.locator('[data-server="docs"] [data-testid="mcp-check"]').click();
  await page.locator('[data-server="broken"] [data-testid="mcp-check"]').click();
  await expect(page.getByTestId("mcp-details")).toHaveCount(2);
  await expectNoHorizontalOverflow(page);
  // Every card action stays inside its card.
  for (const id of ["docs", "broken"]) {
    const card = (await page.locator(`[data-server="${id}"]`).boundingBox())!;
    for (const action of await page.locator(`[data-server="${id}"] .ui-card__actions > *`).all()) {
      const box = (await action.boundingBox())!;
      expect(box.x).toBeGreaterThanOrEqual(card.x);
      expect(box.x + box.width).toBeLessThanOrEqual(card.x + card.width + 0.5);
    }
  }
  const shots = process.env.UI_E2E_SHOTS;
  if (shots) await page.screenshot({ path: `${shots}/${info.project.name}-mcp.png`, fullPage: true, animations: "disabled" });
});

test("IDs and commands keep the fixed-width font in a field", async ({ page }, info) => {
  await openApp(page, info);
  await openMcp(page);
  await page.locator('[data-server="docs"] [data-testid="mcp-edit"]').click();
  const command = page.getByTestId("mcp-command");
  const mono = await page.evaluate(() => getComputedStyle(document.documentElement).getPropertyValue("--font-mono").trim());
  expect(await command.evaluate((element) => getComputedStyle(element).fontFamily)).toBe(mono);
});

test("the settings tabs stay on one screen with the MCP tab", async ({ page }, info) => {
  await openApp(page, info);
  await openMcp(page);
  for (const name of ["models", "speech", "skills", "mcp"]) {
    await expect(page.getByTestId(`settings-${name}-link`)).toBeInViewport({ ratio: 1 });
  }
  await page.getByTestId("settings-skills-link").click();
  await expect(page).toHaveURL(/\/settings\/skills$/);
});

test("a new server is added from the editor", async ({ page }, info) => {
  await openApp(page, info);
  const sent = await openMcp(page, { sandbox: true, file_sources: [], servers: [] });
  await expect(page.getByTestId("mcp-empty")).toBeVisible();
  await page.getByTestId("mcp-add").click();
  const dialog = page.getByRole("dialog", { name: "Новый MCP-сервер" });
  await expect(dialog).toBeVisible();
  await expect(page.getByTestId("mcp-id")).toBeFocused();
  await page.getByTestId("mcp-id").fill("search");
  await page.getByTestId("mcp-command").fill(`node "/Users/someone/My Tools/search.js"`);
  await page.getByTestId("mcp-env-add").click();
  await page.getByTestId("mcp-env-key").fill("SEARCH_TOKEN");
  await page.getByTestId("mcp-env-value").fill("hidden-value");
  await expectNoHorizontalOverflow(page);
  const shots = process.env.UI_E2E_SHOTS;
  if (shots) await page.screenshot({ path: `${shots}/${info.project.name}-mcp-editor.png`, animations: "disabled" });
  await page.getByTestId("mcp-save").click();
  await expect(dialog).toBeHidden();
  await expect(page.locator('[data-server="search"]')).toContainText("Секреты: SEARCH_TOKEN");
  await expect(page.getByTestId("mcp-status")).toContainText("Сервер «search» добавлен");
  expect(sent).toHaveLength(1);
  const body = JSON.parse(sent[0].split(" ").slice(2).join(" "));
  expect(body).toMatchObject({ id: "search", command: ["node", "/Users/someone/My Tools/search.js"], env: { SEARCH_TOKEN: "hidden-value" } });
  expect(body).not.toHaveProperty("read_paths");
  await expect(page.locator("body")).not.toContainText("hidden-value");
});

test("the editor closes with Escape and sends nothing", async ({ page }, info) => {
  await openApp(page, info);
  const sent = await openMcp(page);
  await page.locator('[data-server="docs"] [data-testid="mcp-edit"]').click();
  const dialog = page.getByRole("dialog", { name: "Сервер docs" });
  await expect(dialog).toBeVisible();
  await expect(page.getByTestId("mcp-command")).toBeFocused();
  await page.keyboard.press("Escape");
  await expect(dialog).toBeHidden();
  expect(sent).toEqual([]);
});

test("delete asks first with the safe choice focused", async ({ page }, info) => {
  await openApp(page, info);
  const sent = await openMcp(page);
  await page.locator('[data-server="docs"] [data-testid="mcp-remove"]').click();
  const alert = page.getByRole("alertdialog", { name: "Удалить сервер «docs»?" });
  await expect(alert).toBeVisible();
  await expect(alert.getByRole("button", { name: "Отмена" })).toBeFocused();
  const shots = process.env.UI_E2E_SHOTS;
  if (shots) await page.screenshot({ path: `${shots}/${info.project.name}-mcp-delete.png`, animations: "disabled" });
  await alert.getByTestId("mcp-remove-confirm").click();
  await expect(alert).toBeHidden();
  await expect(page.locator("[data-server]")).toHaveCount(1);
  expect(sent).toEqual(["DELETE /api/mcp/servers/docs "]);
});
