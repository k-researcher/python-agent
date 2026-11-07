import { expect, test, type Page } from "@playwright/test";
import type { McpList, McpServerItem } from "../src/api/mcp";
import { expectNoHorizontalOverflow, openApp } from "./helpers";

// The MCP page: cards with long commands fit, the editor fits a phone, and removal asks first.

const server = (id: string, extra: Partial<McpServerItem> = {}): McpServerItem => ({
  id,
  command: ["npx", "-y", `@example/${id}-server-with-a-long-package-name`, "--stdio"],
  url: null,
  header_keys: [],
  http_protocol: "auto",
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
      list = { ...list, servers: [...list.servers, server(body.id, {
        command: body.command ?? [], env_keys: Object.keys(body.env ?? {}), url: body.url ?? null,
        header_keys: Object.keys(body.headers ?? {}), http_protocol: body.http_protocol ?? "auto",
      })] };
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

test("an HTTP server can be added without command fields and shows its checked protocol", async ({ page }, info) => {
  await openApp(page, info);
  const sent = await openMcp(page, { sandbox: false, file_sources: [], servers: [] });
  await page.getByTestId("mcp-add").click();
  await page.getByTestId("mcp-id").fill("remote");
  await page.getByRole("radio", { name: "Адрес", exact: true }).click();
  await expect(page.getByRole("radio", { name: "Адрес", exact: true })).toHaveAttribute("aria-checked", "true");
  await expect(page.getByTestId("mcp-no-sandbox")).toHaveCount(0);
  for (const id of ["mcp-command", "mcp-network", "mcp-paths"]) await expect(page.getByTestId(id)).toHaveCount(0);
  const address = "https://example.com/mcp/endpoint-with-a-long-path-for-documentation-search";
  await page.getByTestId("mcp-url").fill(address);
  const mono = await page.evaluate(() => getComputedStyle(document.documentElement).getPropertyValue("--font-mono").trim());
  expect(await page.getByTestId("mcp-url").evaluate(element => getComputedStyle(element).fontFamily)).toBe(mono);
  await page.getByTestId("mcp-env-add").click();
  await page.getByTestId("mcp-env-key").fill("Authorization");
  await expect(page.getByTestId("mcp-save")).toBeDisabled();
  await page.getByTestId("mcp-env-value").fill("Bearer hidden-value");
  await expect(page.getByTestId("mcp-env-value")).toHaveAttribute("type", "password");
  for (const name of ["Авто", "Новая (2026-07-28)", "С рукопожатием"]) {
    const option = page.getByRole("radio", { name, exact: true });
    await option.focus();
    await page.keyboard.press("Space");
    await expect(option).toHaveAttribute("aria-checked", "true");
  }
  const options = page.locator('[aria-label="Протокол HTTP"] button');
  for (const option of await options.all()) await expect(option).toBeInViewport({ ratio: 1 });
  const shots = process.env.UI_E2E_SHOTS;
  if (shots) await page.screenshot({ path: `${shots}/${info.project.name}-mcp-http-editor.png`, animations: "disabled" });
  await expectNoHorizontalOverflow(page);
  await page.getByTestId("mcp-save").click();
  await expect(page.getByRole("dialog")).toHaveCount(0);
  const card = page.locator('[data-server="remote"]');
  await expect(card).toContainText(address);
  await expect(card.locator(".ui-badge")).toHaveText("HTTP");
  await expect(card).not.toContainText("без сети");
  expect(sent).toHaveLength(1);
  expect(JSON.parse(sent[0].split(" ").slice(2).join(" "))).toEqual({
    id: "remote", url: address, headers: { Authorization: "Bearer hidden-value" },
    http_protocol: "legacy", project_ids: [], allowed_modes: ["dev"],
  });
  await expect(page.locator("body")).not.toContainText("hidden-value");

  await page.route("**/api/mcp/servers/remote/check", route => route.fulfill({ json: {
    sandbox: false, file_sources: [], servers: [server("remote", {
      command: [], url: address, header_keys: ["Authorization"], http_protocol: "legacy",
      last_check: { ok: true, protocol: "2025-06-18", tool_count: 1, tools: [{ name: "search" }] },
    })],
  } }));
  await card.getByTestId("mcp-check-run").click();
  await expect(card.getByTestId("mcp-check")).toContainText("2025-06-18");
  await expect(card.getByTestId("mcp-details")).toContainText("search");
  await expectNoHorizontalOverflow(page);
  if (shots) await page.screenshot({ path: `${shots}/${info.project.name}-mcp-http-card.png`, fullPage: true, animations: "disabled" });
});

test("HTTP editing preserves hidden headers and a failed save stays open", async ({ page }, info) => {
  await openApp(page, info);
  const sent = await openMcp(page, { sandbox: false, file_sources: [], servers: [server("remote", {
    command: [], url: "https://example.com/mcp", header_keys: ["Authorization", "X-Drop"], http_protocol: "modern",
  })] });
  await page.locator('[data-server="remote"]').getByTestId("mcp-edit").click();
  await expect(page.getByTestId("mcp-url")).toBeFocused();
  await expect(page.getByRole("radio", { name: "Команда", exact: true })).toBeDisabled();
  await expect(page.getByRole("radio", { name: "Новая (2026-07-28)", exact: true })).toHaveAttribute("aria-checked", "true");
  for (const input of await page.getByTestId("mcp-env-value").all()) await expect(input).toHaveValue("");
  await page.getByRole("button", { name: "Убрать заголовок X-Drop", exact: true }).click();
  await page.route("**/api/mcp/servers/remote", route => route.fulfill({ status: 422, json: { detail: "Адрес недоступен" } }));
  const request = page.waitForRequest(request => request.method() === "PUT");
  await page.getByTestId("mcp-save").click();
  await expect(page.getByTestId("mcp-form-error")).toContainText("Адрес недоступен");
  await expect(page.getByRole("dialog")).toBeVisible();
  expect((await request).postDataJSON()).toEqual({ url: "https://example.com/mcp", headers: { "X-Drop": null },
    http_protocol: "modern", project_ids: [], allowed_modes: ["dev"] });
  await page.keyboard.press("Escape");
  await expect(page.getByRole("dialog")).toHaveCount(0);
  expect(sent).toEqual([]);
});
