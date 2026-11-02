import { expect, test } from "@playwright/test";
import type { ModelsView } from "../src/api/modelSettings";
import { expectNoHorizontalOverflow, mockConfig, openApp } from "./helpers";

const FOUND = [
  { id: "nemotron-3.5-transcribe-30b-a4b-ea", context_window: 4096, kind: "llm", state: "not-loaded" },
  { id: "nvidia-nemotron-3.5-lightning-30b-a3b-mlx", context_window: 1048576, kind: "llm", state: "not-loaded" },
  { id: "qwen/qwen3.6-35b-a3b", context_window: 262144, kind: "llm", state: "not-loaded" },
];

function settings(): ModelsView {
  return {
    checksum: "version-0", deleted: { providers: [], models: [] }, warnings: [],
    providers: [{ id: "lmstudio", kind: "openai", base_url: "http://127.0.0.1:1234/v1",
      api_key_env: null, api_key_set: false, api_key_hint: "", configured: true,
      disabled: false, origin: "ui" }],
    models: [
      { id: "chat", provider: "lmstudio", model: "chat", context_window: 32768, max_tokens: 8192,
        pricing: null, disabled: false, origin: "ui" },
      { id: "old", provider: "lmstudio", model: "old", context_window: 32768, max_tokens: 8192,
        pricing: { input_per_mtok: 0, output_per_mtok: 0 }, disabled: true, origin: "ui" },
    ],
    routing: { default: "chat", fallback: [], roles: {}, light: null },
  };
}

test("found models keep their actions in one aligned group and delete stays visible", async ({ page }, info) => {
  await openApp(page, info);
  await page.route("**/api/config/public", route => route.fulfill({ json: { ...mockConfig, session_cost_limit: "1" } }));
  await page.route("**/api/settings/models**", async route => {
    const path = new URL(route.request().url()).pathname.replace("/api/settings/models", "");
    if (path === "/local-servers") return route.fulfill({ json: [] });
    if (path.endsWith("/test")) {
      return route.fulfill({ json: { ok: true, status: 200, models: FOUND.map(item => item.id), details: FOUND } });
    }
    return route.fulfill({ json: settings() });
  });
  await page.goto("/settings/models");
  await page.locator('[data-provider="lmstudio"]').getByTestId("provider-discover").click();
  const rows = page.locator("[data-found]");
  await expect(rows).toHaveCount(FOUND.length);

  // Every row keeps both buttons together; on a wide screen they end at one right edge.
  const groups = [];
  for (const id of FOUND.map(item => item.id)) {
    const row = page.locator(`[data-found="${id}"]`);
    const probe = await row.getByTestId("found-probe").boundingBox();
    const add = await row.getByTestId("found-add").boundingBox();
    expect(probe && add).toBeTruthy();
    expect(Math.abs(probe!.y - add!.y)).toBeLessThan(2);
    groups.push(add!.x + add!.width);
  }
  if (/^(desktop|laptop)/.test(info.project.name)) {
    expect(Math.max(...groups) - Math.min(...groups)).toBeLessThan(2);
  }

  // A switched off model keeps an active, red delete button.
  const remove = page.locator('[data-model="old"]').getByTestId("model-remove");
  await expect(remove).toBeEnabled();
  const look = await remove.evaluate(element => {
    const style = getComputedStyle(element);
    let opacity = 1;
    for (let node: Element | null = element; node; node = node.parentElement) opacity *= Number(getComputedStyle(node).opacity);
    return { opacity, color: style.color, keep: getComputedStyle(element.parentElement!).color };
  });
  expect(look.opacity).toBe(1);
  expect(look.color).not.toBe(look.keep);

  await expect(page.locator('[data-model="chat"]').getByTestId("model-price-warning")).toContainText("лимит стоимости $1");
  await expect(page.locator('[data-model="old"]').getByTestId("model-price")).toHaveText("Без оплаты");
  await expectNoHorizontalOverflow(page);
  const shots = process.env.UI_E2E_SHOTS;
  if (shots) {
    await page.screenshot({ path: `${shots}/${info.project.name}.png` });
    await page.locator("#models-title").locator("..").screenshot({ path: `${shots}/${info.project.name}-models.png` });
  }
});

test("restore buttons of deleted file settings stand in one column", async ({ page }, info) => {
  await openApp(page, info);
  const view = settings();
  view.deleted = { providers: ["letovo"], models: ["main", "light", "nemotron-3.5-transcribe-30b-a4b-ea"] };
  await page.route("**/api/settings/models**", async route => {
    const path = new URL(route.request().url()).pathname.replace("/api/settings/models", "");
    if (path === "/local-servers") return route.fulfill({ json: [] });
    return route.fulfill({ json: view });
  });
  await page.goto("/settings/models");
  const buttons = page.getByTestId("deleted-settings").getByRole("button", { name: "Восстановить" });
  await expect(buttons).toHaveCount(4);
  const lefts = [];
  for (const button of await buttons.all()) lefts.push((await button.boundingBox())!.x);
  expect(Math.max(...lefts) - Math.min(...lefts)).toBeLessThan(2);
  await expectNoHorizontalOverflow(page);
  const shots = process.env.UI_E2E_SHOTS;
  if (shots) await page.getByTestId("deleted-settings").screenshot({ path: `${shots}/${info.project.name}-deleted.png` });
});
