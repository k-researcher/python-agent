import { expect, test, type Locator, type Page } from "@playwright/test";
import type { ModelsView } from "../src/api/modelSettings";
import { expectNoHorizontalOverflow, openApp } from "./helpers";

// Buttons of the same kind stand in one column in every card of a list.

function models(withReset: boolean): ModelsView {
  return {
    checksum: "version-0", deleted: { providers: [], models: [] }, warnings: [],
    providers: [
      { id: "lmstudio", kind: "openai", base_url: "http://127.0.0.1:1234/v1", api_key_env: null,
        api_key_set: false, api_key_hint: "", configured: true, disabled: false, origin: "ui" },
      { id: "gateway-with-a-long-name", kind: "openai", base_url: "https://llm.example.com/v1", api_key_env: null,
        api_key_set: false, api_key_hint: "", configured: true, disabled: true,
        origin: withReset ? "ui" : "file", has_base: withReset },
    ],
    models: [
      { id: "chat", provider: "lmstudio", model: "chat", context_window: 32768, max_tokens: 8192,
        pricing: null, disabled: false, origin: "ui" },
      { id: "a-much-longer-model-name", provider: "lmstudio", model: "old", context_window: 32768, max_tokens: 8192,
        pricing: null, disabled: true, origin: withReset ? "ui" : "file", has_base: withReset },
    ],
    routing: { default: "chat", fallback: [], roles: {}, light: null },
  } as ModelsView;
}

async function lefts(cards: Locator, testId: string): Promise<number[]> {
  const result = [];
  for (const card of await cards.all()) result.push((await card.getByTestId(testId).boundingBox())!.x);
  return result;
}

function expectColumn(values: number[]): void {
  expect(values.length).toBeGreaterThan(1);
  expect(Math.max(...values) - Math.min(...values)).toBeLessThan(2);
}

async function openModels(page: Page, view: ModelsView): Promise<void> {
  await page.route("**/api/settings/models**", async route => {
    const path = new URL(route.request().url()).pathname.replace("/api/settings/models", "");
    if (path === "/local-servers") return route.fulfill({ json: [
      { server: "lmstudio", base_url: "http://127.0.0.1:1234/v1", models: 3, provider: null },
      { server: "ollama", base_url: "http://127.0.0.1:11434/v1", models: 12, provider: "ollama" },
    ] });
    return route.fulfill({ json: view });
  });
  await page.goto("/settings/models");
  await expect(page.locator("[data-model]")).toHaveCount(2);
}

test("model and connection buttons keep their columns when a card is switched off", async ({ page }, info) => {
  await openApp(page, info);
  await openModels(page, models(false));
  const providers = page.locator("[data-provider]");
  const cards = page.locator("[data-model]");
  for (const id of ["provider-discover", "provider-edit-url", "provider-edit-key", "provider-toggle", "provider-remove"]) {
    expectColumn(await lefts(providers, id));
  }
  for (const id of ["model-probe", "model-toggle", "model-remove"]) expectColumn(await lefts(cards, id));
  await expectNoHorizontalOverflow(page);
});

test("an extra reset button does not move the other buttons", async ({ page }, info) => {
  await openApp(page, info);
  await openModels(page, models(true));
  const providers = page.locator("[data-provider]");
  const cards = page.locator("[data-model]");
  await expect(page.getByTestId("model-reset")).toHaveCount(1);
  for (const id of ["provider-discover", "provider-edit-url", "provider-edit-key", "provider-toggle", "provider-remove"]) {
    expectColumn(await lefts(providers, id));
  }
  for (const id of ["model-probe", "model-toggle", "model-remove"]) expectColumn(await lefts(cards, id));
  const shots = process.env.UI_E2E_SHOTS;
  if (shots) await page.screenshot({ path: `${shots}/${info.project.name}-reset.png`, fullPage: true });
});

test("local servers keep their addresses in one column", async ({ page }, info) => {
  await openApp(page, info);
  await openModels(page, models(false));
  await page.getByTestId("local-scan").click();
  const rows = page.locator("[data-local]");
  await expect(rows).toHaveCount(2);
  const xs = [];
  for (const row of await rows.all()) xs.push((await row.locator(".mono").boundingBox())!.x);
  expectColumn(xs);
  await expectNoHorizontalOverflow(page);
});

test("speech profile buttons keep their columns", async ({ page }, info) => {
  await openApp(page, info);
  await page.route("**/api/settings/speech", route => route.fulfill({ json: { profiles: [
    { id: "local", base_url: "http://127.0.0.1:8178", model: "large-v3-turbo", language: "ru", disabled: false },
    { id: "spare", base_url: "http://127.0.0.1:8179", model: "small", language: "ru", disabled: false },
    { id: "very-long-speech-profile-name", base_url: "http://speech.internal.example:9000/v1", model: "parakeet", language: "auto", disabled: true },
  ] } }));
  await page.goto("/settings/speech");
  const cards = page.locator("[data-profile]");
  await expect(cards).toHaveCount(3);
  for (const id of ["speech-test", "speech-toggle", "speech-remove"]) expectColumn(await lefts(cards, id));
  await expectNoHorizontalOverflow(page);
  const shots = process.env.UI_E2E_SHOTS;
  if (shots) await page.screenshot({ path: `${shots}/${info.project.name}-speech.png`, fullPage: true });
});
