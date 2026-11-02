import { expect, test } from "@playwright/test";
import type { ModelsView } from "../src/api/modelSettings";
import { expectNoHorizontalOverflow, mockConfig, openApp } from "./helpers";

function settings(): ModelsView {
  return {
    checksum: "version-0", deleted: { providers: [], models: [] }, warnings: [],
    providers: [{ id: "gateway", kind: "openai", base_url: "https://llm.example.com/v1",
      api_key_env: "API_KEY", key_env: "API_KEY", key_source: "env", api_key_set: true,
      api_key_hint: "…abcd", configured: true, disabled: false, origin: "file", has_base: true }],
    models: [{ id: "chat", provider: "gateway", model: "chat", context_window: 32768,
      max_tokens: 8192, pricing: null, disabled: false, provider_disabled: false,
      origin: "file", has_base: true }],
    routing: { default: "chat", fallback: [], roles: { pm: "chat" }, light: null },
  };
}

test("model settings keep checks explicit and separate key edit, disable, delete and restore", async ({ page }, info) => {
  await openApp(page, info);
  let view = settings();
  let revision = 0;
  const probes: string[] = [];
  const changes: Array<{ path: string; body: Record<string, unknown> | null }> = [];
  await page.route("**/api/config/public", route => route.fulfill({ json: view.providers.length && !view.providers[0].disabled
    ? mockConfig : { ...mockConfig, llm_configured: false, llm_profiles: [], default_llm_profile: "" } }));
  await page.route("**/api/settings/models**", async route => {
    const request = route.request();
    const path = new URL(request.url()).pathname.replace("/api/settings/models", "");
    if (request.method() === "GET") return route.fulfill({ json: view });
    if (path === "/local-servers") return route.fulfill({ json: [] });
    if (path.endsWith("/test")) return route.fulfill({ json: { ok: true, status: 200, models: ["found-chat"] } });
    if (path.endsWith("/probe")) {
      probes.push(path);
      return route.fulfill({ json: { ok: true, status: 200, seconds: 0.1 } });
    }
    expect(request.headers()["if-match"]).toBe(view.checksum);
    const body = request.method() === "PUT" ? request.postDataJSON() : null;
    changes.push({ path, body });
    if (request.method() === "DELETE") {
      view.providers = []; view.models = []; view.routing.default = "";
      view.deleted = { providers: ["gateway"], models: [] };
    } else if (path.endsWith("/reset")) view = settings();
    else if (body && "disabled" in body) {
      view.providers[0].disabled = body.disabled;
      view.providers[0].origin = "ui";
      view.models[0].provider_disabled = body.disabled;
    } else if (body?.key_source) {
      view.providers[0].key_source = body.key_source;
      view.providers[0].origin = "ui";
    }
    view.checksum = `version-${++revision}`;
    return route.fulfill({ json: view });
  });
  await page.goto("/settings/models");
  const provider = page.locator('[data-provider="gateway"]');
  await expect(provider).toBeVisible();
  expect(probes).toEqual([]);
  await provider.getByTestId("provider-discover").click();
  await expect(page.locator('[data-found="found-chat"]')).toBeVisible();
  expect(probes).toEqual([]);
  await page.locator('[data-found="found-chat"]').getByTestId("found-probe").click();
  await expect.poll(() => probes.length).toBe(1);
  await provider.getByTestId("provider-edit-key").click();
  await provider.getByTestId("provider-key-source").selectOption("ui");
  await provider.getByTestId("provider-key-value").fill("test-only-new-key");
  await provider.getByRole("button", { name: "Сохранить ключ", exact: true }).click();
  await expect(provider.getByTestId("provider-key-form")).toHaveCount(0);
  expect(changes[0].body).toEqual({ key_source: "ui", api_key: "test-only-new-key" });
  await provider.getByTestId("provider-toggle").click();
  await expect(page.getByTestId("models-empty")).toBeVisible();
  await expect(page.getByTestId("model-provider-off")).toHaveText("Сервер отключён");
  // Removal asks first and names the models that go with the connection.
  page.once("dialog", (dialog) => {
    expect(dialog.type()).toBe("confirm");
    expect(dialog.message()).toContain("Удалить подключение");
    void dialog.accept();
  });
  await provider.getByTestId("provider-remove").click();
  await expect(provider).toHaveCount(0);
  await expect(page.getByTestId("deleted-settings")).toBeVisible();
  await page.locator('[data-restore-provider="gateway"]').click();
  await expect(provider).toBeVisible();
  await expect(page.locator('[data-routing-role="pm"]')).toHaveValue("chat");
  await expectNoHorizontalOverflow(page);
  expect(probes.length).toBe(1);
  expect(changes.map(change => change.path)).toEqual([
    "/providers/gateway", "/providers/gateway", "/providers/gateway", "/providers/gateway/reset",
  ]);
  await page.screenshot({ path: info.outputPath("model-settings.png") });
});

test("an empty catalog keeps the app and history usable with sending disabled", async ({ page }, info) => {
  await openApp(page, info);
  await page.route("**/api/config/public", route => route.fulfill({ json: {
    ...mockConfig, llm_configured: false, llm_profiles: [], default_llm_profile: "",
  } }));
  await page.reload();
  await expect(page.getByTestId("no-models")).toBeVisible();
  await expect(page.getByTestId("composer-input")).toBeDisabled();
  await expect(page.getByTestId("composer-send")).toBeDisabled();
  await page.goto("/s/s1");
  await expect(page.getByTestId("session-model-unavailable")).toContainText("История сохранена");
  await expect(page.getByTestId("chat-timeline")).toBeVisible();
  await expect(page.getByTestId("composer-send")).toBeDisabled();
  await expectNoHorizontalOverflow(page);
});
