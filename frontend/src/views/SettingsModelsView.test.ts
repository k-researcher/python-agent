import { beforeEach, describe, expect, it, vi } from "vitest";
import { flushPromises, mount } from "@vue/test-utils";
import { createPinia, setActivePinia } from "pinia";
import { createMemoryHistory, createRouter } from "vue-router";
import type { ModelsView } from "../api/modelSettings";
import { ApiError } from "../api/client";

const calls = vi.hoisted(() => ({
  get: vi.fn(),
  putProvider: vi.fn(),
  putModel: vi.fn(),
  putRouting: vi.fn(),
  removeProvider: vi.fn(),
  removeModel: vi.fn(),
  resetProvider: vi.fn(),
  resetModel: vi.fn(),
  testProvider: vi.fn(),
  probeFound: vi.fn(),
  probeModel: vi.fn(),
  scanLocal: vi.fn(),
}));
vi.mock("../api/modelSettings", () => ({ modelSettings: calls }));

import SettingsModelsView from "./SettingsModelsView.vue";
import { useAppStore } from "../stores/app";

function baseView(): ModelsView {
  return {
    checksum: "version-1",
    deleted: { providers: [], models: [] },
    providers: [
      {
        id: "gateway",
        kind: "openai",
        base_url: "https://llm.example.com/v1",
        api_key_env: "AGENT_LLM_API_KEY",
        api_key_set: true,
        api_key_hint: "…abcd",
        configured: true,
        disabled: false,
        origin: "file",
        has_base: true,
        key_source: "env",
        key_env: "AGENT_LLM_API_KEY",
      },
    ],
    models: [
      {
        id: "main",
        provider: "gateway",
        model: "remote",
        context_window: 256000,
        max_tokens: 16000,
        pricing: null,
        disabled: false,
        origin: "file",
      },
    ],
    routing: { default: "main", fallback: [], roles: {}, light: null },
    warnings: [],
  };
}

function withLocalProvider(view: ModelsView): ModelsView {
  return {
    ...view,
    providers: [
      ...view.providers,
      {
        id: "lmstudio",
        kind: "openai",
        base_url: "http://127.0.0.1:1234/v1",
        api_key_env: null,
        api_key_set: false,
        api_key_hint: "",
        configured: true,
        disabled: false,
        origin: "ui",
      },
    ],
  };
}

async function mountView() {
  const router = createRouter({
    history: createMemoryHistory(),
    routes: [
      { path: "/", name: "home", component: { template: "<div/>" } },
      { path: "/settings/models", name: "settings-models", component: SettingsModelsView },
      { path: "/settings/speech", name: "settings-speech", component: { template: "<div/>" } },
    ],
  });
  await router.push("/settings/models");
  const wrapper = mount(SettingsModelsView, { global: { plugins: [router] } });
  await flushPromises();
  return wrapper;
}

const confirm = vi.fn((_question?: string) => true);

beforeEach(() => {
  confirm.mockReset().mockReturnValue(true);
  Object.defineProperty(window, "confirm", { value: confirm, configurable: true });
  setActivePinia(createPinia());
  for (const call of Object.values(calls)) call.mockReset();
  calls.get.mockResolvedValue(baseView());
  calls.scanLocal.mockResolvedValue([]);
  calls.probeModel.mockResolvedValue({ ok: true, status: 200, seconds: 0.4 });
  calls.probeFound.mockResolvedValue({ ok: true, status: 200, seconds: 0.4 });
  vi.spyOn(useAppStore(), "loadConfig").mockResolvedValue();
});

describe("SettingsModelsView", () => {
  it("does not generate when settings open or when models are listed", async () => {
    calls.testProvider.mockResolvedValue({ ok: true, status: 200, models: ["chat"] });
    const wrapper = await mountView();
    expect(calls.probeModel).not.toHaveBeenCalled();
    await wrapper.get('[data-provider="gateway"] [data-testid="provider-discover"]').trigger("click");
    await flushPromises();
    expect(calls.probeFound).not.toHaveBeenCalled();
    expect(wrapper.find('[data-found="chat"] [data-testid="found-add"]').exists()).toBe(true);
  });

  it("separates deleting a file provider from resetting its overrides", async () => {
    const view = baseView();
    view.providers[0] = { ...view.providers[0], origin: "ui", has_base: true };
    calls.get.mockResolvedValue(view);
    const wrapper = await mountView();
    const card = wrapper.get('[data-provider="gateway"]');
    expect(card.get('[data-testid="provider-remove"]').text()).toBe("Удалить");
    expect(card.get('[data-testid="provider-reset"]').text()).toBe("Сбросить изменения");
    expect(card.find('[data-testid="provider-edit-key"]').exists()).toBe(true);
  });
  it("lists providers and models without showing the key", async () => {
    const wrapper = await mountView();
    const card = wrapper.get('[data-provider="gateway"]');
    expect(card.text()).toContain("https://llm.example.com/v1");
    expect(card.text()).toContain("Ключ задан …abcd");
    expect(card.text()).toContain("Из файла");
    expect(card.find('[data-testid="provider-remove"]').exists()).toBe(true);
    expect(wrapper.get('[data-model="main"]').text()).toContain("remote");
  });

  it("connects a local server without a key and finds its models at once", async () => {
    calls.putProvider.mockResolvedValue(withLocalProvider(baseView()));
    calls.testProvider.mockResolvedValue({
      ok: true,
      status: 200,
      models: ["publisher/chat", "embedder"],
      details: [
        { id: "publisher/chat", context_window: 262144, kind: "llm" },
        { id: "embedder", context_window: 2048, kind: "embeddings" },
      ],
    });
    const wrapper = await mountView();
    await wrapper.get('[data-testid="provider-add-open"]').trigger("click");
    expect((wrapper.get('[data-testid="provider-url"]').element as HTMLInputElement).value).toBe(
      "http://127.0.0.1:1234/v1",
    );
    expect(wrapper.find('[data-testid="provider-key"]').exists()).toBe(false);
    await wrapper.get('[data-testid="provider-form"]').trigger("submit");
    await flushPromises();

    expect(calls.putProvider).toHaveBeenCalledWith("lmstudio", { kind: "openai", base_url: "http://127.0.0.1:1234/v1", key_source: "none" }, { checksum: "version-1", create: true });
    expect(calls.testProvider).toHaveBeenCalledWith("lmstudio");
    const found = wrapper.get('[data-provider="lmstudio"] [data-testid="found-models"]');
    expect(found.get('[data-found="publisher/chat"]').text()).toContain("контекст 256K");
    // An embedding model is listed but cannot be added as an agent model.
    expect(found.get('[data-found="embedder"]').text()).toContain("модель эмбеддингов");
    expect(found.get('[data-found="embedder"]').find('[data-testid="found-add"]').exists()).toBe(false);
  });

  it("adds a found model with its context length and a free price", async () => {
    calls.get.mockResolvedValue(withLocalProvider(baseView()));
    calls.testProvider.mockResolvedValue({
      ok: true,
      status: 200,
      models: ["publisher/chat"],
      details: [{ id: "publisher/chat", context_window: 131072, kind: "llm" }],
    });
    const added = withLocalProvider(baseView());
    added.models.push({ ...added.models[0], id: "publisher/chat", provider: "lmstudio", model: "publisher/chat", origin: "ui" });
    calls.putModel.mockResolvedValue(added);
    const wrapper = await mountView();
    await wrapper.get('[data-provider="lmstudio"] [data-testid="provider-discover"]').trigger("click");
    await flushPromises();
    await wrapper.get('[data-found="publisher/chat"] [data-testid="found-add"]').trigger("click");
    await flushPromises();

    expect(calls.putModel).toHaveBeenCalledWith("publisher/chat", {
      provider: "lmstudio",
      model: "publisher/chat",
      context_window: 131072,
      max_tokens: 8192,
      pricing: { input_per_mtok: 0, output_per_mtok: 0 },
    }, { checksum: "version-1", create: true });
    expect(useAppStore().loadConfig).toHaveBeenCalled();
    expect(wrapper.get('[data-found="publisher/chat"]').text()).toContain("Добавлена как publisher/chat");
  });

  it("explains a server that does not answer", async () => {
    calls.get.mockResolvedValue(withLocalProvider(baseView()));
    calls.testProvider.mockResolvedValue({ ok: false, status: null, models: [], details: [] });
    const wrapper = await mountView();
    await wrapper.get('[data-provider="lmstudio"] [data-testid="provider-discover"]').trigger("click");
    await flushPromises();
    expect(wrapper.get('[data-provider="lmstudio"]').text()).toContain("Сервер не отвечает");
  });

  it("saves edited limits only when they are valid", async () => {
    calls.putModel.mockResolvedValue(baseView());
    const wrapper = await mountView();
    const card = wrapper.get('[data-model="main"]');
    await card.get('[data-testid="model-max-tokens"]').setValue(300000);
    expect(card.text()).toContain("Ответ должен быть меньше контекста");
    expect(card.get('[data-testid="model-save"]').attributes("disabled")).toBeDefined();
    await card.get('[data-testid="model-max-tokens"]').setValue(8000);
    await card.get('[data-testid="model-save"]').trigger("click");
    await flushPromises();
    expect(calls.putModel).toHaveBeenCalledWith("main", { context_window: 256000, max_tokens: 8000 }, { checksum: "version-1" });
  });

  it("warns that a model without a price cannot start under the session cost limit", async () => {
    useAppStore().config = { session_cost_limit: "1" } as never;
    const wrapper = await mountView();
    const card = wrapper.get('[data-model="main"]');
    expect(card.get('[data-testid="model-price"]').text()).toBe("Цена не указана");
    expect(card.get('[data-testid="model-price-warning"]').text()).toContain("лимит стоимости $1");

    useAppStore().config = { session_cost_limit: null } as never;
    await flushPromises();
    expect(card.find('[data-testid="model-price-warning"]').exists()).toBe(false);
  });

  it("marks a model as free with one choice and saves the zero price", async () => {
    useAppStore().config = { session_cost_limit: "1" } as never;
    const free = baseView();
    free.models[0].pricing = { input_per_mtok: 0, output_per_mtok: 0 };
    calls.putModel.mockResolvedValue(free);
    const wrapper = await mountView();
    const card = wrapper.get('[data-model="main"]');
    await card.get('[data-testid="model-price-mode"]').setValue("free");
    await card.get('[data-testid="model-save"]').trigger("click");
    await flushPromises();
    expect(calls.putModel).toHaveBeenCalledWith(
      "main",
      { context_window: 256000, max_tokens: 16000, pricing: { input_per_mtok: 0, output_per_mtok: 0 } },
      { checksum: "version-1" },
    );
    expect(card.get('[data-testid="model-price"]').text()).toBe("Без оплаты");
    expect(card.find('[data-testid="model-price-warning"]').exists()).toBe(false);
    expect(card.find('[data-testid="model-save"]').exists()).toBe(false);
  });

  it("saves a paid price and refuses a negative one", async () => {
    calls.putModel.mockResolvedValue(baseView());
    const wrapper = await mountView();
    const card = wrapper.get('[data-model="main"]');
    await card.get('[data-testid="model-price-mode"]').setValue("paid");
    await card.get('[data-testid="model-price-input"]').setValue(-1);
    expect(card.text()).toContain("Цена — число от 0");
    expect(card.get('[data-testid="model-save"]').attributes("disabled")).toBeDefined();
    await card.get('[data-testid="model-price-input"]').setValue(0.14);
    await card.get('[data-testid="model-price-output"]').setValue(0.28);
    await card.get('[data-testid="model-save"]').trigger("click");
    await flushPromises();
    expect(calls.putModel).toHaveBeenCalledWith(
      "main",
      { context_window: 256000, max_tokens: 16000, pricing: { input_per_mtok: 0.14, output_per_mtok: 0.28 } },
      { checksum: "version-1" },
    );
  });

  it("shows a paid price in plain words", async () => {
    const paid = baseView();
    paid.models[0].pricing = { input_per_mtok: 0.14, output_per_mtok: 0.28 };
    calls.get.mockResolvedValue(paid);
    const wrapper = await mountView();
    expect(wrapper.get('[data-model="main"] [data-testid="model-price"]').text()).toBe(
      "Платная: вход $0.14, выход $0.28 за 1M токенов",
    );
  });

  it("keeps the delete button active and readable on a switched off model", async () => {
    const off = baseView();
    off.models[0].disabled = true;
    calls.get.mockResolvedValue(off);
    const wrapper = await mountView();
    const remove = wrapper.get('[data-model="main"] [data-testid="model-remove"]');
    expect(remove.attributes("disabled")).toBeUndefined();
    expect(remove.classes()).toContain("danger");
  });

  it("changes the main and the service model", async () => {
    const two = baseView();
    two.models.push({ ...two.models[0], id: "light" });
    calls.get.mockResolvedValue(two);
    calls.putRouting.mockResolvedValue(two);
    const wrapper = await mountView();
    await wrapper.get('[data-testid="routing-default"]').setValue("light");
    // The selects stay locked while a change is saved.
    await flushPromises();
    await wrapper.get('[data-testid="routing-light"]').setValue("main");
    await flushPromises();
    expect(calls.putRouting).toHaveBeenNthCalledWith(1, { default: "light" }, { checksum: "version-1" });
    expect(calls.putRouting).toHaveBeenNthCalledWith(2, { light: "main" }, { checksum: "version-1" });
  });

  it("changes the address of a connection and forgets the old search result", async () => {
    calls.get.mockResolvedValue(withLocalProvider(baseView()));
    calls.testProvider.mockResolvedValue({ ok: false, status: null, models: [], details: [] });
    calls.putProvider.mockResolvedValue(withLocalProvider(baseView()));
    const wrapper = await mountView();
    const card = wrapper.get('[data-provider="lmstudio"]');
    await card.get('[data-testid="provider-discover"]').trigger("click");
    await flushPromises();
    await card.get('[data-testid="provider-edit-url"]').trigger("click");
    const input = card.get('[data-testid="provider-url-edit"]');
    expect((input.element as HTMLInputElement).value).toBe("http://127.0.0.1:1234/v1");
    await input.setValue("not an address");
    expect(card.get('[data-testid="provider-url-form"] button[type=submit]').attributes("disabled")).toBeDefined();
    await input.setValue("http://127.0.0.1:1235/v1");
    await card.get('[data-testid="provider-url-form"]').trigger("submit");
    await flushPromises();
    expect(calls.putProvider).toHaveBeenCalledWith("lmstudio", { base_url: "http://127.0.0.1:1235/v1" }, { checksum: "version-1" });
    expect(card.find('[data-testid="provider-url-form"]').exists()).toBe(false);
    expect(card.find('[data-testid="found-models"]').exists()).toBe(false);
  });

  it("lists remote models without generation and checks only on an explicit action", async () => {
    calls.testProvider.mockResolvedValue({ ok: true, status: 200, models: ["chat", "vectors"], details: [
      { id: "chat", context_window: 262144, kind: "chat" },
      { id: "vectors", context_window: 8192, kind: "embedding" },
    ] });
    const wrapper = await mountView();
    await wrapper.get('[data-provider="gateway"] [data-testid="provider-discover"]').trigger("click");
    await flushPromises();
    expect(calls.probeFound).not.toHaveBeenCalled();
    const row = wrapper.get('[data-found="chat"]');
    expect(row.find('[data-testid="found-add"]').exists()).toBe(true);
    expect(wrapper.find('[data-found="vectors"] [data-testid="found-add"]').exists()).toBe(false);
    await row.get('[data-testid="found-probe"]').trigger("click");
    await flushPromises();
    expect(calls.probeFound).toHaveBeenCalledExactlyOnceWith("gateway", "chat");
  });

  it("does not check local models by itself, because a check loads the model", async () => {
    calls.get.mockResolvedValue(withLocalProvider(baseView()));
    calls.testProvider.mockResolvedValue({
      ok: true,
      status: 200,
      models: ["publisher/chat"],
      details: [{ id: "publisher/chat", context_window: 131072, kind: "vlm", state: "not-loaded" }],
    });
    const wrapper = await mountView();
    await wrapper.get('[data-provider="lmstudio"] [data-testid="provider-discover"]').trigger("click");
    await flushPromises();
    expect(calls.probeFound).not.toHaveBeenCalled();
    const row = wrapper.get('[data-found="publisher/chat"]');
    expect(row.text()).toContain("не загружена");
    expect(row.find('[data-testid="found-add"]').exists()).toBe(true);
    await row.get('[data-testid="found-probe"]').trigger("click");
    await flushPromises();
    expect(calls.probeFound).toHaveBeenCalledWith("lmstudio", "publisher/chat");
  });

  it("finds a running local server and connects it with one click", async () => {
    calls.scanLocal.mockResolvedValue([
      { base_url: "http://127.0.0.1:1235/v1", server: "lmstudio", models: 6, provider: null },
    ]);
    calls.putProvider.mockResolvedValue(withLocalProvider(baseView()));
    calls.testProvider.mockResolvedValue({ ok: true, status: 200, models: [], details: [] });
    const wrapper = await mountView();
    const row = wrapper.get('[data-local="http://127.0.0.1:1235/v1"]');
    expect(row.text()).toContain("LM Studio");
    expect(row.text()).toContain("моделей: 6");
    await row.get('[data-testid="local-connect"]').trigger("click");
    await flushPromises();
    expect(calls.putProvider).toHaveBeenCalledWith("lmstudio", { kind: "openai", base_url: "http://127.0.0.1:1235/v1", key_source: "none" }, { checksum: "version-1", create: true });
    expect(calls.testProvider).toHaveBeenCalledWith("lmstudio");
    expect(wrapper.get('[data-local="http://127.0.0.1:1235/v1"]').text()).toContain("Подключён как lmstudio");
  });

  it("explains when no local server runs", async () => {
    const wrapper = await mountView();
    expect(wrapper.get('[data-testid="local-servers"]').text()).toContain("Запущенных серверов моделей не найдено");
  });

  it("checks an agent model only after its check button is pressed", async () => {
    calls.probeModel.mockResolvedValue({ ok: false, status: 500, seconds: 5 });
    const wrapper = await mountView();
    expect(calls.probeModel).not.toHaveBeenCalled();
    await wrapper.get('[data-model="main"] [data-testid="model-probe"]').trigger("click");
    await flushPromises();
    expect(calls.probeModel).toHaveBeenCalledWith("main");
    expect(wrapper.get('[data-model="main"] [data-testid="model-check"]').text()).toBe(
      "Не отвечает: HTTP 500 · не предлагается для новых сессий",
    );
    expect(useAppStore().loadConfig).toHaveBeenCalled();
  });

  it("lets the server update dependent roles when a model is disabled or deleted", async () => {
    const view = baseView();
    view.models.push({ ...view.models[0], id: "light", origin: "ui" });
    view.routing = { ...view.routing, light: "light" };
    calls.get.mockResolvedValue(view);
    const wrapper = await mountView();
    const main = wrapper.get('[data-model="main"]');
    expect(main.get('[data-testid="model-roles"]').text()).toBe("Роль: основная");
    expect(main.get('[data-testid="model-toggle"]').attributes("disabled")).toBeUndefined();
    const light = wrapper.get('[data-model="light"]');
    expect(light.get('[data-testid="model-roles"]').text()).toBe("Роль: служебная");
    expect(light.get('[data-testid="model-remove"]').attributes("disabled")).toBeUndefined();
  });

  it("shows the server's refusal instead of hiding it", async () => {
    const view = baseView();
    view.models.push({ ...view.models[0], id: "spare" });
    calls.get.mockResolvedValue(view);
    calls.putModel.mockRejectedValue(new ApiError("Sessions use these models; archive them first: spare", 409));
    const wrapper = await mountView();
    await wrapper.get('[data-model="spare"] [data-testid="model-toggle"]').trigger("click");
    await flushPromises();
    expect(wrapper.get('[data-testid="settings-error"]').text()).toContain("archive them first");
  });

  it("edits a saved key without revealing the old or new value", async () => {
    const saved = baseView();
    saved.providers[0] = { ...saved.providers[0], key_source: "ui", origin: "ui" };
    calls.putProvider.mockResolvedValue(saved);
    const wrapper = await mountView();
    const card = wrapper.get('[data-provider="gateway"]');
    await card.get('[data-testid="provider-edit-key"]').trigger("click");
    await card.get('[data-testid="provider-key-source"]').setValue("ui");
    const input = card.get('[data-testid="provider-key-value"]');
    expect(input.attributes("type")).toBe("password");
    expect((input.element as HTMLInputElement).value).toBe("");
    await input.setValue("private-new-value");
    await card.get('[data-testid="provider-key-form"]').trigger("submit");
    await flushPromises();
    expect(calls.putProvider).toHaveBeenCalledWith("gateway", { key_source: "ui", api_key: "private-new-value" }, { checksum: "version-1" });
    expect(card.find('[data-testid="provider-key-form"]').exists()).toBe(false);
    expect(wrapper.html()).not.toContain("private-new-value");
  });

  it("explicitly clears a key or selects an environment variable", async () => {
    calls.putProvider.mockResolvedValue(baseView());
    const wrapper = await mountView();
    const card = wrapper.get('[data-provider="gateway"]');
    await card.get('[data-testid="provider-edit-key"]').trigger("click");
    await card.get('[data-testid="provider-key-source"]').setValue("none");
    await card.get('[data-testid="provider-key-form"]').trigger("submit");
    await flushPromises();
    expect(calls.putProvider).toHaveBeenLastCalledWith("gateway", { key_source: "none" }, { checksum: "version-1" });
    await card.get('[data-testid="provider-edit-key"]').trigger("click");
    await card.get('[data-testid="provider-key-env"]').setValue("OTHER_KEY");
    await card.get('[data-testid="provider-key-form"]').trigger("submit");
    await flushPromises();
    expect(calls.putProvider).toHaveBeenLastCalledWith("gateway", { key_source: "env", api_key_env: "OTHER_KEY" }, { checksum: "version-1" });
  });

  it("shows deleted file settings last, after the model roles", async () => {
    const view = baseView();
    view.deleted = { providers: [], models: ["old"] };
    calls.get.mockResolvedValue(view);
    const wrapper = await mountView();
    const roles = wrapper.get("#roles-title").element;
    const deleted = wrapper.get('[data-testid="deleted-settings"]');
    expect(roles.compareDocumentPosition(deleted.element) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
    expect(deleted.text()).not.toContain("Они не используются");
  });

  it("uses reset only when the owner requests restoration", async () => {
    const removed = baseView();
    removed.providers = []; removed.models = [];
    removed.deleted = { providers: ["gateway"], models: ["chat"] };
    calls.get.mockResolvedValue(removed);
    calls.resetProvider.mockResolvedValue(baseView());
    const wrapper = await mountView();
    expect(calls.resetProvider).not.toHaveBeenCalled();
    expect(wrapper.get('[data-testid="models-empty"]').text()).toContain("Нет включённых моделей");
    expect(wrapper.get('[data-remove-hint-provider="gateway"]').text()).toContain("удалите блок gateway из раздела providers в config/models.yaml");
    expect(wrapper.get('[data-remove-hint-model="chat"]').text()).toContain("удалите блок chat из раздела models в config/models.yaml");
    await wrapper.get('[data-restore-provider="gateway"]').trigger("click");
    await flushPromises();
    expect(calls.resetProvider).toHaveBeenCalledExactlyOnceWith("gateway", { checksum: "version-1" });
  });

  it("deletes a provider from the file without resetting it", async () => {
    const removed = baseView(); removed.providers = []; removed.models = [];
    calls.removeProvider.mockResolvedValue(removed);
    const wrapper = await mountView();
    await wrapper.get('[data-provider="gateway"] [data-testid="provider-remove"]').trigger("click");
    await flushPromises();
    expect(calls.removeProvider).toHaveBeenCalledExactlyOnceWith("gateway", { checksum: "version-1" });
    expect(calls.resetProvider).not.toHaveBeenCalled();
    expect(wrapper.find('[data-provider="gateway"]').exists()).toBe(false);
  });

  it("asks before removing a connection with its models and keeps it when cancelled", async () => {
    const wrapper = await mountView();
    const models = baseView().models.filter((model) => model.provider === "gateway").length;
    confirm.mockReturnValue(false);
    await wrapper.get('[data-provider="gateway"] [data-testid="provider-remove"]').trigger("click");
    await flushPromises();
    expect(confirm).toHaveBeenCalledWith(
      `Удалить подключение «gateway»? Вместе с ним удалятся его модели: ${models}.`,
    );
    expect(calls.removeProvider).not.toHaveBeenCalled();
    expect(wrapper.find('[data-provider="gateway"]').exists()).toBe(true);
  });

  it("asks before removing a model", async () => {
    const wrapper = await mountView();
    confirm.mockReturnValue(false);
    const button = wrapper.findAll('[data-testid="model-remove"]').find((item) => item.attributes("disabled") === undefined);
    expect(button).toBeDefined();
    await button!.trigger("click");
    await flushPromises();
    expect(confirm.mock.calls[0][0]).toMatch(/^Удалить модель «.+»\?$/);
    expect(calls.removeModel).not.toHaveBeenCalled();
  });

  it("edits named role mappings and excludes models of disabled providers", async () => {
    const view = baseView();
    view.routing.roles = { review: "main" };
    view.models.push({ ...view.models[0], id: "off", provider_disabled: true });
    calls.get.mockResolvedValue(view); calls.putRouting.mockResolvedValue(view);
    const wrapper = await mountView();
    const role = wrapper.get('[data-routing-role="review"]');
    expect(role.findAll("option").map(option => option.attributes("value"))).toEqual(["", "main"]);
    await role.setValue(""); await flushPromises();
    expect(calls.putRouting).toHaveBeenCalledWith({ roles: {} }, { checksum: "version-1" });
    expect(wrapper.get('[data-model="off"] [data-testid="model-provider-off"]').text()).toContain("Сервер отключён");
  });

  it("reloads a conflicted version but keeps the typed key for review and retry", async () => {
    const fresh = baseView(); fresh.checksum = "version-2";
    calls.get.mockResolvedValueOnce(baseView()).mockResolvedValue(fresh);
    calls.putProvider.mockRejectedValueOnce(new ApiError("Настройки изменены", 409)).mockResolvedValue(fresh);
    const wrapper = await mountView();
    const card = wrapper.get('[data-provider="gateway"]');
    await card.get('[data-testid="provider-edit-key"]').trigger("click");
    await card.get('[data-testid="provider-key-source"]').setValue("ui");
    await card.get('[data-testid="provider-key-value"]').setValue("private-retry-value");
    await card.get('[data-testid="provider-key-form"]').trigger("submit");
    await flushPromises();
    expect(wrapper.get('[data-testid="settings-error"]').text()).toContain("Настройки изменены");
    expect((card.get('[data-testid="provider-key-value"]').element as HTMLInputElement).value).toBe("private-retry-value");
    await card.get('[data-testid="provider-key-form"]').trigger("submit");
    await flushPromises();
    expect(calls.putProvider).toHaveBeenLastCalledWith("gateway", { key_source: "ui", api_key: "private-retry-value" }, { checksum: "version-2" });
  });

  it("reloads mounted settings on an external change and preserves edits until save", async () => {
    const wrapper = await mountView();
    const fresh = baseView(); fresh.checksum = "version-2";
    calls.get.mockResolvedValue(fresh);
    useAppStore().modelsChecksum = "version-2";
    await flushPromises();
    expect(calls.get).toHaveBeenCalledTimes(2);
    await wrapper.get('[data-model="main"] [data-testid="model-max-tokens"]').setValue(8000);
    useAppStore().modelsChecksum = "version-3";
    await flushPromises();
    expect(calls.get).toHaveBeenCalledTimes(2);
    expect((wrapper.get('[data-model="main"] [data-testid="model-max-tokens"]').element as HTMLInputElement).value).toBe("8000");
    expect(wrapper.get('[data-testid="settings-error"]').text()).toContain("другом окне");
    wrapper.unmount();
    useAppStore().modelsChecksum = "version-4"; await flushPromises();
    expect(calls.get).toHaveBeenCalledTimes(2);
  });

  it("does not let a late read undo a successful mutation", async () => {
    let finish!: (value: ModelsView) => void;
    const late = new Promise<ModelsView>(resolve => { finish = resolve; });
    const saved = baseView(); saved.checksum = "version-3";
    saved.providers[0].disabled = true; saved.models[0].provider_disabled = true;
    calls.get.mockResolvedValueOnce(baseView()).mockReturnValueOnce(late).mockResolvedValue(saved);
    calls.putProvider.mockResolvedValue(saved);
    const wrapper = await mountView();
    useAppStore().modelsChecksum = "version-2";
    await flushPromises();
    await wrapper.get('[data-provider="gateway"] [data-testid="provider-toggle"]').trigger("click");
    await flushPromises();
    finish(baseView()); await flushPromises();
    expect(wrapper.get('[data-provider="gateway"] [data-testid="provider-toggle"]').text()).toBe("Включить");
    expect(wrapper.find('[data-testid="models-empty"]').exists()).toBe(true);
  });
});

it("clears a completed failed check after replacing the key", async () => {
  calls.probeModel.mockResolvedValue({ ok: false, status: 401, seconds: 0.1 });
  const saved = baseView();
  saved.checksum = "version-2";
  saved.providers[0] = { ...saved.providers[0], key_source: "ui", api_key_hint: "…wxyz" };
  calls.putProvider.mockResolvedValue(saved);
  const wrapper = await mountView();
  await wrapper.get('[data-testid="model-probe"]').trigger("click");
  await flushPromises();
  expect(wrapper.text()).toContain("HTTP 401");
  await wrapper.get('[data-testid="provider-edit-key"]').trigger("click");
  await wrapper.get('[data-testid="provider-key-source"]').setValue("ui");
  await wrapper.get('[data-testid="provider-key-value"]').setValue("replacement-value");
  await wrapper.get('[data-testid="provider-key-form"]').trigger("submit");
  await flushPromises();
  expect(wrapper.text()).not.toContain("HTTP 401");
  wrapper.unmount();
});

it("ignores an old failed check that finishes after replacing the key", async () => {
  let finish!: (value: { ok: boolean; status: number; seconds: number }) => void;
  calls.probeModel.mockReturnValue(new Promise(resolve => { finish = resolve; }));
  const saved = baseView();
  saved.checksum = "version-2";
  calls.putProvider.mockResolvedValue(saved);
  const wrapper = await mountView();
  await wrapper.get('[data-testid="model-probe"]').trigger("click");
  await wrapper.get('[data-testid="provider-edit-key"]').trigger("click");
  await wrapper.get('[data-testid="provider-key-source"]').setValue("ui");
  await wrapper.get('[data-testid="provider-key-value"]').setValue("replacement-value");
  await wrapper.get('[data-testid="provider-key-form"]').trigger("submit");
  await flushPromises();
  finish({ ok: false, status: 401, seconds: 0.1 });
  await flushPromises();
  expect(wrapper.text()).not.toContain("HTTP 401");
  wrapper.unmount();
});

it("ignores a late found check after rediscovery", async () => {
  let finish!: (value: { ok: boolean; status: number; seconds: number }) => void;
  calls.testProvider.mockResolvedValue({ ok: true, status: 200, models: ["chat"] });
  calls.probeFound.mockReturnValue(new Promise(resolve => { finish = resolve; }));
  const wrapper = await mountView();
  await wrapper.get('[data-testid="provider-discover"]').trigger("click");
  await flushPromises();
  await wrapper.get('[data-testid="found-probe"]').trigger("click");
  await wrapper.get('[data-testid="provider-discover"]').trigger("click");
  await flushPromises();
  finish({ ok: false, status: 401, seconds: 0.1 });
  await flushPromises();
  expect(wrapper.text()).not.toContain("HTTP 401");
  wrapper.unmount();
});

it("does not apply a late probe after unmount", async () => {
  let finish!: (value: { ok: boolean; status: number; seconds: number }) => void;
  calls.probeModel.mockReturnValue(new Promise(resolve => { finish = resolve; }));
  const wrapper = await mountView();
  await wrapper.get('[data-testid="model-probe"]').trigger("click");
  wrapper.unmount();
  finish({ ok: false, status: 401, seconds: 0.1 });
  await flushPromises();
  expect(useAppStore().loadConfig).not.toHaveBeenCalled();
});
