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
    ],
  });
  await router.push("/settings/models");
  const wrapper = mount(SettingsModelsView, { global: { plugins: [router] } });
  await flushPromises();
  return wrapper;
}

beforeEach(() => {
  setActivePinia(createPinia());
  for (const call of Object.values(calls)) call.mockReset();
  calls.get.mockResolvedValue(baseView());
  calls.scanLocal.mockResolvedValue([]);
  calls.probeModel.mockResolvedValue({ ok: true, status: 200, seconds: 0.4 });
  calls.probeFound.mockResolvedValue({ ok: true, status: 200, seconds: 0.4 });
  vi.spyOn(useAppStore(), "loadConfig").mockResolvedValue();
});

describe("SettingsModelsView", () => {
  it("lists providers and models without showing the key", async () => {
    const wrapper = await mountView();
    const card = wrapper.get('[data-provider="gateway"]');
    expect(card.text()).toContain("https://llm.example.com/v1");
    expect(card.text()).toContain("Ключ задан …abcd");
    expect(card.text()).toContain("Из файла");
    // A provider from the file can be switched off but not removed here.
    expect(card.find('[data-testid="provider-remove"]').exists()).toBe(false);
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

    expect(calls.putProvider).toHaveBeenCalledWith("lmstudio", { kind: "openai", base_url: "http://127.0.0.1:1234/v1" });
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
    });
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
    expect(calls.putModel).toHaveBeenCalledWith("main", { context_window: 256000, max_tokens: 8000 });
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
    expect(calls.putRouting).toHaveBeenNthCalledWith(1, { default: "light" });
    expect(calls.putRouting).toHaveBeenNthCalledWith(2, { light: "main" });
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
    expect(calls.putProvider).toHaveBeenCalledWith("lmstudio", { base_url: "http://127.0.0.1:1235/v1" });
    expect(card.find('[data-testid="provider-url-form"]').exists()).toBe(false);
    expect(card.find('[data-testid="found-models"]').exists()).toBe(false);
  });

  it("offers only gateway models that answered a check", async () => {
    calls.testProvider.mockResolvedValue({
      ok: true,
      status: 200,
      models: ["chat-fast", "chat-broken", "vectors"],
      details: [
        { id: "chat-fast", context_window: 262144, max_output_tokens: 262144, kind: "chat" },
        { id: "chat-broken", context_window: null, kind: null },
        { id: "vectors", context_window: 8192, kind: "embedding" },
      ],
    });
    calls.probeFound.mockImplementation(async (_provider: string, model: string) =>
      model === "chat-broken" ? { ok: false, status: 500, seconds: 5.1 } : { ok: true, status: 200, seconds: 0.4 },
    );
    const wrapper = await mountView();
    await wrapper.get('[data-provider="gateway"] [data-testid="provider-discover"]').trigger("click");
    await flushPromises();

    // The embedding model is not a chat model, so it is not checked.
    expect(calls.probeFound.mock.calls).toEqual([
      ["gateway", "chat-fast"],
      ["gateway", "chat-broken"],
    ]);
    const card = wrapper.get('[data-provider="gateway"]');
    expect(card.get('[data-found="chat-fast"] [data-testid="found-check"]').text()).toBe("Отвечает за 0,4 с");
    expect(card.find('[data-found="chat-fast"] [data-testid="found-add"]').exists()).toBe(true);
    expect(card.find('[data-found="chat-broken"]').exists()).toBe(false);
    expect(card.get('[data-testid="found-dead"]').text()).toContain("chat-broken (HTTP 500)");
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
    expect(calls.putProvider).toHaveBeenCalledWith("lmstudio", { kind: "openai", base_url: "http://127.0.0.1:1235/v1" });
    expect(calls.testProvider).toHaveBeenCalledWith("lmstudio");
    expect(wrapper.get('[data-local="http://127.0.0.1:1235/v1"]').text()).toContain("Подключён как lmstudio");
  });

  it("explains when no local server runs", async () => {
    const wrapper = await mountView();
    expect(wrapper.get('[data-testid="local-servers"]').text()).toContain("Запущенных серверов моделей не найдено");
  });

  it("checks cloud agent models on open and marks a dead one", async () => {
    calls.probeModel.mockResolvedValue({ ok: false, status: 500, seconds: 5 });
    const wrapper = await mountView();
    expect(calls.probeModel).toHaveBeenCalledWith("main");
    expect(wrapper.get('[data-model="main"] [data-testid="model-check"]').text()).toBe(
      "Не отвечает: HTTP 500 · не предлагается для новых сессий",
    );
    expect(useAppStore().loadConfig).toHaveBeenCalled();
  });

  it("does not let a model in a role be switched off or removed", async () => {
    const view = baseView();
    view.models.push({ ...view.models[0], id: "light", origin: "ui" });
    view.routing = { ...view.routing, light: "light" };
    calls.get.mockResolvedValue(view);
    const wrapper = await mountView();
    const main = wrapper.get('[data-model="main"]');
    expect(main.get('[data-testid="model-roles"]').text()).toBe("Роль: основная");
    expect(main.get('[data-testid="model-toggle"]').attributes("disabled")).toBeDefined();
    const light = wrapper.get('[data-model="light"]');
    expect(light.get('[data-testid="model-roles"]').text()).toBe("Роль: служебная");
    expect(light.get('[data-testid="model-remove"]').attributes("disabled")).toBeDefined();
    expect(light.get('[data-testid="model-remove"]').attributes("title")).toContain("сначала выберите");
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
});
