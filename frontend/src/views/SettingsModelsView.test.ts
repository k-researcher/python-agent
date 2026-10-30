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
      models: ["qwen/q", "embedder"],
      details: [
        { id: "qwen/q", context_window: 262144, kind: "llm" },
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
    expect(found.get('[data-found="qwen/q"]').text()).toContain("контекст 256K");
    // An embedding model is listed but cannot be added as an agent model.
    expect(found.get('[data-found="embedder"]').text()).toContain("модель эмбеддингов");
    expect(found.get('[data-found="embedder"]').find('[data-testid="found-add"]').exists()).toBe(false);
  });

  it("adds a found model with its context length and a free price", async () => {
    calls.get.mockResolvedValue(withLocalProvider(baseView()));
    calls.testProvider.mockResolvedValue({
      ok: true,
      status: 200,
      models: ["qwen/q"],
      details: [{ id: "qwen/q", context_window: 131072, kind: "llm" }],
    });
    const added = withLocalProvider(baseView());
    added.models.push({ ...added.models[0], id: "qwen/q", provider: "lmstudio", model: "qwen/q", origin: "ui" });
    calls.putModel.mockResolvedValue(added);
    const wrapper = await mountView();
    await wrapper.get('[data-provider="lmstudio"] [data-testid="provider-discover"]').trigger("click");
    await flushPromises();
    await wrapper.get('[data-found="qwen/q"] [data-testid="found-add"]').trigger("click");
    await flushPromises();

    expect(calls.putModel).toHaveBeenCalledWith("qwen/q", {
      provider: "lmstudio",
      model: "qwen/q",
      context_window: 131072,
      max_tokens: 8192,
      pricing: { input_per_mtok: 0, output_per_mtok: 0 },
    });
    expect(useAppStore().loadConfig).toHaveBeenCalled();
    expect(wrapper.get('[data-found="qwen/q"]').text()).toContain("Добавлена как qwen/q");
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

  it("shows the server's refusal instead of hiding it", async () => {
    calls.putModel.mockRejectedValue(new ApiError("Sessions use these models; archive them first: main", 409));
    const wrapper = await mountView();
    await wrapper.get('[data-model="main"]').findAll("button")[0].trigger("click");
    await flushPromises();
    expect(wrapper.get('[data-testid="settings-error"]').text()).toContain("archive them first");
  });
});
