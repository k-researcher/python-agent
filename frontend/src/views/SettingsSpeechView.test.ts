import { flushPromises, mount } from "@vue/test-utils";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { createMemoryHistory, createRouter } from "vue-router";
import SettingsSpeechView from "./SettingsSpeechView.vue";
import { speechSettings, type SpeechProfile } from "../api/speech";

vi.mock("../api/speech", () => ({
  speechSettings: {
    get: vi.fn(),
    putProfile: vi.fn(),
    removeProfile: vi.fn(),
    makePrimary: vi.fn(),
    testProfile: vi.fn(),
  },
}));

const profile = (id: string, extra: Partial<SpeechProfile> = {}): SpeechProfile => ({
  id,
  base_url: `http://127.0.0.1:${id === "main" ? 8178 : 8179}/v1`,
  model: "",
  language: "",
  disabled: false,
  ...extra,
});

async function mountView(profiles: SpeechProfile[]) {
  vi.mocked(speechSettings.get).mockResolvedValue({ profiles });
  const router = createRouter({
    history: createMemoryHistory(),
    routes: [
      { path: "/", name: "home", component: { template: "<div/>" } },
      { path: "/settings/models", name: "settings-models", component: { template: "<div/>" } },
      { path: "/settings/speech", name: "settings-speech", component: SettingsSpeechView },
    ],
  });
  await router.push("/settings/speech");
  const wrapper = mount(SettingsSpeechView, { global: { plugins: [router] } });
  await flushPromises();
  return wrapper;
}

const confirm = vi.fn((_question?: string) => true);

beforeEach(() => {
  for (const call of Object.values(speechSettings)) vi.mocked(call).mockReset();
  confirm.mockReset().mockReturnValue(true);
  Object.defineProperty(window, "confirm", { value: confirm, configurable: true });
});

it("asks before removing a profile and keeps it when cancelled", async () => {
  const wrapper = await mountView([profile("main")]);
  confirm.mockReturnValue(false);
  await wrapper.get('[data-profile="main"] [data-testid="speech-remove"]').trigger("click");
  await flushPromises();
  expect(confirm).toHaveBeenCalledWith("Удалить профиль распознавания «main»?");
  expect(speechSettings.removeProfile).not.toHaveBeenCalled();
});

afterEach(() => {
  vi.restoreAllMocks();
});

it("shows the order of profiles: main, spare and switched off", async () => {
  const wrapper = await mountView([
    profile("main", { language: "ru" }),
    profile("spare"),
    profile("old", { disabled: true }),
  ]);
  const text = (id: string) => wrapper.get(`[data-profile="${id}"]`).text();
  expect(text("main")).toContain("Основной · Русский");
  expect(text("spare")).toContain("Запасной 1 · Определять автоматически");
  expect(text("old")).toContain("Отключён");
  expect(wrapper.find('[data-profile="main"] [data-testid="speech-primary"]').exists()).toBe(false);
  expect(wrapper.find('[data-profile="old"] [data-testid="speech-primary"]').exists()).toBe(false);
});

it("explains the empty state", async () => {
  const wrapper = await mountView([]);
  expect(wrapper.get('[data-testid="speech-empty"]').text()).toContain("микрофон");
});

it("accepts only a server on this computer and saves a new profile", async () => {
  const wrapper = await mountView([]);
  vi.mocked(speechSettings.putProfile).mockResolvedValue({ profiles: [profile("main")] });
  await wrapper.get('[data-testid="speech-add-open"]').trigger("click");
  await wrapper.get('[data-testid="speech-id"]').setValue("main");
  await wrapper.get('[data-testid="speech-url"]').setValue("https://speech.example.com/v1");
  expect(wrapper.get('[data-testid="speech-form-error"]').text()).toContain("этот компьютер");
  expect(wrapper.get('[data-testid="speech-save"]').attributes("disabled")).toBeDefined();

  await wrapper.get('[data-testid="speech-url"]').setValue("http://localhost:8178/v1");
  await wrapper.get('[data-testid="speech-language"]').setValue("ru");
  await wrapper.get('[data-testid="speech-form"]').trigger("submit");
  await flushPromises();

  expect(speechSettings.putProfile).toHaveBeenCalledWith("main", {
    base_url: "http://localhost:8178/v1",
    language: "ru",
    model: "",
  });
  expect(wrapper.find('[data-testid="speech-form"]').exists()).toBe(false);
  expect(wrapper.find('[data-profile="main"]').exists()).toBe(true);
});

it("checks, reorders, switches off and removes profiles", async () => {
  const wrapper = await mountView([profile("main"), profile("spare")]);
  vi.mocked(speechSettings.testProfile).mockResolvedValue({ ok: false, status: null, seconds: null });
  await wrapper.get('[data-profile="main"] [data-testid="speech-test"]').trigger("click");
  await flushPromises();
  expect(wrapper.get('[data-profile="main"] [data-testid="speech-check"]').text()).toBe("Сервер недоступен");

  vi.mocked(speechSettings.makePrimary).mockResolvedValue({ profiles: [profile("spare"), profile("main")] });
  await wrapper.get('[data-profile="spare"] [data-testid="speech-primary"]').trigger("click");
  await flushPromises();
  expect(speechSettings.makePrimary).toHaveBeenCalledWith("spare");
  expect(wrapper.get('[data-profile="spare"]').text()).toContain("Основной");

  vi.mocked(speechSettings.putProfile).mockResolvedValue({
    profiles: [profile("spare"), profile("main", { disabled: true })],
  });
  await wrapper.get('[data-profile="main"] [data-testid="speech-toggle"]').trigger("click");
  await flushPromises();
  expect(speechSettings.putProfile).toHaveBeenCalledWith("main", { disabled: true });

  vi.mocked(speechSettings.removeProfile).mockResolvedValue({ profiles: [profile("spare")] });
  await wrapper.get('[data-profile="main"] [data-testid="speech-remove"]').trigger("click");
  await flushPromises();
  expect(speechSettings.removeProfile).toHaveBeenCalledWith("main");
  expect(wrapper.find('[data-profile="main"]').exists()).toBe(false);
});

it("shows a save error in plain words", async () => {
  const wrapper = await mountView([profile("main"), profile("spare")]);
  vi.mocked(speechSettings.makePrimary).mockRejectedValue(new Error("HTTP 500"));
  await wrapper.get('[data-profile="spare"] [data-testid="speech-primary"]').trigger("click");
  await flushPromises();
  expect(wrapper.get('[data-testid="speech-error"]').text()).toBe("Не удалось сохранить: HTTP 500");
});
