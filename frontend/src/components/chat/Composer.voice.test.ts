import { flushPromises, mount } from "@vue/test-utils";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import Composer from "./Composer.vue";
import { transcribe } from "../../api/speech";

vi.mock("../../api/speech", () => ({ transcribe: vi.fn() }));

class FakeRecorder {
  static isTypeSupported = () => true;
  state = "inactive";
  mimeType = "audio/webm";
  ondataavailable: ((event: { data: Blob }) => void) | null = null;
  onstop: (() => void) | null = null;
  start(): void {
    this.state = "recording";
  }
  stop(): void {
    this.state = "inactive";
    this.ondataavailable?.({ data: new Blob(["sound"]) });
    this.onstop?.();
  }
}

function mountComposer(props: Record<string, unknown> = {}) {
  return mount(Composer, { props: { draftKey: "", ...props }, attachTo: document.body });
}

beforeEach(() => {
  vi.mocked(transcribe).mockReset();
  vi.stubGlobal("MediaRecorder", FakeRecorder);
  Object.defineProperty(navigator, "mediaDevices", {
    value: { getUserMedia: vi.fn().mockResolvedValue({ getTracks: () => [{ stop: vi.fn() }] }) },
    configurable: true,
  });
});

afterEach(() => {
  vi.unstubAllGlobals();
  document.body.innerHTML = "";
});

it("puts recognized speech at the cursor and does not send it", async () => {
  vi.mocked(transcribe).mockResolvedValue({ text: "быстро", profile: "local" });
  const wrapper = mountComposer();
  const input = wrapper.get<HTMLTextAreaElement>('[data-testid="composer-input"]');
  await input.setValue("сделай коммит");
  input.element.setSelectionRange(6, 6);

  await wrapper.get('[data-testid="voice-start"]').trigger("click");
  await flushPromises();
  expect(wrapper.get('[data-testid="voice-status"]').text()).toContain("Запись 0:00");
  expect(wrapper.find('[data-testid="composer-send"]').exists()).toBe(false);

  await wrapper.get('[data-testid="voice-finish"]').trigger("click");
  await flushPromises();

  expect(input.element.value).toBe("сделай быстро коммит");
  expect(wrapper.emitted("send")).toBeUndefined();
  expect(wrapper.find('[data-testid="composer-send"]').exists()).toBe(true);
  wrapper.unmount();
});

it("shows a recognition error and lets the user hide it", async () => {
  vi.mocked(transcribe).mockRejectedValue(new Error("Распознавание речи не настроено"));
  const wrapper = mountComposer();
  await wrapper.get('[data-testid="voice-start"]').trigger("click");
  await flushPromises();
  await wrapper.get('[data-testid="voice-finish"]').trigger("click");
  await flushPromises();
  const error = wrapper.get('[data-testid="voice-error"]');
  expect(error.text()).toContain("Распознавание речи не настроено");
  await error.get("button").trigger("click");
  expect(wrapper.find('[data-testid="voice-error"]').exists()).toBe(false);
  wrapper.unmount();
});

it("does not record while the agent is busy", () => {
  const wrapper = mountComposer({ busy: true });
  expect(wrapper.get('[data-testid="voice-start"]').attributes("disabled")).toBeDefined();
  wrapper.unmount();
});
