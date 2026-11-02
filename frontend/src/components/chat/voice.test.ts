import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { defineComponent, h } from "vue";
import { flushPromises, mount } from "@vue/test-utils";
import { MAX_RECORDING_SECONDS, insertText, microphoneError, pickAudioType, useVoiceInput } from "./voice";

class FakeRecorder {
  static instances: FakeRecorder[] = [];
  static isTypeSupported = (type: string) => type === "audio/webm";
  state: "inactive" | "recording" = "inactive";
  mimeType: string;
  ondataavailable: ((event: { data: Blob }) => void) | null = null;
  onstop: (() => void) | null = null;
  constructor(_stream: unknown, options?: { mimeType?: string }) {
    this.mimeType = options?.mimeType ?? "";
    FakeRecorder.instances.push(this);
  }
  start(): void {
    this.state = "recording";
  }
  stop(): void {
    this.state = "inactive";
    this.ondataavailable?.({ data: new Blob(["sound"], { type: this.mimeType }) });
    this.onstop?.();
  }
}

const stopTrack = vi.fn();
const getUserMedia = vi.fn();

function harness(transcribe: (audio: Blob) => Promise<string>) {
  const texts: string[] = [];
  let api!: ReturnType<typeof useVoiceInput>;
  const wrapper = mount(
    defineComponent({
      setup() {
        api = useVoiceInput({ transcribe, onText: (text) => texts.push(text) });
        return () => h("div");
      },
    }),
  );
  return { api, texts, wrapper };
}

beforeEach(() => {
  FakeRecorder.instances = [];
  stopTrack.mockReset();
  getUserMedia.mockReset().mockResolvedValue({ getTracks: () => [{ stop: stopTrack }] });
  vi.stubGlobal("MediaRecorder", FakeRecorder);
  Object.defineProperty(navigator, "mediaDevices", { value: { getUserMedia }, configurable: true });
});

afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
});

describe("helpers", () => {
  it("picks the first supported container type", () => {
    expect(pickAudioType((type) => type.startsWith("audio/ogg"))).toBe("audio/ogg;codecs=opus");
    expect(pickAudioType(() => false)).toBeNull();
  });

  it("explains microphone failures in plain words", () => {
    expect(microphoneError(new DOMException("x", "NotAllowedError"))).toContain("Нет доступа к микрофону");
    expect(microphoneError(new DOMException("x", "NotFoundError"))).toBe("Микрофон не найден.");
    expect(microphoneError(new DOMException("x", "NotReadableError"))).toContain("занят");
    expect(microphoneError("odd")).toBe("Не удалось включить микрофон.");
  });

  it("inserts text at the cursor with single spaces", () => {
    expect(insertText("", 0, 0, "привет")).toEqual({ text: "привет", caret: 6 });
    expect(insertText("сделай коммит", 6, 6, "быстро")).toEqual({ text: "сделай быстро коммит", caret: 13 });
    expect(insertText("начало ", 7, 7, "конец")).toEqual({ text: "начало конец", caret: 12 });
    expect(insertText("заменить это", 9, 12, "то")).toEqual({ text: "заменить то", caret: 11 });
  });
});

describe("useVoiceInput", () => {
  it("records, sends the audio and gives back the text without sending it anywhere else", async () => {
    const transcribe = vi.fn().mockResolvedValue("  готовый текст  ");
    const { api, texts } = harness(transcribe);

    await api.start();
    expect(api.state.value).toBe("recording");
    expect(FakeRecorder.instances[0].mimeType).toBe("audio/webm");

    await api.finish();

    const audio = transcribe.mock.calls[0][0] as Blob;
    expect(audio.type).toBe("audio/webm");
    expect(await audio.text()).toBe("sound");
    expect(texts).toEqual(["готовый текст"]);
    expect(api.state.value).toBe("idle");
    expect(stopTrack).toHaveBeenCalled();
  });

  it("cancel throws the recording away", async () => {
    const transcribe = vi.fn();
    const { api, texts } = harness(transcribe);
    await api.start();
    api.cancel();
    expect(api.state.value).toBe("idle");
    expect(stopTrack).toHaveBeenCalled();
    expect(transcribe).not.toHaveBeenCalled();
    expect(texts).toEqual([]);
  });

  it("ignores a late answer after cancel during recognition", async () => {
    let answer!: (text: string) => void;
    const { api, texts } = harness(() => new Promise((resolve) => (answer = resolve)));
    await api.start();
    const done = api.finish();
    expect(api.state.value).toBe("transcribing");
    api.cancel();
    answer("поздно");
    await done;
    expect(texts).toEqual([]);
    expect(api.state.value).toBe("idle");
    expect(api.error.value).toBe("");
  });

  it("shows why the microphone did not start", async () => {
    getUserMedia.mockRejectedValue(new DOMException("denied", "NotAllowedError"));
    const { api } = harness(vi.fn());
    await api.start();
    expect(api.state.value).toBe("idle");
    expect(api.error.value).toContain("Нет доступа к микрофону");
  });

  it("says when the browser cannot record", async () => {
    vi.stubGlobal("MediaRecorder", undefined);
    const { api } = harness(vi.fn());
    await api.start();
    expect(api.error.value).toContain("не поддерживает запись");
    expect(getUserMedia).not.toHaveBeenCalled();
  });

  it("reports empty speech and server errors", async () => {
    const empty = harness(vi.fn().mockResolvedValue("   "));
    await empty.api.start();
    await empty.api.finish();
    expect(empty.api.error.value).toContain("Речь не распознана");
    expect(empty.texts).toEqual([]);

    const failing = harness(vi.fn().mockRejectedValue(new Error("Серверы распознавания не отвечают")));
    await failing.api.start();
    await failing.api.finish();
    expect(failing.api.error.value).toBe("Серверы распознавания не отвечают");
  });

  it("stops a long recording by itself and sends it", async () => {
    vi.useFakeTimers();
    const transcribe = vi.fn().mockResolvedValue("длинная речь");
    const { api, texts } = harness(transcribe);
    await api.start();
    vi.advanceTimersByTime(MAX_RECORDING_SECONDS * 1000);
    await flushPromises();
    expect(transcribe).toHaveBeenCalledTimes(1);
    expect(texts).toEqual(["длинная речь"]);
  });

  it("stops the microphone when the view closes", async () => {
    const { api, wrapper } = harness(vi.fn());
    await api.start();
    wrapper.unmount();
    expect(stopTrack).toHaveBeenCalled();
  });
});
