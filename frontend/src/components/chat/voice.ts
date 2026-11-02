// Voice input for the composer: record from the microphone, get the text from the
// server and give it back for editing. Nothing is sent to the agent automatically.

import { onBeforeUnmount, ref } from "vue";

export type VoiceState = "idle" | "requesting" | "recording" | "transcribing";

/** A longer recording stops by itself and goes to recognition. */
export const MAX_RECORDING_SECONDS = 300;

const AUDIO_TYPES = ["audio/webm;codecs=opus", "audio/webm", "audio/ogg;codecs=opus", "audio/mp4"];

/** The first container type the browser can record, or null to let it choose. */
export function pickAudioType(isSupported: (type: string) => boolean): string | null {
  return AUDIO_TYPES.find((type) => isSupported(type)) ?? null;
}

/** A plain message for a microphone failure. */
export function microphoneError(error: unknown): string {
  const name = error instanceof Error || error instanceof DOMException ? error.name : "";
  if (name === "NotAllowedError" || name === "SecurityError") {
    return "Нет доступа к микрофону. Разрешите его в настройках браузера и попробуйте снова.";
  }
  if (name === "NotFoundError" || name === "OverconstrainedError") return "Микрофон не найден.";
  if (name === "NotReadableError") return "Микрофон занят другим приложением.";
  return "Не удалось включить микрофон.";
}

/** Put text at the cursor with one space between it and the neighbouring words. */
export function insertText(
  current: string,
  start: number,
  end: number,
  piece: string,
): { text: string; caret: number } {
  const before = current.slice(0, start);
  const after = current.slice(end);
  const lead = before && !/\s$/.test(before) ? " " : "";
  const tail = after && !/^\s/.test(after) ? " " : "";
  const text = `${before}${lead}${piece}${tail}${after}`;
  return { text, caret: before.length + lead.length + piece.length };
}

export function useVoiceInput(options: {
  transcribe: (audio: Blob) => Promise<string>;
  onText: (text: string) => void;
}) {
  const state = ref<VoiceState>("idle");
  const seconds = ref(0);
  const error = ref("");
  let recorder: MediaRecorder | null = null;
  let stream: MediaStream | null = null;
  let timer: ReturnType<typeof setInterval> | null = null;
  let chunks: Blob[] = [];
  // A new start or a cancel makes the answer of an older recording stale.
  let attempt = 0;

  function release(): void {
    if (timer !== null) clearInterval(timer);
    timer = null;
    stream?.getTracks().forEach((track) => track.stop());
    stream = null;
    recorder = null;
  }

  async function start(): Promise<void> {
    if (state.value !== "idle") return;
    error.value = "";
    if (typeof MediaRecorder === "undefined" || !navigator.mediaDevices?.getUserMedia) {
      error.value = "Браузер не поддерживает запись звука на этой странице.";
      return;
    }
    const current = ++attempt;
    state.value = "requesting";
    try {
      const media = await navigator.mediaDevices.getUserMedia({ audio: true });
      if (current !== attempt) {
        media.getTracks().forEach((track) => track.stop());
        return;
      }
      stream = media;
      const type = pickAudioType((value) => MediaRecorder.isTypeSupported(value));
      recorder = new MediaRecorder(media, type ? { mimeType: type } : undefined);
    } catch (cause) {
      release();
      if (current === attempt) {
        state.value = "idle";
        error.value = microphoneError(cause);
      }
      return;
    }
    chunks = [];
    recorder.ondataavailable = (event) => {
      if (event.data.size > 0) chunks.push(event.data);
    };
    recorder.start();
    seconds.value = 0;
    state.value = "recording";
    timer = setInterval(() => {
      seconds.value += 1;
      if (seconds.value >= MAX_RECORDING_SECONDS) void finish();
    }, 1000);
  }

  function cancel(): void {
    attempt += 1;
    if (recorder && recorder.state !== "inactive") {
      recorder.ondataavailable = null;
      recorder.onstop = null;
      recorder.stop();
    }
    release();
    chunks = [];
    state.value = "idle";
  }

  function finish(): Promise<void> {
    const active = recorder;
    if (state.value !== "recording" || !active) return Promise.resolve();
    const current = attempt;
    state.value = "transcribing";
    if (timer !== null) clearInterval(timer);
    timer = null;
    return new Promise((resolve) => {
      active.onstop = async () => {
        const audio = new Blob(chunks, { type: active.mimeType || "audio/webm" });
        chunks = [];
        release();
        try {
          const text = (await options.transcribe(audio)).trim();
          if (current !== attempt) return;
          if (text) options.onText(text);
          else error.value = "Речь не распознана. Попробуйте ещё раз.";
        } catch (cause) {
          if (current === attempt) {
            error.value = cause instanceof Error ? cause.message : "Не удалось распознать речь.";
          }
        } finally {
          if (current === attempt) state.value = "idle";
          resolve();
        }
      };
      active.stop();
    });
  }

  onBeforeUnmount(cancel);

  return { state, seconds, error, start, finish, cancel };
}
