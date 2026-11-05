import { flushPromises, mount } from "@vue/test-utils";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import Composer from "./Composer.vue";
import { MAX_FILE_BYTES, firstCodePoints, formatSize } from "./attachments";
import { toChatMessage } from "../../uiAdapters";
import MessageItem from "./MessageItem.vue";
import type { Message } from "../../api/types";

beforeEach(() => {
  const store = new Map<string, string>();
  Object.defineProperty(window, "localStorage", {
    value: {
      getItem: (key: string) => store.get(key) ?? null,
      setItem: (key: string, value: string) => void store.set(key, value),
      removeItem: (key: string) => void store.delete(key),
    },
    configurable: true,
  });
});

afterEach(() => {
  Object.defineProperty(window, "localStorage", { value: undefined, configurable: true });
});

function mountComposer() {
  return mount(Composer, { props: { draftKey: "draft:files", efforts: ["auto"] } });
}

function textFile(name: string, text: string): File {
  return new File([text], name, { type: "text/plain" });
}

async function choose(wrapper: ReturnType<typeof mountComposer>, files: File[]): Promise<void> {
  const input = wrapper.get('[data-testid="composer-file-input"]');
  Object.defineProperty(input.element, "files", { value: files, configurable: true });
  await input.trigger("change");
  await flushPromises();
}

function chips(wrapper: ReturnType<typeof mountComposer>): string[] {
  return wrapper.findAll('[data-testid="composer-files"] .file').map((item) => item.text());
}

describe("Composer attachments", () => {
  it("sends chosen text files with the text, also without text, and clears them", async () => {
    const wrapper = mountComposer();
    const send = wrapper.get('[data-testid="composer-send"]');
    expect(send.attributes("disabled")).toBeDefined();

    await choose(wrapper, [textFile("notes.md", "# Заметки"), textFile("run.py", 'print("ok")')]);
    expect(chips(wrapper)).toEqual(["notes.md16 Б", "run.py11 Б"]);
    expect(send.attributes("disabled")).toBeUndefined();

    await wrapper.get('[aria-label="Убрать файл run.py"]').trigger("click");
    expect(chips(wrapper)).toEqual(["notes.md16 Б"]);

    await send.trigger("click");
    expect(wrapper.emitted("send")?.[0]).toEqual([
      { text: "", effort: "auto", files: [{ name: "notes.md", text: "# Заметки", size: 16 }] },
    ]);
    expect(wrapper.find('[data-testid="composer-files"]').exists()).toBe(false);
  });

  it("refuses binary, non UTF-8, too large and extra files with a message", async () => {
    const wrapper = mountComposer();
    await choose(wrapper, [
      new File([new Uint8Array([0x89, 0x50, 0x00, 0x47])], "image.png"),
      new File([new Uint8Array([0xff, 0xfe, 0x41])], "latin1.txt"),
      textFile("big.txt", "x".repeat(MAX_FILE_BYTES + 1)),
    ]);
    expect(chips(wrapper)).toEqual([]);
    const error = wrapper.get('[data-testid="composer-file-error"]').text();
    expect(error).toContain("image.png не текстовый");
    expect(error).toContain("latin1.txt не в кодировке UTF-8");
    expect(error).toContain("big.txt больше 200 КБ");

    await choose(wrapper, Array.from({ length: 11 }, (_, index) => textFile(`${index}.txt`, "x")));
    expect(chips(wrapper)).toHaveLength(10);
    expect(wrapper.get('[data-testid="composer-file-error"]').text()).toContain("не больше 10");
  });

  it("takes files from paste and drop and puts them back after a failed send", async () => {
    const wrapper = mountComposer();
    const pasted = new Event("paste", { cancelable: true });
    Object.defineProperty(pasted, "clipboardData", { value: { files: [textFile("a.txt", "a")] } });
    wrapper.get('[data-testid="composer-input"]').element.dispatchEvent(pasted);
    await flushPromises();
    expect(pasted.defaultPrevented).toBe(true);

    const dropped = new Event("drop", { cancelable: true });
    Object.defineProperty(dropped, "dataTransfer", { value: { files: [textFile("b.txt", "bb")] } });
    wrapper.get('[data-testid="composer"]').element.dispatchEvent(dropped);
    await flushPromises();
    expect(chips(wrapper)).toEqual(["a.txt1 Б", "b.txt2 Б"]);

    await wrapper.get('[data-testid="composer-send"]').trigger("click");
    expect(chips(wrapper)).toEqual([]);
    const sent = (wrapper.emitted("send")?.[0]?.[0] as { files: unknown[] }).files;
    (wrapper.vm as unknown as { restoreFiles: (files: unknown[]) => void }).restoreFiles(sent);
    await flushPromises();
    expect(chips(wrapper)).toEqual(["a.txt1 Б", "b.txt2 Б"]);
  });
});

describe("attached files in the timeline", () => {
  const message: Message = {
    id: 7,
    session_id: "s",
    role: "user",
    content: 'Смотри 😀 файл\n\n<attachment name="notes.md">\n# Заметки\n</attachment>',
    tool_call_id: null,
    tool_calls: null,
    token_count: null,
    skipped: false,
    kind: "normal",
    reasoning_content: null,
    reasoning_effort: null,
    attachments: { text_end: 13, files: [{ name: "notes.md", size: 16, chars: 9 }] },
    created_at: "2026-11-05T10:00:00Z",
  };

  it("shows only the typed text and the file chips", () => {
    const chat = toChatMessage(message);
    expect(chat.content).toBe("Смотри 😀 файл");
    expect(chat.attachments).toEqual([{ name: "notes.md", size: 16 }]);
    const wrapper = mount(MessageItem, { props: { message: chat } });
    expect(wrapper.text()).not.toContain("<attachment");
    expect(wrapper.get('[data-testid="message-files"]').text()).toContain("notes.md16 Б");
    expect(toChatMessage({ ...message, attachments: null }).content).toBe(message.content);
  });

  it("counts characters as the server does and formats sizes", () => {
    expect(firstCodePoints("a😀b", 2)).toBe("a😀");
    expect(firstCodePoints("ab", 5)).toBe("ab");
    expect(formatSize(900)).toBe("900 Б");
    expect(formatSize(200 * 1024)).toBe("200 КБ");
  });
});
