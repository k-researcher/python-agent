import { describe, expect, it } from "vitest";
import type { ModelView, ProviderView } from "../../api/modelSettings";
import {
  addedAs,
  canChat,
  describeFound,
  formatTokens,
  isLocalUrl,
  isValidId,
  keyStatus,
  probeText,
  providerIdFor,
  rolesOf,
  suggestModel,
} from "./modelForms";

function provider(patch: Partial<ProviderView> = {}): ProviderView {
  return {
    id: "lmstudio",
    kind: "openai",
    base_url: "http://127.0.0.1:1234/v1",
    api_key_env: null,
    api_key_set: false,
    api_key_hint: "",
    configured: true,
    disabled: false,
    origin: "ui",
    ...patch,
  };
}

function model(patch: Partial<ModelView> = {}): ModelView {
  return {
    id: "main",
    provider: "gateway",
    model: "remote",
    context_window: 256000,
    max_tokens: 16000,
    pricing: null,
    disabled: false,
    origin: "file",
    ...patch,
  };
}

describe("suggestModel", () => {
  it("takes the context length from the server and a free price for a local server", () => {
    const { id, patch } = suggestModel(provider(), { id: "publisher/chat-model-35b", context_window: 262144, kind: "llm" }, []);
    expect(id).toBe("publisher/chat-model-35b");
    expect(patch).toEqual({
      provider: "lmstudio",
      model: "publisher/chat-model-35b",
      context_window: 262144,
      max_tokens: 8192,
      pricing: { input_per_mtok: 0, output_per_mtok: 0 },
    });
  });

  it("uses a safe default context and no price for a remote server", () => {
    const remote = provider({ id: "cloud", base_url: "https://llm.example.com/v1" });
    const { patch } = suggestModel(remote, { id: "m", context_window: null, kind: null }, []);
    expect(patch.context_window).toBe(32768);
    expect(patch.max_tokens).toBe(8192);
    expect(patch.pricing).toBeNull();
  });

  it("keeps the answer within the output limit the server reports", () => {
    const { patch } = suggestModel(provider(), { id: "c", context_window: 262144, max_output_tokens: 4096, kind: "chat" }, []);
    expect(patch.max_tokens).toBe(4096);
  });

  it("keeps the context at the server minimum and the answer below the context", () => {
    const { patch } = suggestModel(provider(), { id: "tiny", context_window: 2048, kind: "llm" }, []);
    expect(patch.context_window).toBe(4096);
    expect(patch.max_tokens).toBe(1024);
  });

  it("gives a free ID when the plain one is taken and replaces invalid characters", () => {
    const taken = [model({ id: "my model" }), model({ id: "my-model" })];
    expect(suggestModel(provider(), { id: "my model", context_window: null, kind: null }, taken).id).toBe("lmstudio:my-model");
    expect(isValidId(suggestModel(provider(), { id: "a b#c", context_window: null, kind: null }, []).id)).toBe(true);
  });
});

describe("local servers and checks", () => {
  it("gives a found server a free provider ID", () => {
    expect(providerIdFor("lmstudio", [])).toBe("lmstudio");
    expect(providerIdFor("lmstudio", [provider(), provider({ id: "lmstudio-2" })])).toBe("lmstudio-3");
    expect(providerIdFor("openai", [])).toBe("local");
  });

  it("states the check result in plain words", () => {
    expect(probeText({ ok: true, status: 200, seconds: 0.45 })).toBe("Отвечает за 0,45 с");
    expect(probeText({ ok: false, status: 500, seconds: 5 })).toBe("Не отвечает: HTTP 500");
    expect(probeText({ ok: false, status: null, seconds: 120, message: "The model did not answer within the time limit" })).toBe(
      "Не ответила вовремя",
    );
    expect(probeText({ ok: false, status: null, seconds: 0.1, message: "Cannot connect to provider" })).toBe(
      "Сервер недоступен",
    );
  });
});

describe("helpers", () => {
  it("recognises local addresses only", () => {
    expect(isLocalUrl("http://127.0.0.1:1234/v1")).toBe(true);
    expect(isLocalUrl("http://localhost:11434/v1")).toBe(true);
    expect(isLocalUrl("http://[::1]:8080/v1")).toBe(true);
    expect(isLocalUrl("https://llm.example.com/v1")).toBe(false);
    expect(isLocalUrl("not a url")).toBe(false);
  });

  it("offers only chat models for the agent", () => {
    expect(canChat({ id: "e", context_window: 2048, kind: "embeddings" })).toBe(false);
    expect(canChat({ id: "e", context_window: 8192, kind: "embedding" })).toBe(false);
    expect(canChat({ id: "i", context_window: null, kind: "image_generation" })).toBe(false);
    expect(canChat({ id: "c", context_window: null, kind: null })).toBe(true);
    expect(canChat({ id: "c", context_window: null, kind: "chat" })).toBe(true);
    expect(canChat({ id: "v", context_window: null, kind: "vlm" })).toBe(true);
  });

  it("describes what the server reports about a found model", () => {
    expect(describeFound({ id: "c", context_window: 262144, max_output_tokens: 65536, kind: "chat" })).toBe(
      "контекст 256K · ответ до 64K",
    );
    expect(describeFound({ id: "c", context_window: null, kind: null })).toBe("сервер не сообщает лимиты");
    expect(describeFound({ id: "e", context_window: 2048, kind: "embeddings" })).toBe("модель эмбеддингов");
    expect(describeFound({ id: "i", context_window: null, kind: "image_generation" })).toBe(
      "не для чата (image_generation)",
    );
  });

  it("lists the roles that use a model", () => {
    const routing = { default: "main", fallback: ["light"], roles: { review: "main" }, light: "light" };
    expect(rolesOf(routing, "main")).toEqual(["основная", "роль review"]);
    expect(rolesOf(routing, "light")).toEqual(["служебная", "запасная"]);
    expect(rolesOf(routing, "spare")).toEqual([]);
  });

  it("finds a model already added for the same provider and server model", () => {
    const models = [model({ id: "chat", provider: "lmstudio", model: "publisher/chat" })];
    expect(addedAs(models, "lmstudio", "publisher/chat")).toBe("chat");
    expect(addedAs(models, "ollama", "publisher/chat")).toBeNull();
  });

  it("describes the key without showing it", () => {
    expect(keyStatus(provider())).toBe("Ключ не нужен");
    expect(keyStatus(provider({ api_key_env: "API_KEY" }))).toBe("Нет ключа: задайте API_KEY");
    expect(keyStatus(provider({ api_key_set: true, api_key_hint: "…abcd" }))).toBe("Ключ задан …abcd");
  });

  it("shows round token counts in K", () => {
    expect(formatTokens(262144)).toBe("256K");
    expect(formatTokens(null)).toBe("—");
    expect(formatTokens(16000)).toBe((16000).toLocaleString("ru-RU"));
  });
});
