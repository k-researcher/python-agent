import { describe, expect, it } from "vitest";
import type { ModelView, ProviderView } from "../../api/modelSettings";
import { addedAs, canChat, formatTokens, isLocalUrl, isValidId, keyStatus, suggestModel } from "./modelForms";

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
    const { id, patch } = suggestModel(provider(), { id: "qwen/qwen3.6-35b-a3b", context_window: 262144, kind: "llm" }, []);
    expect(id).toBe("qwen/qwen3.6-35b-a3b");
    expect(patch).toEqual({
      provider: "lmstudio",
      model: "qwen/qwen3.6-35b-a3b",
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

describe("helpers", () => {
  it("recognises local addresses only", () => {
    expect(isLocalUrl("http://127.0.0.1:1234/v1")).toBe(true);
    expect(isLocalUrl("http://localhost:11434/v1")).toBe(true);
    expect(isLocalUrl("http://[::1]:8080/v1")).toBe(true);
    expect(isLocalUrl("https://llm.example.com/v1")).toBe(false);
    expect(isLocalUrl("not a url")).toBe(false);
  });

  it("does not offer embedding models for chat", () => {
    expect(canChat({ id: "e", context_window: 2048, kind: "embeddings" })).toBe(false);
    expect(canChat({ id: "c", context_window: null, kind: null })).toBe(true);
  });

  it("finds a model already added for the same provider and server model", () => {
    const models = [model({ id: "qwen", provider: "lmstudio", model: "qwen/q" })];
    expect(addedAs(models, "lmstudio", "qwen/q")).toBe("qwen");
    expect(addedAs(models, "ollama", "qwen/q")).toBeNull();
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
