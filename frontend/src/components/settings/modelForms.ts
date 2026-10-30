// Pure helpers for the model settings screen: provider presets, the suggested settings
// of a found model and plain labels.

import type { DiscoveredModel, ModelPatch, ModelView, ProviderView } from "../../api/modelSettings";

export interface ProviderPreset {
  key: string;
  label: string;
  id: string;
  baseUrl: string;
  needsKey: boolean;
}

export const PROVIDER_PRESETS: ProviderPreset[] = [
  { key: "lmstudio", label: "LM Studio", id: "lmstudio", baseUrl: "http://127.0.0.1:1234/v1", needsKey: false },
  { key: "ollama", label: "Ollama", id: "ollama", baseUrl: "http://127.0.0.1:11434/v1", needsKey: false },
  { key: "other", label: "Другой OpenAI-совместимый сервер", id: "", baseUrl: "https://", needsKey: true },
];

/** The smallest context window the server accepts. */
export const MIN_CONTEXT = 4096;
/** Used when the server does not report the context length of a model. */
export const DEFAULT_CONTEXT = 32768;

const ID_PATTERN = /^[A-Za-z0-9._:/@-]{1,100}$/;
const LOCAL_HOSTS = new Set(["localhost", "127.0.0.1", "[::1]", "host.docker.internal"]);

export function isValidId(id: string): boolean {
  return ID_PATTERN.test(id);
}

export function isLocalUrl(url: string): boolean {
  try {
    return LOCAL_HOSTS.has(new URL(url).hostname);
  } catch {
    return false;
  }
}

/** Embedding models cannot answer in a chat, so they are not offered as agent models. */
export function canChat(found: DiscoveredModel): boolean {
  return found.kind !== "embeddings" && found.kind !== "embedding";
}

/** The ID of a model already added for this provider and server model, if any. */
export function addedAs(models: ModelView[], provider: string, model: string): string | null {
  return models.find((item) => item.provider === provider && item.model === model)?.id ?? null;
}

/** Settings for a found model: its own ID, context length and a free price for a local server. */
export function suggestModel(
  provider: ProviderView,
  found: DiscoveredModel,
  models: ModelView[],
): { id: string; patch: ModelPatch } {
  const taken = new Set(models.map((item) => item.id));
  const base = found.id.replace(/[^A-Za-z0-9._:/@-]/g, "-").slice(0, 100) || "model";
  const id = taken.has(base) ? `${provider.id}:${base}`.slice(0, 100) : base;
  const context = Math.max(MIN_CONTEXT, found.context_window ?? DEFAULT_CONTEXT);
  return {
    id,
    patch: {
      provider: provider.id,
      model: found.id,
      context_window: context,
      max_tokens: Math.min(8192, Math.floor(context / 4)),
      // A local server costs nothing; without a price a session with a money limit stops.
      pricing: isLocalUrl(provider.base_url) ? { input_per_mtok: 0, output_per_mtok: 0 } : null,
    },
  };
}

export function keyStatus(provider: ProviderView): string {
  if (provider.api_key_set) return `Ключ задан ${provider.api_key_hint}`.trim();
  if (provider.api_key_env) return `Нет ключа: задайте ${provider.api_key_env}`;
  return "Ключ не нужен";
}

export function originLabel(origin: string): string {
  return origin === "ui" ? "Добавлено здесь" : "Из файла";
}

export function formatTokens(value: number | null): string {
  if (value === null) return "—";
  return value >= 1024 && value % 1024 === 0 ? `${value / 1024}K` : value.toLocaleString("ru-RU");
}
