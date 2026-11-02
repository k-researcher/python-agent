// Model settings API: providers, models and routing. The file gives the base layer;
// changes made here are stored in the database on top of it.

import { request } from "./client";

export type Origin = "file" | "ui" | "legacy";
export type KeySource = "ui" | "env" | "none";

export interface ProviderView {
  id: string;
  kind: string;
  base_url: string;
  api_key_env: string | null;
  api_key_set: boolean;
  api_key_hint: string;
  configured: boolean;
  disabled: boolean;
  origin: Origin;
  has_base?: boolean;
  key_source?: KeySource;
  key_env?: string | null;
}

export interface ModelView {
  id: string;
  provider: string;
  model: string;
  context_window: number;
  max_tokens: number;
  pricing: { input_per_mtok: number; output_per_mtok: number } | null;
  disabled: boolean;
  origin: Origin;
  has_base?: boolean;
  provider_disabled?: boolean;
}

export interface RoutingView {
  default: string;
  fallback: string[];
  roles: Record<string, string>;
  light: string | null;
}

export interface ModelsView {
  providers: ProviderView[];
  models: ModelView[];
  routing: RoutingView;
  warnings: string[];
  checksum?: string;
  deleted?: { providers: string[]; models: string[] };
}

export interface DiscoveredModel {
  id: string;
  context_window: number | null;
  max_output_tokens?: number | null;
  kind: string | null;
  /** A local server says whether the model is loaded into memory. */
  state?: string | null;
}

export interface ProbeResult {
  ok: boolean;
  status: number | null;
  seconds: number | null;
  message?: string;
}

export interface LocalServer {
  base_url: string;
  server: "lmstudio" | "ollama" | "openai";
  models: number;
  provider: string | null;
}

export interface ConnectionResult {
  ok: boolean;
  status: number | null;
  models: string[];
  details?: DiscoveredModel[];
  message?: string;
}

export interface ProviderPatch {
  kind?: "openai";
  base_url?: string;
  api_key?: string;
  api_key_env?: string | null;
  key_source?: KeySource;
  disabled?: boolean;
}

export interface ModelPatch {
  provider?: string;
  model?: string;
  context_window?: number;
  max_tokens?: number;
  pricing?: { input_per_mtok: number; output_per_mtok: number } | null;
  disabled?: boolean;
}

const PREFIX = "/api/settings/models";

// A model ID may contain "/" ("publisher/model"); the server route keeps it as a path.
const modelPath = (id: string) => id.split("/").map(encodeURIComponent).join("/");
export interface SaveOptions { checksum?: string; create?: boolean }

const headers = (options?: SaveOptions): Record<string, string> => ({
  ...(options?.checksum ? { "If-Match": options.checksum } : {}),
  ...(options?.create ? { "If-None-Match": "*" } : {}),
});
const put = <T>(url: string, body: unknown, options?: SaveOptions) =>
  request<T>(url, { method: "PUT", body: JSON.stringify(body), headers: headers(options) });

export const modelSettings = {
  get: () => request<ModelsView>(PREFIX),
  putProvider: (id: string, patch: ProviderPatch, options?: SaveOptions) =>
    put<ModelsView>(`${PREFIX}/providers/${modelPath(id)}`, patch, options),
  putModel: (id: string, patch: ModelPatch, options?: SaveOptions) => put<ModelsView>(`${PREFIX}/models/${modelPath(id)}`, patch, options),
  putRouting: (patch: Partial<RoutingView>, options?: SaveOptions) => put<ModelsView>(`${PREFIX}/routing`, patch, options),
  removeProvider: (id: string, options?: SaveOptions) =>
    request<ModelsView>(`${PREFIX}/providers/${modelPath(id)}`, { method: "DELETE", headers: headers(options) }),
  removeModel: (id: string, options?: SaveOptions) => request<ModelsView>(`${PREFIX}/models/${modelPath(id)}`, { method: "DELETE", headers: headers(options) }),
  resetProvider: (id: string, options?: SaveOptions) =>
    request<ModelsView>(`${PREFIX}/providers/${modelPath(id)}/reset`, { method: "POST", headers: headers(options) }),
  resetModel: (id: string, options?: SaveOptions) =>
    request<ModelsView>(`${PREFIX}/models/${modelPath(id)}/reset`, { method: "POST", headers: headers(options) }),
  testProvider: (id: string) =>
    request<ConnectionResult>(`${PREFIX}/providers/${modelPath(id)}/test`, { method: "POST" }),
  probeFound: (providerId: string, model: string) =>
    request<ProbeResult>(`${PREFIX}/providers/${modelPath(providerId)}/probe`, {
      method: "POST",
      body: JSON.stringify({ model }),
    }),
  probeModel: (id: string) => request<ProbeResult>(`${PREFIX}/models/${modelPath(id)}/probe`, { method: "POST" }),
  scanLocal: () => request<LocalServer[]>(`${PREFIX}/local-servers`, { method: "POST" }),
};
