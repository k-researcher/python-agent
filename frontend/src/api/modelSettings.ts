// Model settings API: providers, models and routing. The file gives the base layer;
// changes made here are stored in the database on top of it.

import { request } from "./client";

export type Origin = "file" | "ui" | "legacy";

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
const put = <T>(url: string, body: unknown) => request<T>(url, { method: "PUT", body: JSON.stringify(body) });

export const modelSettings = {
  get: () => request<ModelsView>(PREFIX),
  putProvider: (id: string, patch: ProviderPatch) =>
    put<ModelsView>(`${PREFIX}/providers/${encodeURIComponent(id)}`, patch),
  putModel: (id: string, patch: ModelPatch) => put<ModelsView>(`${PREFIX}/models/${modelPath(id)}`, patch),
  putRouting: (patch: Partial<RoutingView>) => put<ModelsView>(`${PREFIX}/routing`, patch),
  removeProvider: (id: string) =>
    request<ModelsView>(`${PREFIX}/providers/${encodeURIComponent(id)}`, { method: "DELETE" }),
  removeModel: (id: string) => request<ModelsView>(`${PREFIX}/models/${modelPath(id)}`, { method: "DELETE" }),
  testProvider: (id: string) =>
    request<ConnectionResult>(`${PREFIX}/providers/${encodeURIComponent(id)}/test`, { method: "POST" }),
  probeFound: (providerId: string, model: string) =>
    request<ProbeResult>(`${PREFIX}/providers/${encodeURIComponent(providerId)}/probe`, {
      method: "POST",
      body: JSON.stringify({ model }),
    }),
  probeModel: (id: string) => request<ProbeResult>(`${PREFIX}/models/${modelPath(id)}/probe`, { method: "POST" }),
  scanLocal: () => request<LocalServer[]>(`${PREFIX}/local-servers`, { method: "POST" }),
};
