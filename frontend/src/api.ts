export interface Project {
  id: string;
  name: string;
  root_path: string;
}

export interface Session {
  id: string;
  project_id: string;
  parent_id: string | null;
  mode: string;
  llm_profile: string;
  title: string;
  status: string;
  error: string | null;
  archived: boolean;
  created_at: string;
  updated_at: string;
}

export interface OutboundAudit {
  id: number;
  category: string;
  destination: string;
  operation: string;
  payload_bytes: number;
  payload_sha256: string;
  status: string;
  detail: string | null;
  created_at: string;
}

export interface ContextStats {
  approximate_tokens: number;
  omitted_messages: number;
  truncated_tool_results: number;
  context_window: number;
  reserved_tokens: number;
}

export interface PublicConfig {
  llm_configured: boolean;
  llm_model: string;
  llm_destination: string;
  knowledge_base_enabled: boolean;
  network_tools_enabled: boolean;
  network_allowlist: string[];
  api_auth_enabled: boolean;
  execution_mode: "embedded" | "redis";
  default_llm_profile: string;
  llm_profiles: Array<{
    name: string;
    model: string;
    destination: string;
    configured: boolean;
  }>;
}

export interface Message {
  id: number;
  role: string;
  content: string | null;
  tool_call_id: string | null;
  created_at: string;
}

export interface Approval {
  id: string;
  tool_call_id: string;
  tool_name: string;
  arguments: Record<string, unknown>;
  risk_level: string;
  status: string;
  comment: string | null;
}

let apiToken = "";

export function setApiToken(value: string): void {
  apiToken = value.trim();
}

export function websocketToken(): string {
  return apiToken;
}

async function request<T>(url: string, options?: RequestInit): Promise<T> {
  const response = await fetch(url, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      ...(apiToken ? { "X-Agent-Token": apiToken } : {}),
      ...(options?.headers ?? {}),
    },
  });
  if (!response.ok) {
    const body = await response.json().catch(() => ({ detail: response.statusText }));
    throw new Error(body.detail ?? `HTTP ${response.status}`);
  }
  if (response.status === 204) return undefined as T;
  return response.json() as Promise<T>;
}

export const api = {
  config: () => request<PublicConfig>("/api/config/public"),
  projects: () => request<Project[]>("/api/projects"),
  createProject: (name: string, rootPath: string) =>
    request<Project>("/api/projects", {
      method: "POST",
      body: JSON.stringify({ name, root_path: rootPath }),
    }),
  sessions: () => request<Session[]>("/api/sessions"),
  createSession: (projectId: string, prompt: string, mode: string, llmProfile: string) =>
    request<Session>("/api/sessions", {
      method: "POST",
      body: JSON.stringify({
        project_id: projectId,
        prompt,
        mode,
        llm_profile: llmProfile,
        auto_start: true,
      }),
    }),
  messages: (sessionId: string) =>
    request<Message[]>(`/api/sessions/${sessionId}/messages`),
  children: (sessionId: string) =>
    request<Session[]>(`/api/sessions/${sessionId}/children`),
  audit: (sessionId: string) =>
    request<OutboundAudit[]>(`/api/sessions/${sessionId}/outbound-audit`),
  context: (sessionId: string) =>
    request<ContextStats>(`/api/sessions/${sessionId}/context`),
  sendMessage: (sessionId: string, content: string) =>
    request<Message>(`/api/sessions/${sessionId}/messages`, {
      method: "POST",
      body: JSON.stringify({ content }),
    }),
  approvals: (sessionId: string) =>
    request<Approval[]>(`/api/sessions/${sessionId}/approvals`),
  resolveApproval: (approvalId: string, decision: "approve" | "reject", comment?: string) =>
    request(`/api/approvals/${approvalId}`, {
      method: "POST",
      body: JSON.stringify({ decision, comment: comment || null }),
    }),
  stop: (sessionId: string) =>
    request(`/api/sessions/${sessionId}/stop`, { method: "POST" }),
  archive: (sessionId: string) =>
    request(`/api/sessions/${sessionId}/archive`, { method: "POST" }),
};
