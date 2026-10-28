import type {
  Approval,
  ContextStats,
  Message,
  OutboundAudit,
  Project,
  PublicConfig,
  Session,
  SnapshotStamp,
} from "./types";

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

// The CSRF token lives only in memory; the session itself is an HttpOnly cookie.
let csrfToken = "";

export function setCsrfToken(value: string): void {
  csrfToken = value;
}

const MUTATING = new Set(["POST", "PUT", "PATCH", "DELETE"]);

export async function request<T>(url: string, options?: RequestInit): Promise<T> {
  const method = (options?.method ?? "GET").toUpperCase();
  const response = await fetch(url, {
    ...options,
    credentials: "same-origin",
    headers: {
      "Content-Type": "application/json",
      ...(MUTATING.has(method) && csrfToken ? { "X-Agent-CSRF": csrfToken } : {}),
      ...(options?.headers ?? {}),
    },
  });
  if (!response.ok) {
    const body = await response.json().catch(() => ({ detail: response.statusText }));
    const detail = typeof body.detail === "string" ? body.detail : null;
    throw new ApiError(detail ?? `HTTP ${response.status}`, response.status);
  }
  if (response.status === 204) return undefined as T;
  return response.json() as Promise<T>;
}

export interface CreateSessionInput {
  project_id: string;
  prompt: string;
  mode?: string;
  llm_profile?: string;
  reasoning_effort?: string;
  title?: string;
  auto_start?: boolean;
}

// Object form (session store — forwards optional NewSession fields) and legacy
// positional form used by App.vue are both supported.
function createSessionRequest(projectId: string, prompt: string, mode: string, llmProfile: string): Promise<Session>;
function createSessionRequest(input: CreateSessionInput): Promise<Session>;
function createSessionRequest(
  a: CreateSessionInput | string,
  b?: string,
  c?: string,
  d?: string,
): Promise<Session> {
  const input = typeof a === "object" && a !== null
    ? a
    : {
        project_id: String(a),
        prompt: String(b ?? ""),
        mode: c,
        llm_profile: d,
        reasoning_effort: undefined as string | undefined,
        title: undefined as string | undefined,
        auto_start: true,
      };
  return request<Session>("/api/sessions", {
    method: "POST",
    body: JSON.stringify({
      project_id: input.project_id,
      prompt: input.prompt,
      mode: input.mode,
      llm_profile: input.llm_profile,
      reasoning_effort: input.reasoning_effort ?? null,
      title: input.title,
      auto_start: input.auto_start ?? true,
    }),
  });
}

async function snapshot<T extends object>(sessionId: string, resource: string): Promise<T & SnapshotStamp> {
  const envelope = await request<{ data: T; run_generation: number; revision: number; consistent: boolean; completed_streams?: SnapshotStamp["completed_streams"] }>(
    `/api/sessions/${sessionId}/snapshot/${resource}`,
  );
  return Object.assign(Array.isArray(envelope.data) ? [...envelope.data] : { ...envelope.data }, { run_generation: envelope.run_generation,
    revision: envelope.revision, consistent: envelope.consistent,
    ...(envelope.completed_streams ? { completed_streams: envelope.completed_streams } : {}) }) as T & SnapshotStamp;
}

export const api = {
  config: (): Promise<PublicConfig> => request<PublicConfig>("/api/config/public"),
  projects: (): Promise<Project[]> => request<Project[]>("/api/projects"),
  createProject: (name: string, rootPath: string): Promise<Project> =>
    request<Project>("/api/projects", {
      method: "POST",
      body: JSON.stringify({ name, root_path: rootPath }),
    }),
  sessions: (): Promise<Session[]> => request<Session[]>("/api/sessions"),
  createSession: createSessionRequest,
  session: (sessionId: string): Promise<Session> =>
    snapshot<Session>(sessionId, "session"),
  messages: (sessionId: string): Promise<Message[]> =>
    snapshot<Message[]>(sessionId, "messages"),
  children: (sessionId: string): Promise<Session[]> =>
    snapshot<Session[]>(sessionId, "children"),
  audit: (sessionId: string): Promise<OutboundAudit[]> =>
    snapshot<OutboundAudit[]>(sessionId, "audit"),
  context: (sessionId: string): Promise<ContextStats> =>
    snapshot<ContextStats>(sessionId, "context"),
  sendMessage: (sessionId: string, content: string, reasoningEffort?: string): Promise<Message> =>
    request<Message>(`/api/sessions/${sessionId}/messages`, {
      method: "POST",
      body: JSON.stringify({
        content,
        reasoning_effort: reasoningEffort && reasoningEffort.trim() !== "" ? reasoningEffort : null,
      }),
    }),
  approvals: (sessionId: string): Promise<Approval[]> =>
    snapshot<Approval[]>(sessionId, "approvals"),
  resolveApproval: (
    approvalId: string,
    decision: "approve" | "reject",
    comment?: string,
  ): Promise<{ session_id: string; decision: string }> =>
    request(`/api/approvals/${approvalId}`, {
      method: "POST",
      body: JSON.stringify({ decision, comment: comment || null }),
    }),
  stop: (sessionId: string): Promise<{ stopped: boolean }> =>
    request(`/api/sessions/${sessionId}/stop`, { method: "POST" }),
  archive: (sessionId: string): Promise<{ archived: boolean }> =>
    request(`/api/sessions/${sessionId}/archive`, { method: "POST" }),
};
