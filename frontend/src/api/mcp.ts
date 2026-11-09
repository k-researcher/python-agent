// MCP servers added in the UI: the list, changes and the connection check.

import { request } from "./client";

/** A permission profile of the session roles: "dev" may change files, "ask" only reads. */
export type McpProfile = "dev" | "ask";
export type McpHttpProtocol = "auto" | "modern" | "legacy";

export interface McpTool {
  name: string;
  description?: string;
}

export interface McpCheck {
  ok: boolean;
  protocol?: string;
  checked_at?: string;
  server?: { name: string; version: string };
  tools?: McpTool[];
  tool_count?: number;
  error?: string;
  stderr?: string;
}

export interface McpServerItem {
  id: string;
  command: string[];
  url: string | null;
  header_keys: string[];
  http_protocol: McpHttpProtocol;
  /** Names only: the values are secret and never come back. */
  env_keys: string[];
  /** Empty: every project. */
  project_ids: string[];
  allowed_modes: McpProfile[];
  allow_network: boolean;
  read_paths: string[];
  enabled: boolean;
  /** How long one tool call may wait for the server, in seconds. */
  call_timeout: number;
  last_check: McpCheck | null;
  updated_at: string | null;
}

export interface McpList {
  /** False: the OS sandbox does not work, so commands cannot start. */
  sandbox: boolean;
  /** Servers from the operator file; their IDs are taken. */
  file_sources: string[];
  servers: McpServerItem[];
}

export interface McpDraft {
  command?: string[];
  url?: string;
  headers?: Record<string, string | null>;
  http_protocol?: McpHttpProtocol;
  /** A value sets the secret, null removes the variable, a missing name keeps it. */
  env?: Record<string, string | null>;
  project_ids: string[];
  allowed_modes: McpProfile[];
  allow_network?: boolean;
  /** Null on creation: the install directory of the program. */
  read_paths?: string[] | null;
  enabled?: boolean;
  call_timeout?: number;
}

export const MCP_CALL_SECONDS = { min: 5, max: 600, default: 120 } as const;
export const MCP_SERVER_ID = /^[a-z0-9][a-z0-9_-]{0,31}$/;
export const MCP_ENV_KEY = /^[A-Za-z_][A-Za-z0-9_]*$/;

const PREFIX = "/api/mcp/servers";
const serverPath = (id: string) => `${PREFIX}/${encodeURIComponent(id)}`;

/** Split a command line into its parts; quotes keep spaces inside one part. */
export function splitCommand(line: string): string[] {
  const parts: string[] = [];
  let current = "";
  let quote: string | null = null;
  let started = false;
  for (const char of line.trim()) {
    if (quote) {
      if (char === quote) quote = null;
      else current += char;
    } else if (char === '"' || char === "'") {
      quote = char;
      started = true;
    } else if (/\s/.test(char)) {
      if (started) parts.push(current);
      current = "";
      started = false;
    } else {
      current += char;
      started = true;
    }
  }
  if (quote) throw new Error("Не закрыта кавычка в команде");
  if (started) parts.push(current);
  return parts;
}

/** Join the parts back into one line; a part with a space or a quote gets quotes. */
export function joinCommand(parts: string[]): string {
  return parts
    .map((part) => {
      if (part && !/[\s"']/.test(part)) return part;
      return part.includes('"') ? `'${part}'` : `"${part}"`;
    })
    .join(" ");
}

export const mcpApi = {
  list: () => request<McpList>(PREFIX),
  create: (id: string, draft: McpDraft) => {
    const secrets = Object.fromEntries(
      (["env", "headers"] as const).filter((key) => draft[key] !== undefined).map((key) => [
        key, Object.fromEntries(Object.entries(draft[key]!).filter((entry) => entry[1] !== null)),
      ]),
    );
    return request<McpList>(PREFIX, {
      method: "POST",
      body: JSON.stringify({ id, ...draft, ...secrets, ...(draft.read_paths === null ? { read_paths: undefined } : {}) }),
    });
  },
  update: (id: string, patch: Partial<McpDraft>) =>
    request<McpList>(serverPath(id), { method: "PUT", body: JSON.stringify(patch) }),
  remove: (id: string) => request<McpList>(serverPath(id), { method: "DELETE" }),
  check: (id: string) => request<McpList>(`${serverPath(id)}/check`, { method: "POST" }),
};
