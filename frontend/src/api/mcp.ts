// MCP servers over stdio added in the UI: the list, changes and the start check.

import { request } from "./client";

/** A permission profile of the session roles: "dev" may change files, "ask" only reads. */
export type McpProfile = "dev" | "ask";

export interface McpTool {
  name: string;
  description?: string;
}

export interface McpCheck {
  ok: boolean;
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
  /** Names only: the values are secret and never come back. */
  env_keys: string[];
  /** Empty: every project. */
  project_ids: string[];
  allowed_modes: McpProfile[];
  allow_network: boolean;
  read_paths: string[];
  enabled: boolean;
  last_check: McpCheck | null;
  updated_at: string | null;
}

export interface McpList {
  /** False: the OS sandbox does not work on this computer, so no server starts. */
  sandbox: boolean;
  /** Servers from the operator file; their IDs are taken. */
  file_sources: string[];
  servers: McpServerItem[];
}

export interface McpDraft {
  command: string[];
  /** A value sets the secret, null removes the variable, a missing name keeps it. */
  env: Record<string, string | null>;
  project_ids: string[];
  allowed_modes: McpProfile[];
  allow_network: boolean;
  /** Null on creation: the install directory of the program. */
  read_paths: string[] | null;
  enabled?: boolean;
}

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
    const env = Object.fromEntries(Object.entries(draft.env).filter((entry) => entry[1] !== null));
    return request<McpList>(PREFIX, {
      method: "POST",
      body: JSON.stringify({ id, ...draft, env, ...(draft.read_paths === null ? { read_paths: undefined } : {}) }),
    });
  },
  update: (id: string, patch: Partial<McpDraft>) =>
    request<McpList>(serverPath(id), { method: "PUT", body: JSON.stringify(patch) }),
  remove: (id: string) => request<McpList>(serverPath(id), { method: "DELETE" }),
  check: (id: string) => request<McpList>(`${serverPath(id)}/check`, { method: "POST" }),
};
