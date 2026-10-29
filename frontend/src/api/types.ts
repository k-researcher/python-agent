// Shared API and domain types. Kept independent from the fetch layer so both the
// REST client and the WebSocket path can use them.

import type { SessionMode } from "../modes";

export interface Project {
  id: string;
  name: string;
  root_path: string;
}

export interface SnapshotStamp {
  run_generation?: number;
  revision?: number;
  consistent?: boolean;
  completed_streams?: Array<{ generation: number; stream_no: number; stream_id: string }>;
}

export interface Session extends SnapshotStamp {
  error_code?: string | null;
  id: string;
  project_id: string;
  parent_id: string | null;
  mode: string;
  llm_profile: string | null;
  engine_profile?: string | null;
  engine_state?: {
    health?: { status: string; diagnostic?: string | null };
    usage?: { input_tokens: number | null; output_tokens: number | null; cache_tokens: number | null; reported_cost: string | null; estimated_cost: boolean } | null;
  } | null;
  configuration?: Record<string, unknown>;
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
  context_window: number | null;
  status?: "unknown";
  reserved_tokens: number;
  // Extended calibration/budget fields from the new API. Optional so older server
  // responses or existing rows continue to satisfy the type.
  model_id?: string | null;
  provider?: string | null;
  wire_version?: string | null;
  raw_tokens?: number | null;
  tokens_before?: number | null;
  calibration_factor?: number | null;
  calibration_samples?: number | null;
  input_budget?: number | null;
  output_tokens?: number | null;
  effective_max_tokens?: number | null;
  safety_margin?: number | null;
  target_tokens?: number | null;
  stage?: string | null;
  fits?: boolean | null;
  masked_tool_results?: number | null;
  folded_tool_calls?: number | null;
  summary_used?: boolean | null;
  summary_calls?: number | null;
  summary_checkpoint_id?: string | null;
  covers_until?: number | null;
  would_summarize?: boolean | null;
  error?: string | null;
}

export interface PublicConfig {
  engine_profiles?: Array<{ id: string; status: "available" | "unavailable" }>;
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
    reasoning_efforts?: string[];
    default_reasoning_effort?: string | null;
  }>;
}

/** One raw LLM tool call stored on an assistant message. */
export interface ToolCallRaw {
  id: string;
  type?: string;
  function?: {
    name?: string;
    arguments?: string;
  };
  [key: string]: unknown;
}

export interface Message {
  id: number;
  session_id: string;
  role: string;
  content: string | null;
  tool_call_id: string | null;
  tool_calls: ToolCallRaw[] | null;
  token_count: number | null;
  skipped: boolean;
  kind: string;
  reasoning_content: string | null;
  reasoning_effort: string | null;
  created_at: string;
  // Summary/context metadata (nullable/optional for compatibility with older rows).
  covers_until?: number | null;
  source_hash?: string | null;
  summary_version?: string | null;
  summary_model?: string | null;
}

export interface Approval {
  executor?: "python" | "engine";
  generation?: number | null;
  scope?: "request" | "turn";
  request_id?: string | null;
  expires_at?: string | null;
  id: string;
  tool_call_id: string;
  tool_name: string;
  arguments: Record<string, unknown>;
  risk_level: string;
  status: string;
  comment: string | null;
  created_at?: string;
}

export interface NewSession {
  project_id: string;
  prompt: string;
  mode?: SessionMode;
  llm_profile?: string;
  engine_profile?: string;
  reasoning_effort?: string;
  title?: string;
  auto_start?: boolean;
}

/** A persisted server event row. */
export interface PersistentServerEvent {
  id: number;
  type: string;
  session_id: string | null;
  payload: Record<string, unknown>;
  created_at: string;
}

/**
 * An ephemeral stream fragment: not stored in the events table. `seq` grows inside a
 * single `stream_id`; a client drops repeats and reorders.
 */
export interface StreamDeltaEvent {
  type: "message.delta" | "reasoning.delta" | "stream.reset";
  session_id: string;
  ephemeral: true;
  payload: { generation?: number; stream_no?: number; stream_id: string; seq?: number; delta?: string };
}

export type ServerEvent = PersistentServerEvent | StreamDeltaEvent;

export function isStreamDeltaEvent(event: ServerEvent): event is StreamDeltaEvent {
  return "ephemeral" in event && (event as { ephemeral?: unknown }).ephemeral === true;
}

/** A session in the sidebar tree: parent followed by its descendants. */
export interface SessionNode {
  session: Session;
  children: SessionNode[];
}

/** Parsed result of a `role="tool"` message (stored as JSON text). */
export interface ToolResult {
  patch?: string;
  paths?: string[];
  status?: "proposed" | "applied";
  success: boolean;
  output?: unknown;
  error?: string;
}

export interface ToolCallView {
  id: string;
  name: string;
  arguments: Record<string, unknown>;
  argumentsRaw: string;
  result: ToolResult | null;
  approval: Approval | null;
  status: "pending" | "awaiting_approval" | "running" | "completed" | "error" | "rejected";
}

/** Merged session feed produced by the session store. */
export type TimelineItem =
  | { kind: "user"; message: Message }
  | { kind: "assistant"; message: Message }
  | { kind: "tool"; call: ToolCallView }
  | { kind: "summary"; message: Message }
  | { kind: "streaming"; content: string; reasoning: string };
