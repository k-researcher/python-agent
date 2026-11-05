// Structural types for the chat timeline.
// The components in this folder only depend on these shapes — no store or API imports.

export type ChatRole = "user" | "assistant";

/** Reasoning tokens (exact from the provider, else an estimate) and the reasoning time. */
export interface ReasoningStats {
  tokens: number | null;
  exact: boolean;
  seconds: number | null;
}

/** One chat message as rendered by MessageItem. */
export interface ChatMessage {
  id: string;
  role: ChatRole;
  // User message with files: only the typed text; the files are listed in ``attachments``.
  content: string;
  attachments?: { name: string; size: number }[];
  reasoning_content: string | null;
  reasoning_effort: string | null;
  reasoning_stats?: ReasoningStats | null;
  created_at: string;
  kind: ChatRole;
}

/** A tool approval request (structurally compatible with the backend Approval). */
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
}

export type ToolStatus =
  | "pending"
  | "awaiting_approval"
  | "running"
  | "completed"
  | "error"
  | "rejected";

/** Result of one tool call, taken from the matching role="tool" message. */
export interface ToolResult {
  patch?: string;
  paths?: string[];
  status?: "proposed" | "applied";
  success: boolean;
  output?: unknown;
  error?: string;
}

/** One tool call in the timeline. */
export interface ToolCallView {
  id: string;
  name: string;
  arguments: Record<string, unknown>;
  argumentsRaw: string;
  result: ToolResult | null;
  approval: Approval | null;
  status: ToolStatus;
}

/** Unified feed item. */
export type TimelineItem =
  | { kind: "user"; message: ChatMessage }
  | { kind: "assistant"; message: ChatMessage }
  | { kind: "tool"; call: ToolCallView }
  | { kind: "summary"; message: ChatMessage }
  | { kind: "streaming"; content: string; reasoning: string; stats?: ReasoningStats };

/** Payload of the Composer `send` event. */
/** A text file attached in the composer. */
export interface AttachedFile {
  name: string;
  text: string;
  size: number;
}

export interface SendPayload {
  text: string;
  effort: string;
  files: AttachedFile[];
}

/** Payload of the ToolCallCard / ApprovalInline `reject` event. */
export interface RejectPayload {
  approvalId: string;
  comment: string;
}
