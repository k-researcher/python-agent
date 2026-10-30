// Structural types for the chat timeline.
// The components in this folder only depend on these shapes — no store or API imports.

export type ChatRole = "user" | "assistant";

/** One chat message as rendered by MessageItem. */
export interface ChatMessage {
  id: string;
  role: ChatRole;
  content: string;
  reasoning_content: string | null;
  reasoning_effort: string | null;
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
  | { kind: "streaming"; content: string; reasoning: string };

/** Payload of the Composer `send` event. */
export interface SendPayload {
  text: string;
  effort: string;
}

/** Payload of the ToolCallCard / ApprovalInline `reject` event. */
export interface RejectPayload {
  approvalId: string;
  comment: string;
}
