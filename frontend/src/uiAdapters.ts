// Typed adapters between store/API types and component prop types.
// Component layer keeps its own contracts; these functions map fields explicitly.

import type {
  Message as ApiMessage,
  TimelineItem as ApiTimelineItem,
  ToolCallView as ApiToolCallView,
  Approval as ApiApproval,
  ToolResult as ApiToolResult,
  Session as ApiSession,
  SessionNode as ApiSessionNode,
  ContextStats as ApiContextStats,
  OutboundAudit as ApiOutboundAudit,
  Project as ApiProject,
  PublicConfig,
} from "./api/types";
import type {
  SessionNode as LayoutSessionNode,
  ContextStats as LayoutContextStats,
  ModelOption,
  ProjectOption,
  ChildSession,
  AuditEntry,
  ToolEntry,
} from "./components/layout/types";
import type {
  ChatMessage,
  TimelineItem as ChatTimelineItem,
  ToolCallView as ChatToolCallView,
  Approval as ChatApproval,
  ToolResult as ChatToolResult,
} from "./components/chat/types";

// ---------------------------------------------------------------------------
// Layout type adapters
// ---------------------------------------------------------------------------

export function toModelOptions(config: PublicConfig | null): ModelOption[] {
  if (!config) return [];
  return config.llm_profiles
    .filter((profile) => profile.configured)
    .map((profile) => ({
      id: profile.name,
      label: `${profile.name} · ${profile.model}`,
    }));
}

export function toProjectOptions(projects: ApiProject[]): ProjectOption[] {
  return projects.map((p) => ({ id: p.id, name: p.name }));
}

export function toLayoutSessionNode(node: ApiSessionNode): LayoutSessionNode {
  return {
    id: node.session.id,
    title: node.session.title,
    status: node.session.status,
    archived: node.session.archived,
    children: node.children.map(toLayoutSessionNode),
  };
}

export function toLayoutSessionTree(nodes: ApiSessionNode[]): LayoutSessionNode[] {
  return nodes.map(toLayoutSessionNode);
}

export function toLayoutContext(
  stats: ApiContextStats | null,
): LayoutContextStats | null {
  if (!stats) return null;
  return {
    approximate_tokens: stats.approximate_tokens,
    omitted_messages: stats.omitted_messages,
    context_window: stats.context_window,
    input_budget: stats.input_budget ?? undefined,
    fits: stats.fits ?? undefined,
    error: stats.error ?? undefined,
    stage: stats.stage ?? undefined,
    summary_used: stats.summary_used ?? undefined,
    would_summarize: stats.would_summarize ?? undefined,
    output_tokens: stats.output_tokens ?? undefined,
    safety_margin: stats.safety_margin ?? undefined,
    calibration_factor: stats.calibration_factor ?? undefined,
  };
}

export function toChildSessions(sessions: ApiSession[]): ChildSession[] {
  return sessions.map((s) => ({ id: s.id, title: s.title, status: s.status }));
}

export function toAuditEntries(rows: ApiOutboundAudit[]): AuditEntry[] {
  return rows.map((row) => ({
    id: row.id,
    operation: row.operation,
    destination: row.destination,
    status: row.status,
    created_at: row.created_at,
  }));
}

export function toToolEntries(config: PublicConfig | null): ToolEntry[] {
  if (!config) return [];
  const tools: ToolEntry[] = [
    { name: "list_dir", enabled: true },
    { name: "read_file", enabled: true },
    { name: "write_file", enabled: true },
    { name: "edit_file", enabled: true },
    { name: "multi_edit", enabled: true },
    { name: "remove_file", enabled: true },
    { name: "glob", enabled: true },
    { name: "search_files", enabled: true },
    { name: "shell", enabled: true },
    { name: "run_agent", enabled: true },
  ];
  tools.push({ name: "http_get", enabled: config.network_tools_enabled });
  tools.push({ name: "knowledge_search", enabled: config.knowledge_base_enabled });
  return tools;
}

// ---------------------------------------------------------------------------
// Chat type adapters
// ---------------------------------------------------------------------------

function toChatRole(role: string): "user" | "assistant" {
  return role === "user" ? "user" : "assistant";
}

export function toChatMessage(message: ApiMessage): ChatMessage {
  return {
    id: String(message.id),
    role: toChatRole(message.role),
    content: message.content ?? "",
    reasoning_content: message.reasoning_content,
    reasoning_effort: message.reasoning_effort,
    created_at: message.created_at,
    kind: toChatRole(message.kind),
  };
}

export function toChatToolResult(result: ApiToolResult | null): ChatToolResult | null {
  if (!result) return null;
  return {
    success: result.success,
    ...(result.output !== undefined ? { output: result.output } : {}),
    ...(result.error !== undefined ? { error: result.error } : {}),
  };
}

export function toChatApproval(approval: ApiApproval | null): ChatApproval | null {
  if (!approval) return null;
  return {
    id: approval.id,
    tool_call_id: approval.tool_call_id,
    tool_name: approval.tool_name,
    arguments: approval.arguments,
    risk_level: approval.risk_level,
    status: approval.status,
    comment: approval.comment,
  };
}

export function toChatToolCallView(call: ApiToolCallView): ChatToolCallView {
  return {
    id: call.id,
    name: call.name,
    arguments: call.arguments,
    argumentsRaw: call.argumentsRaw,
    result: toChatToolResult(call.result),
    approval: toChatApproval(call.approval),
    status: call.status,
  };
}

export function toChatTimelineItems(items: ApiTimelineItem[]): ChatTimelineItem[] {
  return items.map((item) => {
    switch (item.kind) {
      case "user":
        return { kind: "user", message: toChatMessage(item.message) };
      case "assistant":
        return { kind: "assistant", message: toChatMessage(item.message) };
      case "tool":
        return { kind: "tool", call: toChatToolCallView(item.call) };
      case "summary":
        return { kind: "summary", message: toChatMessage(item.message) };
      case "streaming":
        return { kind: "streaming", content: item.content, reasoning: item.reasoning };
    }
  });
}
