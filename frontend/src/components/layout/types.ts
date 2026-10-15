// Shared types for the layout components (spec §6).

export interface SessionNode {
  id: string;
  title: string;
  status: string;
  archived: boolean;
  children: SessionNode[];
}

export interface ModelOption {
  id: string;
  label: string;
}

export interface ProjectOption {
  id: string;
  name: string;
}

export interface ChildSession {
  id: string;
  title: string;
  status: string;
}

export interface AuditEntry {
  id: number;
  operation: string;
  destination: string;
  status: string;
  created_at: string;
}

export interface ContextStats {
  approximate_tokens: number;
  context_window: number;
  omitted_messages: number;
  input_budget?: number;
  fits?: boolean;
  error?: string;
  stage?: string;
  summary_used?: boolean;
  would_summarize?: boolean;
  calibration_factor?: number;
  output_tokens?: number;
  safety_margin?: number;
}

export interface ToolEntry {
  name: string;
  enabled: boolean;
}
