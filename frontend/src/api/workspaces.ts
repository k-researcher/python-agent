import { request } from "./client";

export type WorkspaceRole = "owner" | "admin" | "member" | "viewer";

export type AssignableRole = "admin" | "member" | "viewer";

export interface WorkspaceCapabilities {
  view_project: boolean;
  run_agent: boolean;
  manage_members: boolean;
  manage_workspace: boolean;
  manage_models: boolean;
}

export interface WorkspaceSummary {
  id: string;
  name: string;
  role: WorkspaceRole;
  capabilities: WorkspaceCapabilities;
}

export interface WorkspaceMember {
  user_id: string;
  display_name: string;
  role: WorkspaceRole;
  enabled: boolean;
  can_remove: boolean;
  assignable_roles: AssignableRole[];
}

const validateId = (id: string): void => {
  if (typeof id !== "string" || id.trim() === "") {
    throw new Error("Invalid ID");
  }
};

const validateRole = (role: string): void => {
  if (!["admin", "member", "viewer"].includes(role)) {
    throw new Error("Invalid assignable role");
  }
};

export async function listWorkspaces(): Promise<WorkspaceSummary[]> {
  return request<WorkspaceSummary[]>("/api/workspaces");
}

export async function listWorkspaceMembers(workspaceId: string): Promise<WorkspaceMember[]> {
  validateId(workspaceId);
  return request<WorkspaceMember[]>(`/api/workspaces/${encodeURIComponent(workspaceId)}/members`);
}

export async function changeWorkspaceMemberRole(
  workspaceId: string,
  userId: string,
  role: AssignableRole,
): Promise<WorkspaceMember> {
  validateId(workspaceId);
  validateId(userId);
  validateRole(role);
  return request<WorkspaceMember>(`/api/workspaces/${encodeURIComponent(workspaceId)}/members/${encodeURIComponent(userId)}`, {
    method: "PATCH",
    body: JSON.stringify({ role }),
  });
}

export async function removeWorkspaceMember(workspaceId: string, userId: string): Promise<void> {
  validateId(workspaceId);
  validateId(userId);
  await request<void>(`/api/workspaces/${encodeURIComponent(workspaceId)}/members/${encodeURIComponent(userId)}`, {
    method: "DELETE",
  });
}
