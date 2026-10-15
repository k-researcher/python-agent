import { afterEach, describe, expect, it, vi } from "vitest";
import {
  listWorkspaces,
  listWorkspaceMembers,
  changeWorkspaceMemberRole,
  removeWorkspaceMember,
  type WorkspaceSummary,
  type WorkspaceMember,
  type AssignableRole,
} from "./workspaces";
import { ApiError } from "./client";

type FetchImpl = (input: unknown, init?: RequestInit) => Promise<unknown>;

function jsonResponse(status: number, body: unknown): unknown {
  return {
    ok: status >= 200 && status < 300,
    status,
    statusText: status >= 200 && status < 300 ? "OK" : "Error",
    json: async () => body,
    headers: new Headers(),
  };
}

function stubFetch(impl: FetchImpl): void {
  vi.stubGlobal("fetch", vi.fn(impl));
}

function lastFetchCall(init?: RequestInit): { url: string; init?: RequestInit } {
  const mock = vi.mocked(fetch);
  const call = mock.mock.calls.at(-1);
  if (!call) throw new Error("fetch was not called");
  return { url: String(call[0]), init: (call[1] as RequestInit | undefined) ?? init };
}

afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe("workspaces api", () => {
  it("listWorkspaces returns summaries", async () => {
    const summaries: WorkspaceSummary[] = [
      {
        id: "ws1",
        name: "Project A",
        role: "admin",
        capabilities: {
          view_project: true,
          run_agent: true,
          manage_members: true,
          manage_workspace: true,
          manage_models: false,
        },
      },
    ];
    stubFetch(async () => jsonResponse(200, summaries));

    const result = await listWorkspaces();
    expect(result).toEqual(summaries);
    expect(lastFetchCall().url).toBe("/api/workspaces");
  });

  it("listWorkspaceMembers handles encoded IDs and returns members", async () => {
    const members: WorkspaceMember[] = [
      {
        user_id: "u1",
        display_name: "Alice",
        role: "admin",
        enabled: true,
        can_remove: true,
        assignable_roles: ["admin", "member", "viewer"],
      },
    ];
    const wsId = "ws space/123";
    stubFetch(async () => jsonResponse(200, members));

    const result = await listWorkspaceMembers(wsId);
    expect(result).toEqual(members);
    expect(lastFetchCall().url).toBe(`/api/workspaces/${encodeURIComponent(wsId)}/members`);
  });

  it("changeWorkspaceMemberRole sends correct payload and URL", async () => {
    const updatedMember: WorkspaceMember = {
      user_id: "u1",
      display_name: "Alice",
      role: "member",
      enabled: true,
      can_remove: true,
      assignable_roles: ["admin", "member", "viewer"],
    };
    const wsId = "ws1";
    const userId = "u1";
    const newRole: AssignableRole = "member";

    stubFetch(async (_url, init) => {
      const body = JSON.parse(String(init?.body)) as { role: AssignableRole };
      expect(body.role).toBe(newRole);
      return jsonResponse(200, updatedMember);
    });

    const result = await changeWorkspaceMemberRole(wsId, userId, newRole);
    expect(result).toEqual(updatedMember);
    expect(lastFetchCall().url).toBe(`/api/workspaces/${wsId}/members/${userId}`);
    expect(lastFetchCall().init?.method).toBe("PATCH");
  });

  it("removeWorkspaceMember sends DELETE and handles 204", async () => {
    const wsId = "ws1";
    const userId = "u1";
    stubFetch(async () => jsonResponse(204, null));

    await expect(removeWorkspaceMember(wsId, userId)).resolves.toBeUndefined();
    expect(lastFetchCall().url).toBe(`/api/workspaces/${wsId}/members/${userId}`);
    expect(lastFetchCall().init?.method).toBe("DELETE");
  });

  it("encodes both mutation ids without trimming or changing their scope", async () => {
    const workspaceId = " команда/?space ";
    const userId = "участник/?id";
    stubFetch(async () => jsonResponse(200, { user_id: userId, role: "viewer" }));
    await changeWorkspaceMemberRole(workspaceId, userId, "viewer");
    expect(lastFetchCall()).toMatchObject({
      url: `/api/workspaces/${encodeURIComponent(workspaceId)}/members/${encodeURIComponent(userId)}`,
      init: {
        method: "PATCH",
        credentials: "same-origin",
        body: JSON.stringify({ role: "viewer" }),
      },
    });
  });

  it("rejects empty or whitespace IDs before fetch", async () => {
    stubFetch(async () => jsonResponse(200, []));
    await expect(listWorkspaceMembers(" ")).rejects.toThrow("Invalid ID");
    await expect(changeWorkspaceMemberRole("", "u1", "admin")).rejects.toThrow("Invalid ID");
    await expect(removeWorkspaceMember("ws1", "  ")).rejects.toThrow("Invalid ID");
    expect(fetch).not.toHaveBeenCalled();
  });

  it.each(["owner", "unknown", "", " admin"])("rejects %s at runtime before fetch", async (role) => {
    stubFetch(async () => jsonResponse(200, {}));
    await expect(changeWorkspaceMemberRole("ws1", "u1", role as AssignableRole))
      .rejects.toThrow("Invalid assignable role");
    expect(fetch).not.toHaveBeenCalled();
  });

  it.each([403, 404])("propagates HTTP %s as ApiError", async (status) => {
    stubFetch(async () => jsonResponse(status, { detail: "Denied" }));
    await expect(listWorkspaces()).rejects.toBeInstanceOf(ApiError);
    await expect(listWorkspaces()).rejects.toMatchObject({ status, message: "Denied" });
  });

  it("propagates network errors", async () => {
    stubFetch(async () => {
      throw new Error("Network Failure");
    });
    await expect(listWorkspaces()).rejects.toThrow("Network Failure");
  });
});
