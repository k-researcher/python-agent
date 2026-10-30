import { afterEach, describe, expect, it, vi } from "vitest";
import { api, ApiError, request, setCsrfToken } from "./client";
import type { Approval } from "./types";

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
  setCsrfToken("");
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe("api client request", () => {
  it("sends same-origin credentials and parses JSON on GET", async () => {
    stubFetch(async (input) => {
      expect(String(input)).toBe("/api/projects");
      return jsonResponse(200, [{ id: "p1" }]);
    });
    const result = await request<unknown>("/api/projects");
    expect(result).toEqual([{ id: "p1" }]);
    const { init } = lastFetchCall();
    expect(init?.credentials).toBe("same-origin");
  });

  it("adds the CSRF header to mutating requests when a token is set", async () => {
    setCsrfToken("csrf-123");
    stubFetch(async (_input, init) => {
      const headers = init?.headers as Record<string, string>;
      expect(headers["X-Agent-CSRF"]).toBe("csrf-123");
      return jsonResponse(200, { stopped: true });
    });
    await request("/api/sessions/s1/stop", { method: "POST" });
  });

  it("omits the CSRF header when no token is set and for GET", async () => {
    stubFetch(async (_input, init) => {
      const headers = (init?.headers as Record<string, string>) ?? {};
      expect(headers["X-Agent-CSRF"]).toBeUndefined();
      return jsonResponse(200, {});
    });
    await request("/api/sessions/s1/stop", { method: "POST" });
  });

  it("reports the server detail on a JSON error body", async () => {
    stubFetch(async () => jsonResponse(409, { detail: "Session is busy" }));
    const error = await request("/api/sessions/s1/messages", { method: "POST" }).catch(
      (e: unknown) => e,
    );
    expect(error).toBeInstanceOf(ApiError);
    if (error instanceof ApiError) {
      expect(error.status).toBe(409);
      expect(error.message).toBe("Session is busy");
    }
  });

  it("joins a validation error list into one readable message", async () => {
    stubFetch(async () =>
      jsonResponse(422, {
        detail: [
          { loc: ["routing", "default"], msg: "Value error, unknown model reference" },
          { msg: "Input should be greater than or equal to 4096" },
          { nothing: true },
        ],
      }),
    );
    const error = await request("/api/settings/models/routing", { method: "PUT" }).catch((e: unknown) => e);
    expect(error).toBeInstanceOf(ApiError);
    if (error instanceof ApiError) {
      expect(error.status).toBe(422);
      expect(error.message).toBe(
        "routing.default: Value error, unknown model reference; Input should be greater than or equal to 4096",
      );
    }
  });

  it("falls back to the HTTP status text when the body has no detail", async () => {
    stubFetch(async () => jsonResponse(500, { oops: true }));
    const error = await request("/api/nope").catch((e: unknown) => e);
    expect(error).toBeInstanceOf(ApiError);
    if (error instanceof ApiError) {
      expect(error.status).toBe(500);
      expect(error.message).toBe("HTTP 500");
    }
  });

  it("returns undefined for a 204 response", async () => {
    stubFetch(async () => jsonResponse(204, null));
    await expect(request("/api/deleted", { method: "DELETE" })).resolves.toBeUndefined();
  });
});

describe("api.sendMessage", () => {
  it("sends reasoning_effort when provided", async () => {
    stubFetch(async (_input, init) => {
      const body = JSON.parse(String(init?.body)) as Record<string, unknown>;
      expect(body.reasoning_effort).toBe("high");
      return jsonResponse(201, { id: 1 });
    });
    await api.sendMessage("s1", "text", "high");
  });

  it("sends a null reasoning_effort when omitted", async () => {
    stubFetch(async (_input, init) => {
      const body = JSON.parse(String(init?.body)) as Record<string, unknown>;
      expect(body.reasoning_effort).toBeNull();
      return jsonResponse(201, { id: 2 });
    });
    await api.sendMessage("s1", "text");
  });

  it("treats a blank reasoning effort as unset", async () => {
    stubFetch(async (_input, init) => {
      const body = JSON.parse(String(init?.body)) as Record<string, unknown>;
      expect(body.reasoning_effort).toBeNull();
      return jsonResponse(201, { id: 3 });
    });
    await api.sendMessage("s1", "text", "   ");
  });

  it("always includes the content in the body", async () => {
    stubFetch(async (_input, init) => {
      const body = JSON.parse(String(init?.body)) as Record<string, unknown>;
      expect(body.content).toBe("hello");
      return jsonResponse(201, { id: 4 });
    });
    await api.sendMessage("s1", "hello");
  });
});

describe("api.approvals", () => {
  it("requests the complete approval history", async () => {
    const approvals: Approval[] = ["pending", "approved", "rejected", "cancelled"].map((status) => ({
      id: status, tool_call_id: `call-${status}`, tool_name: "shell", arguments: {},
      risk_level: "high", status, comment: null,
    }));
    stubFetch(async () => jsonResponse(200, { data: approvals, revision: 17, run_generation: 2, consistent: true }));
    const result = await api.approvals("session");
    expect(lastFetchCall().url).toBe("/api/sessions/session/snapshot/approvals");
    expect(Array.from(result)).toEqual(approvals);
    expect(result).toMatchObject({ revision: 17, run_generation: 2 });
  });
});
