import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { createPinia, setActivePinia } from "pinia";
import { flushPromises } from "@vue/test-utils";
import { api } from "../api/client";
import type { Message, Project, PublicConfig, Session } from "../api/types";
import { useAppStore } from "./app";
import { useSessionStore } from "./session";

function deferred<Value>() {
  let resolve!: (value: Value) => void;
  const promise = new Promise<Value>((complete) => { resolve = complete; });
  return { promise, resolve };
}

function project(id: string): Project {
  return { id, name: id, root_path: `/projects/${id}` };
}

function config(model: string): PublicConfig {
  return {
    llm_configured: true, llm_model: model, llm_destination: "http://local",
    knowledge_base_enabled: false, network_tools_enabled: false,
    network_allowlist: [], api_auth_enabled: true, execution_mode: "embedded",
    default_llm_profile: "default", llm_profiles: [],
  };
}

function session(id: string): Session {
  return {
    id, project_id: "new", parent_id: null, mode: "dev", llm_profile: "default",
    title: id, status: "completed", error: null, archived: false,
    created_at: "2026-01-01T00:00:00Z", updated_at: "2026-01-01T00:00:00Z",
  };
}

function message(id: number): Message {
  return {
    id, session_id: "same", role: "user", content: "private", tool_call_id: null,
    tool_calls: null, token_count: null, skipped: false, kind: "normal",
    reasoning_content: null, reasoning_effort: null, created_at: "2026-01-01T00:00:00Z",
  };
}

beforeEach(() => {
  setActivePinia(createPinia());
  vi.spyOn(api, "config").mockResolvedValue(config("new"));
  vi.spyOn(api, "projects").mockResolvedValue([project("new")]);
  vi.spyOn(api, "sessions").mockResolvedValue([session("new")]);
  vi.spyOn(api, "messages").mockResolvedValue([]);
  vi.spyOn(api, "approvals").mockResolvedValue([]);
  vi.spyOn(api, "children").mockResolvedValue([]);
  vi.spyOn(api, "audit").mockResolvedValue([]);
  vi.spyOn(api, "context").mockResolvedValue({
    approximate_tokens: 0, omitted_messages: 0, truncated_tool_results: 0,
    context_window: 10000, reserved_tokens: 1000,
  });
  vi.spyOn(api, "session").mockResolvedValue(session("same"));
});

afterEach(() => vi.restoreAllMocks());

describe("application access boundary", () => {
  it("reset clears all loaded application data", async () => {
    const store = useAppStore();
    await store.load();
    store.connection = "live";
    store.reset();
    expect(store.config).toBeNull();
    expect(store.projects).toEqual([]);
    expect(store.sessions).toEqual([]);
    expect(store.selectedProjectId).toBeNull();
    expect(store.connection).toBe("offline");
  });

  it("a late full reload cannot overwrite the next access scope", async () => {
    const previousProjects = deferred<Project[]>();
    vi.mocked(api.projects).mockReturnValueOnce(previousProjects.promise);
    vi.mocked(api.config).mockResolvedValueOnce(config("previous"));
    const store = useAppStore();
    const previousLoad = store.load();
    store.reset();
    await store.load();
    previousProjects.resolve([project("previous")]);
    await previousLoad;
    expect(store.config?.llm_model).toBe("new");
    expect(store.projects).toEqual([project("new")]);
    expect(store.selectedProjectId).toBe("new");
  });

  it("a late project event reload cannot restore cleared data", async () => {
    const previousProjects = deferred<Project[]>();
    vi.mocked(api.projects).mockReturnValueOnce(previousProjects.promise);
    const store = useAppStore();
    store.applyEvent({ id: 1, created_at: "2026-01-01T00:00:00Z", type: "project.created", session_id: null, payload: {} });
    store.reset();
    previousProjects.resolve([project("previous")]);
    await flushPromises();
    expect(store.projects).toEqual([]);
    expect(store.selectedProjectId).toBeNull();
  });

  it("a late session event reload cannot restore cleared data", async () => {
    const previousSessions = deferred<Session[]>();
    vi.mocked(api.sessions).mockReturnValueOnce(previousSessions.promise);
    const store = useAppStore();
    store.applyEvent({ id: 1, created_at: "2026-01-01T00:00:00Z", type: "session.created", session_id: "previous", payload: {} });
    store.reset();
    previousSessions.resolve([session("previous")]);
    await flushPromises();
    expect(store.sessions).toEqual([]);
  });

  it("a late project creation does not select or reload into the next scope", async () => {
    const creation = deferred<Project>();
    vi.spyOn(api, "createProject").mockReturnValue(creation.promise);
    const store = useAppStore();
    const pending = store.createProject("previous", "/previous");
    store.reset();
    creation.resolve(project("previous"));
    await pending;
    expect(api.projects).not.toHaveBeenCalled();
    expect(store.selectedProjectId).toBeNull();
  });

  it("a late session creation does not reload into the next scope", async () => {
    const creation = deferred<Session>();
    vi.spyOn(api, "createSession").mockReturnValue(creation.promise);
    const store = useAppStore();
    const pending = store.createSession({ project_id: "previous", prompt: "task" });
    store.reset();
    creation.resolve(session("previous"));
    await pending;
    expect(api.sessions).not.toHaveBeenCalled();
    expect(store.sessions).toEqual([]);
  });

  it("project selection reconciles when the server removes the selected project", async () => {
    const store = useAppStore();
    store.selectedProjectId = "removed";
    await store.load();
    expect(store.selectedProjectId).toBe("new");
    vi.mocked(api.projects).mockResolvedValue([]);
    await store.load();
    expect(store.selectedProjectId).toBeNull();
  });
});

describe("session access boundary", () => {
  it("close clears the session and invalidates an unfinished reload", async () => {
    const previousMessages = deferred<Message[]>();
    vi.mocked(api.messages).mockReturnValueOnce(previousMessages.promise);
    const store = useSessionStore();
    const pending = store.open("same");
    store.close();
    previousMessages.resolve([message(1)]);
    await pending;
    expect(store.sessionId).toBeNull();
    expect(store.session).toBeNull();
    expect(store.messages).toEqual([]);
    expect(store.approvals).toEqual([]);
    expect(store.context).toBeNull();
    expect(store.audit).toEqual([]);
    expect(store.children).toEqual([]);
    expect(store.streaming).toBeNull();
    expect(store.timeline).toEqual([]);
  });

  it("closing and reopening the same ID still rejects the previous generation", async () => {
    const previousMessages = deferred<Message[]>();
    vi.mocked(api.messages).mockReturnValueOnce(previousMessages.promise);
    vi.mocked(api.messages).mockResolvedValueOnce([message(2)]);
    const store = useSessionStore();
    const pending = store.open("same");
    store.close();
    await store.open("same");
    previousMessages.resolve([message(1)]);
    await pending;
    expect(store.messages.map((item) => item.id)).toEqual([2]);
  });
});
