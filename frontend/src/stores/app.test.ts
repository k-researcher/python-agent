import { beforeEach, describe, expect, it, vi } from "vitest";
import { createPinia, setActivePinia } from "pinia";
import { flushPromises } from "@vue/test-utils";
import { api } from "../api/client";
import type { Project, ServerEvent, Session } from "../api/types";
import { useAppStore } from "./app";

function project(id: string, name = id): Project {
  return { id, name, root_path: `/root/${id}` };
}

function session(id: string, overrides: Partial<Session> = {}): Session {
  return {
    id,
    parent_id: null,
    mode: "dev",
    llm_profile: "default",
    title: id,
    status: "completed",
    error: null,
    archived: false,
    created_at: "2026-01-01T00:00:00Z",
    updated_at: "2026-01-01T00:00:00Z",
    ...overrides,
    project_id: overrides.project_id ?? "p1",
  };
}

function persistentEvent(type: string, sessionId: string | null, payload: Record<string, unknown>): ServerEvent {
  return { id: 1, type, session_id: sessionId, payload, created_at: "2026-01-01T00:00:00Z" };
}

describe("app store", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    setActivePinia(createPinia());
  });

  it("load() fills config, projects and sessions and picks a default project", async () => {
    vi.spyOn(api, "config").mockResolvedValue({
      llm_configured: true,
      llm_model: "m",
      llm_destination: "http://local",
      knowledge_base_enabled: false,
      network_tools_enabled: true,
      network_allowlist: [],
      api_auth_enabled: true,
      execution_mode: "embedded",
      default_llm_profile: "default",
      llm_profiles: [],
    });
    vi.spyOn(api, "projects").mockResolvedValue([project("p1"), project("p2")]);
    vi.spyOn(api, "sessions").mockResolvedValue([session("s1")]);

    const store = useAppStore();
    await store.load();

    expect(store.projects.map((p) => p.id)).toEqual(["p1", "p2"]);
    expect(store.sessions.map((s) => s.id)).toEqual(["s1"]);
    expect(store.config?.llm_configured).toBe(true);
    expect(store.selectedProjectId).toBe("p1");
  });

  it("load() replaces lists on re-run without duplicating entries", async () => {
    vi.spyOn(api, "config").mockResolvedValue({
      llm_configured: true,
      llm_model: "m",
      llm_destination: "http://local",
      knowledge_base_enabled: false,
      network_tools_enabled: true,
      network_allowlist: [],
      api_auth_enabled: true,
      execution_mode: "embedded",
      default_llm_profile: "default",
      llm_profiles: [],
    });
    const projectsSpy = vi
      .spyOn(api, "projects")
      .mockResolvedValueOnce([project("p1")])
      .mockResolvedValueOnce([project("p1"), project("p2")]);
    vi.spyOn(api, "sessions").mockResolvedValue([]);

    const store = useAppStore();
    await store.load();
    await store.load();

    expect(projectsSpy).toHaveBeenCalledTimes(2);
    expect(store.projects.map((p) => p.id)).toEqual(["p1", "p2"]); // replaced, not appended
  });

  it("load() leaves state untouched when one request fails mid-flight", async () => {
    vi.spyOn(api, "config").mockRejectedValue(new Error("boom"));
    const projectsSpy = vi.spyOn(api, "projects").mockResolvedValue([project("p1")]);
    vi.spyOn(api, "sessions").mockResolvedValue([]);

    const store = useAppStore();
    store.projects = [project("old")];
    await expect(store.load()).rejects.toThrow("boom");

    // No half-written state: the config failure aborts before any slice is assigned.
    expect(projectsSpy).toHaveBeenCalledTimes(1);
    expect(store.projects.map((p) => p.id)).toEqual(["old"]);
    expect(store.config).toBeNull();
  });

  it("createProject selects the new project and reloads the list from the server", async () => {
    vi.spyOn(api, "createProject").mockResolvedValue(project("new"));
    const projectsSpy = vi.spyOn(api, "projects").mockResolvedValue([project("p0"), project("new")]);

    const store = useAppStore();
    await store.createProject("Новый", "/tmp/new");

    expect(store.selectedProjectId).toBe("new");
    expect(projectsSpy).toHaveBeenCalledTimes(1);
    expect(store.projects.map((p) => p.id)).toEqual(["p0", "new"]);
  });

  it("createSession calls the API with mode/profile defaults and refreshes sessions", async () => {
    const createSpy = vi
      .spyOn(api, "createSession")
      .mockResolvedValue(session("sx", { title: "task" }));
    const sessionsSpy = vi.spyOn(api, "sessions").mockResolvedValue([session("sx")]);

    const store = useAppStore();
    await store.createSession({ project_id: "p1", prompt: "task" });

    expect(createSpy).toHaveBeenCalledWith({
      project_id: "p1",
      prompt: "task",
      mode: undefined,
      llm_profile: undefined,
      reasoning_effort: undefined,
      title: undefined,
      auto_start: undefined,
    });
    expect(sessionsSpy).toHaveBeenCalledTimes(1);
    expect(store.sessions.map((s) => s.id)).toEqual(["sx"]);
  });

  it("createSession forwards the chosen mode and profile", async () => {
    const createSpy = vi.spyOn(api, "createSession").mockResolvedValue(session("sx"));
    vi.spyOn(api, "sessions").mockResolvedValue([]);

    const store = useAppStore();
    await store.createSession({
      project_id: "p1",
      prompt: "task",
      mode: "ask",
      llm_profile: "pro",
    });
    expect(createSpy).toHaveBeenCalledWith({
      project_id: "p1",
      prompt: "task",
      mode: "ask",
      llm_profile: "pro",
      reasoning_effort: undefined,
      title: undefined,
      auto_start: undefined,
    });
  });

  describe("sessionTree", () => {
    it("builds parent → children and sorts newest first", () => {
      const store = useAppStore();
      store.sessions = [
        session("child1", { parent_id: "root", updated_at: "2026-01-02T00:00:00Z" }),
        session("root", { updated_at: "2026-01-03T00:00:00Z" }),
        session("solo", { updated_at: "2026-01-01T00:00:00Z" }),
        session("child2", { parent_id: "root", updated_at: "2026-01-04T00:00:00Z" }),
      ];

      const tree = store.sessionTree("p1");
      expect(tree.map((n) => n.session.id)).toEqual(["root", "solo"]);
      const root = tree[0];
      expect(root.children.map((n) => n.session.id)).toEqual(["child2", "child1"]);
    });

    it("hides archived sessions and foreign projects", () => {
      const store = useAppStore();
      store.sessions = [
        session("a1", { project_id: "p1", archived: true }),
        session("a2", { project_id: "p2" }),
        session("a3", { project_id: "p1", parent_id: "a1" }),
      ];
      expect(store.sessionTree("p1")).toEqual([]);
      expect(store.sessionTree("p2").map((n) => n.session.id)).toEqual(["a2"]);
    });

    it("returns an empty tree for a null project and sorts a plain list deterministically", () => {
      const store = useAppStore();
      store.sessions = [session("x"), session("y")];
      expect(store.sessionTree(null)).toEqual([]);
      // Tie on updated_at falls back to id order.
      expect(store.sessionTree("p1").map((n) => n.session.id)).toEqual(["x", "y"]);
    });
  });

  describe("applyEvent", () => {
    it("reloads sessions on session.status", async () => {
      const fresh = session("live", { status: "running" });
      const spy = vi.spyOn(api, "sessions").mockResolvedValueOnce([fresh]);
      const store = useAppStore();
      store.sessions = [session("s1")];

      store.applyEvent(persistentEvent("session.status", "live", { status: "running" }));
      await flushPromises();

      expect(spy).toHaveBeenCalledTimes(1);
      expect(store.sessions.map((s) => s.status)).toEqual(["running"]);
    });

    it.each(["session.created", "session.archived", "session.child_created", "session.deleted"])(
      "%s reloads the sessions list",
      async (type) => {
        const spy = vi.spyOn(api, "sessions").mockResolvedValueOnce([session("fresh")]);
        const store = useAppStore();
        store.sessions = [session("old")];

        store.applyEvent(persistentEvent(type, "fresh", {}));
        await flushPromises();

        expect(spy).toHaveBeenCalledTimes(1);
        expect(store.sessions.map((s) => s.id)).toEqual(["fresh"]);
      },
    );

    it("createProject does not leave selectedProjectId pointing at an absent project when reload fails", async () => {
      vi.spyOn(api, "createProject").mockResolvedValue(project("new"));
      const projectsSpy = vi.spyOn(api, "projects").mockRejectedValue(new Error("boom"));
      const store = useAppStore();
      store.selectedProjectId = "old";

      await expect(store.createProject("Новый", "/tmp/new")).rejects.toThrow("boom");

      // No half-written state: the new project was created but the reload failed, so
      // the previous selection is kept rather than pointing at an absent project.
      expect(store.selectedProjectId).toBe("old");
      expect(projectsSpy).toHaveBeenCalledTimes(1);
      expect(store.projects).toHaveLength(0);
    });

    it("reloads projects on project.created", async () => {
      const spy = vi.spyOn(api, "projects").mockResolvedValueOnce([project("p1"), project("p2")]);
      const store = useAppStore();
      store.projects = [project("p1")];

      store.applyEvent(persistentEvent("project.created", null, { project_id: "p2" }));
      await flushPromises();

      expect(spy).toHaveBeenCalledTimes(1);
      expect(store.projects.map((p) => p.id)).toEqual(["p1", "p2"]);
    });

    it("ignores other project/session events and stream deltas", async () => {
      const sessionsSpy = vi.spyOn(api, "sessions").mockResolvedValue([]);
      const projectsSpy = vi.spyOn(api, "projects").mockResolvedValue([]);
      const store = useAppStore();

      store.applyEvent(persistentEvent("message.created", "s1", { role: "user" }));
      store.applyEvent({
        type: "message.delta",
        session_id: "s1",
        ephemeral: true,
        payload: { stream_id: "st", seq: 1, delta: "x" },
      });
      await flushPromises();

      expect(projectsSpy).not.toHaveBeenCalled();
      expect(sessionsSpy).not.toHaveBeenCalled();
    });
  });
});
