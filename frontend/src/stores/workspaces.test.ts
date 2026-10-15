import { beforeEach, describe, expect, it, vi } from "vitest";
import { createPinia, setActivePinia } from "pinia";
import { watch } from "vue";
import * as api from "../api/workspaces";
import { useWorkspaceStore } from "./workspaces";

interface DeferredPromise<T> {
  promise: Promise<T>;
  resolve: (value: T | PromiseLike<T>) => void;
  reject: (reason?: unknown) => void;
}

function createDeferred<T>(): DeferredPromise<T> {
  let resolve!: (value: T | PromiseLike<T>) => void;
  let reject!: (reason?: unknown) => void;
  const promise = new Promise<T>((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
}

function createMockWorkspace(id: string, name: string, role: api.WorkspaceRole = "admin"): api.WorkspaceSummary {
  return {
    id,
    name,
    role,
    capabilities: {
      view_project: true,
      run_agent: true,
      manage_members: true,
      manage_workspace: true,
      manage_models: false,
    },
  };
}

function createMemberWorkspace(id: string, name: string): api.WorkspaceSummary {
  return {
    id,
    name,
    role: "member",
    capabilities: {
      view_project: true,
      run_agent: true,
      manage_members: false,
      manage_workspace: false,
      manage_models: false,
    },
  };
}

describe("workspaces store", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    setActivePinia(createPinia());
  });

  it("invalidates scope before publishing reset state", async () => {
    vi.spyOn(api, "listWorkspaces").mockResolvedValue([createMemberWorkspace("w1", "W1")]);
    const store = useWorkspaceStore();
    await store.load();
    const generation = store.scopeGeneration;
    let observedGeneration = -1;
    const unwatch = watch(() => store.workspaces, () => {
      observedGeneration = store.scopeGeneration;
    }, { flush: "sync" });
    store.reset();
    unwatch();
    expect(observedGeneration).toBe(generation + 1);
  });

  it("invalidates scope before publishing failed reload state", async () => {
    vi.spyOn(api, "listWorkspaces").mockResolvedValueOnce([createMemberWorkspace("w1", "W1")])
      .mockRejectedValueOnce(new Error("Access denied"));
    const store = useWorkspaceStore();
    await store.load();
    const generation = store.scopeGeneration;
    let observedGeneration = -1;
    const unwatch = watch(() => store.workspaces, () => {
      observedGeneration = store.scopeGeneration;
    }, { flush: "sync" });
    await store.load();
    unwatch();
    expect(observedGeneration).toBe(generation + 1);
  });

  it("first selection invalidates the previously empty scope", async () => {
    vi.spyOn(api, "listWorkspaces").mockResolvedValue([createMemberWorkspace("w1", "W1")]);
    const store = useWorkspaceStore();
    await store.load();
    expect(store.scopeGeneration).toBe(1);
    expect(store.selectedWorkspace?.role).toBe("member");
  });

  it("preserves a valid manual selection and rejects unknown ids", async () => {
    vi.spyOn(api, "listWorkspaces").mockResolvedValue([
      createMemberWorkspace("w1", "W1"), createMemberWorkspace("w2", "W2"),
    ]);
    const store = useWorkspaceStore();
    await store.load();
    expect(store.selectWorkspace("w2")).toBe(true);
    const generation = store.scopeGeneration;
    expect(store.selectWorkspace("unknown")).toBe(false);
    await store.load();
    expect(store.selectedWorkspaceId).toBe("w2");
    expect(store.scopeGeneration).toBe(generation);
  });

  it("removes access on empty reload", async () => {
    vi.spyOn(api, "listWorkspaces").mockResolvedValueOnce([createMemberWorkspace("w1", "W1")])
      .mockResolvedValueOnce([]);
    const store = useWorkspaceStore();
    await store.load();
    const generation = store.scopeGeneration;
    await store.load();
    expect(store.selectedWorkspace).toBeNull();
    expect(store.selectedWorkspaceId).toBeNull();
    expect(store.scopeGeneration).toBe(generation + 1);
  });

  it("clears old capabilities on current request failure", async () => {
    vi.spyOn(api, "listWorkspaces").mockResolvedValueOnce([createMemberWorkspace("w1", "W1")])
      .mockRejectedValueOnce(new Error("Access denied"));
    const store = useWorkspaceStore();
    await store.load();
    const generation = store.scopeGeneration;
    await store.load();
    expect(store.workspaces).toEqual([]);
    expect(store.selectedWorkspace).toBeNull();
    expect(store.error).toBe("Access denied");
    expect(store.loading).toBe(false);
    expect(store.scopeGeneration).toBe(generation + 1);
  });

  it("stale success cannot clear loading for the pending latest request", async () => {
    const first = createDeferred<api.WorkspaceSummary[]>();
    const second = createDeferred<api.WorkspaceSummary[]>();
    vi.spyOn(api, "listWorkspaces").mockReturnValueOnce(first.promise).mockReturnValueOnce(second.promise);
    const store = useWorkspaceStore();
    const firstLoad = store.load();
    const secondLoad = store.load();
    first.resolve([createMemberWorkspace("w1", "W1")]);
    await firstLoad;
    expect(store.workspaces).toEqual([]);
    expect(store.loading).toBe(true);
    second.resolve([createMemberWorkspace("w2", "W2")]);
    await secondLoad;
    expect(store.selectedWorkspaceId).toBe("w2");
    expect(store.loading).toBe(false);
  });

  it("load() handles successful data and maintains selection", async () => {
    const workspace1 = createMockWorkspace("w1", "W1");
    const workspace2 = createMockWorkspace("w2", "W2");
    vi.spyOn(api, "listWorkspaces").mockResolvedValue([workspace1, workspace2]);

    const store = useWorkspaceStore();
    await store.load();

    expect(store.workspaces).toEqual([workspace1, workspace2]);
    expect(store.selectedWorkspaceId).toBe("w1");
    expect(store.loading).toBe(false);
  });

  it("load() invalidates scope on role change", async () => {
    const workspace1 = createMockWorkspace("w1", "W1", "admin");
    const workspace1Member = createMemberWorkspace("w1", "W1");

    vi.spyOn(api, "listWorkspaces")
      .mockResolvedValueOnce([workspace1])
      .mockResolvedValueOnce([workspace1Member]);

    const store = useWorkspaceStore();
    await store.load();
    const scopeAfterFirst = store.scopeGeneration;

    expect(store.selectedWorkspaceId).toBe("w1");
    expect(store.selectedWorkspace?.role).toBe("admin");

    await store.load();
    expect(store.scopeGeneration).toBe(scopeAfterFirst + 1);
    expect(store.selectedWorkspace?.role).toBe("member");
  });

  it.each(["view_project", "run_agent", "manage_members", "manage_workspace", "manage_models"] as const)("load() invalidates scope on %s change", async (capability) => {
    const workspace1 = createMockWorkspace("w1", "W1", "admin");
    const workspace1Limited = {
      ...workspace1,
      capabilities: { ...workspace1.capabilities, [capability]: !workspace1.capabilities[capability] }
    };

    vi.spyOn(api, "listWorkspaces")
      .mockResolvedValueOnce([workspace1])
      .mockResolvedValueOnce([workspace1Limited]);

    const store = useWorkspaceStore();
    await store.load();
    const scopeAfterFirst = store.scopeGeneration;

    await store.load();
    expect(store.scopeGeneration).toBe(scopeAfterFirst + 1);
    expect(store.selectedWorkspace?.capabilities[capability]).toBe(!workspace1.capabilities[capability]);
  });

  it("load() does NOT invalidate scope on name change", async () => {
    const workspace1 = createMockWorkspace("w1", "W1");
    const workspace1Renamed = createMockWorkspace("w1", "W1 Renamed");

    vi.spyOn(api, "listWorkspaces")
      .mockResolvedValueOnce([workspace1])
      .mockResolvedValueOnce([workspace1Renamed]);

    const store = useWorkspaceStore();
    await store.load();
    const scopeAfterFirst = store.scopeGeneration;

    await store.load();
    expect(store.scopeGeneration).toBe(scopeAfterFirst);
    expect(store.selectedWorkspace?.name).toBe("W1 Renamed");
  });

  it("load() invalidates scope when workspace disappears", async () => {
    const workspace1 = createMockWorkspace("w1", "W1");
    const workspace2 = createMockWorkspace("w2", "W2");

    vi.spyOn(api, "listWorkspaces")
      .mockResolvedValueOnce([workspace1, workspace2])
      .mockResolvedValueOnce([workspace2]);

    const store = useWorkspaceStore();
    await store.load();
    store.selectWorkspace("w1");
    const scopeAfterSelection = store.scopeGeneration;

    await store.load();
    expect(store.selectedWorkspaceId).toBe("w2");
    expect(store.scopeGeneration).toBe(scopeAfterSelection + 1);
  });

  it("selectWorkspace() returns true for same id and does not increment scope", async () => {
    const workspace1 = createMockWorkspace("w1", "W1");
    vi.spyOn(api, "listWorkspaces").mockResolvedValue([workspace1]);

    const store = useWorkspaceStore();
    await store.load();
    const initialScope = store.scopeGeneration;

    expect(store.selectWorkspace("w1")).toBe(true);
    expect(store.scopeGeneration).toBe(initialScope);
  });

  it("reset() is synchronous and cancels pending loads via requestId", async () => {
    const deferred = createDeferred<api.WorkspaceSummary[]>();
    vi.spyOn(api, "listWorkspaces").mockReturnValue(deferred.promise);

    const store = useWorkspaceStore();
    const loadPromise = store.load();

    expect(store.loading).toBe(true);

    store.reset();

    expect(store.loading).toBe(false);
    expect(store.workspaces).toEqual([]);
    expect(store.selectedWorkspaceId).toBeNull();

    deferred.resolve([createMockWorkspace("w1", "W1")]);
    await loadPromise;

    expect(store.workspaces).toEqual([]);
    expect(store.selectedWorkspaceId).toBeNull();
  });

  it("latest request wins: second successful response overwrites first", async () => {
    const deferred1 = createDeferred<api.WorkspaceSummary[]>();
    const deferred2 = createDeferred<api.WorkspaceSummary[]>();

    vi.spyOn(api, "listWorkspaces")
      .mockReturnValueOnce(deferred1.promise)
      .mockReturnValueOnce(deferred2.promise);

    const store = useWorkspaceStore();
    const firstLoad = store.load();
    const secondLoad = store.load();

    deferred2.resolve([createMockWorkspace("w2", "W2")]);
    await secondLoad;
    expect(store.workspaces[0].id).toBe("w2");

    deferred1.resolve([createMockWorkspace("w1", "W1")]);
    await firstLoad;

    expect(store.workspaces[0].id).toBe("w2");
  });

  it("stale reject does not corrupt state after successful load", async () => {
    const deferred1 = createDeferred<api.WorkspaceSummary[]>();
    const deferred2 = createDeferred<api.WorkspaceSummary[]>();

    vi.spyOn(api, "listWorkspaces")
      .mockReturnValueOnce(deferred1.promise)
      .mockReturnValueOnce(deferred2.promise);

    const store = useWorkspaceStore();
    const firstLoad = store.load();
    const secondLoad = store.load();

    deferred2.resolve([createMockWorkspace("w2", "W2")]);
    await secondLoad;
    expect(store.error).toBeNull();
    expect(store.workspaces[0].id).toBe("w2");

    deferred1.reject(new Error("Old error"));
    await firstLoad;

    expect(store.workspaces[0].id).toBe("w2");
    expect(store.error).toBeNull();
  });

  it("loading remains true while actual request is pending", async () => {
    const deferred = createDeferred<api.WorkspaceSummary[]>();
    vi.spyOn(api, "listWorkspaces").mockReturnValue(deferred.promise);

    const store = useWorkspaceStore();
    const loadPromise = store.load();

    expect(store.loading).toBe(true);

    deferred.resolve([]);
    await loadPromise;
    expect(store.loading).toBe(false);
  });
});
