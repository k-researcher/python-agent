import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("./api", () => ({
  request: vi.fn(),
  setCsrfToken: vi.fn(),
}));

vi.mock("./draftStorage", () => ({
  clearDrafts: vi.fn(),
}));

import { request, setCsrfToken } from "./api";
import { clearDrafts } from "./draftStorage";

interface Deferred<T> {
  promise: Promise<T>;
  resolve: (value: T) => void;
}

function deferred<T>(): Deferred<T> {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((res) => {
    resolve = res;
  });
  return { promise, resolve };
}

interface AuthModule {
  refreshAuth(): Promise<{ authenticated: boolean; csrf_token?: string }>;
  loginFromFragment(): Promise<{ authenticated: boolean; csrf_token?: string } | null>;
  logout(): Promise<void>;
  clearAuth(): void;
}

const requestMock = vi.mocked(request);
const setCsrfMock = vi.mocked(setCsrfToken);
const clearDraftsMock = vi.mocked(clearDrafts);

let originalHash = "";

beforeEach(() => {
  vi.resetModules();
  vi.clearAllMocks();
  originalHash = window.location.hash;
});

afterEach(() => {
  window.location.hash = originalHash;
  vi.restoreAllMocks();
});

describe("auth lifecycle generation guard", () => {
  it("a refreshAuth that resolves after clearAuth does not install the stale csrf token", async () => {
    const auth = (await import("./auth")) as AuthModule;
    const pending = deferred<{ authenticated: boolean; csrf_token?: string }>();
    requestMock.mockReturnValueOnce(pending.promise);

    const refreshPromise = auth.refreshAuth();
    auth.clearAuth();
    pending.resolve({ authenticated: true, csrf_token: "stale-csrf" });
    await refreshPromise;

    expect(requestMock).toHaveBeenCalledTimes(1);
    expect(requestMock).toHaveBeenCalledWith("/auth/session");
    expect(setCsrfMock).toHaveBeenCalledTimes(1);
    expect(setCsrfMock).toHaveBeenCalledWith("");
    expect(setCsrfMock).not.toHaveBeenCalledWith("stale-csrf");
  });

  it("a logout from the old generation does not clear the new session csrf or drafts", async () => {
    const auth = (await import("./auth")) as AuthModule;
    const logoutDeferred = deferred<void>();
    const refreshDeferred = deferred<{ authenticated: boolean; csrf_token?: string }>();
    requestMock.mockReturnValueOnce(logoutDeferred.promise);
    requestMock.mockReturnValueOnce(refreshDeferred.promise);

    const logoutPromise = auth.logout();
    auth.clearAuth();
    const refreshPromise = auth.refreshAuth();
    refreshDeferred.resolve({ authenticated: true, csrf_token: "new-csrf" });
    await refreshPromise;
    logoutDeferred.resolve(undefined);
    await logoutPromise;

    expect(requestMock).toHaveBeenCalledTimes(2);
    expect(requestMock.mock.calls[0]).toEqual(["/auth/logout", { method: "POST" }]);
    expect(requestMock.mock.calls[1]).toEqual(["/auth/session"]);
    expect(setCsrfMock).toHaveBeenCalledTimes(2);
    expect(setCsrfMock.mock.calls[0]).toEqual([""]);
    expect(setCsrfMock.mock.calls[1]).toEqual(["new-csrf"]);
    expect(clearDraftsMock).toHaveBeenCalledTimes(1);
    expect(requestMock).not.toHaveBeenCalledTimes(3);
  });

  it("a loginFromFragment that resolves after clearAuth does not restore the stale csrf token", async () => {
    window.location.hash = "#login=CODE123";
    const replaceStateSpy = vi.spyOn(history, "replaceState");

    const auth = (await import("./auth")) as AuthModule;
    const pending = deferred<{ authenticated: boolean; csrf_token?: string }>();
    requestMock.mockReturnValueOnce(pending.promise);

    const loginPromise = auth.loginFromFragment();
    auth.clearAuth();
    pending.resolve({ authenticated: true, csrf_token: "stale-csrf" });
    const result = await loginPromise;

    expect(result).toEqual({ authenticated: true, csrf_token: "stale-csrf" });
    expect(requestMock).toHaveBeenCalledTimes(1);
    expect(requestMock).toHaveBeenCalledWith("/auth/login", {
      method: "POST",
      body: JSON.stringify({ code: "CODE123" }),
    });
    expect(replaceStateSpy).toHaveBeenCalledTimes(1);
    expect(replaceStateSpy).toHaveBeenCalledWith(
      null,
      "",
      window.location.pathname + window.location.search,
    );
    expect(setCsrfMock).toHaveBeenCalledTimes(1);
    expect(setCsrfMock).toHaveBeenCalledWith("");
    expect(setCsrfMock).not.toHaveBeenCalledWith("stale-csrf");
  });

  it("a current-generation refreshAuth sets the exact csrf token", async () => {
    const auth = (await import("./auth")) as AuthModule;
    requestMock.mockResolvedValueOnce({ authenticated: true, csrf_token: "exact-token" });

    const state = await auth.refreshAuth();

    expect(state).toEqual({ authenticated: true, csrf_token: "exact-token" });
    expect(requestMock).toHaveBeenCalledTimes(1);
    expect(requestMock).toHaveBeenCalledWith("/auth/session");
    expect(setCsrfMock).toHaveBeenCalledTimes(1);
    expect(setCsrfMock).toHaveBeenCalledWith("exact-token");
  });
});
