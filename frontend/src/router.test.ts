import { describe, expect, it, vi, beforeEach, afterEach } from "vitest";
import { ApiError } from "./api/client";
import { confirmAuthenticated, ensureAuthenticated, resetAuthCache, router } from "./router";

// The router imports refreshAuth from ./auth; mock it so the guard tests the
// caching/redirect logic without hitting a real network.
vi.mock("./auth", () => ({
  refreshAuth: vi.fn(),
}));

import { refreshAuth } from "./auth";

const mockedRefresh = vi.mocked(refreshAuth);

async function navigate(to: string): Promise<void> {
  try {
    await router.push(to);
  } catch {
    // A guard redirect rejects the original push but still updates currentRoute.
  }
  await router.isReady();
}

beforeEach(async () => {
  resetAuthCache();
  mockedRefresh.mockReset();
  // Start each test from the always-allowed login route so that a protected-route
  // navigation actually re-runs the guard (a navigate to the same path the
  // previous test left us on would be treated as a no-op duplicate).
  try {
    await router.push("/login");
  } catch {
    // ignore
  }
  await router.isReady();
});

afterEach(() => {
  vi.restoreAllMocks();
});

describe("router auth guard", () => {
  it("a confirmed sign-in preserves an initial protected navigation", async () => {
    let finishCheck!: (state: { authenticated: boolean }) => void;
    mockedRefresh.mockReturnValueOnce(new Promise((resolve) => { finishCheck = resolve; }));
    const pendingNavigation = navigate("/s/startup");
    await vi.waitFor(() => expect(mockedRefresh).toHaveBeenCalledOnce());
    confirmAuthenticated();
    finishCheck({ authenticated: true });
    await pendingNavigation;
    expect(router.currentRoute.value.fullPath).toBe("/s/startup");
    await expect(ensureAuthenticated()).resolves.toBe(true);
    expect(mockedRefresh).toHaveBeenCalledOnce();
  });

  it("an old refusal cannot override a newly confirmed sign-in", async () => {
    let rejectCheck!: (cause: unknown) => void;
    mockedRefresh.mockReturnValueOnce(new Promise((_resolve, reject) => { rejectCheck = reject; }));
    const pendingCheck = ensureAuthenticated();
    confirmAuthenticated();
    rejectCheck(new ApiError("expired", 401));
    await expect(pendingCheck).resolves.toBe(true);
    await expect(ensureAuthenticated()).resolves.toBe(true);
    expect(mockedRefresh).toHaveBeenCalledOnce();
  });

  it("logout clears a confirmed sign-in before an old response completes", async () => {
    let finishCheck!: (state: { authenticated: boolean }) => void;
    mockedRefresh.mockReturnValueOnce(new Promise((resolve) => { finishCheck = resolve; }));
    const pendingCheck = ensureAuthenticated();
    confirmAuthenticated();
    resetAuthCache();
    finishCheck({ authenticated: true });
    await expect(pendingCheck).resolves.toBe(false);
    mockedRefresh.mockResolvedValueOnce({ authenticated: false });
    await expect(ensureAuthenticated()).resolves.toBe(false);
    expect(mockedRefresh).toHaveBeenCalledTimes(2);
  });

  it("an old successful check cannot restore authorization after reset", async () => {
    let finishCheck!: (state: { authenticated: boolean }) => void;
    mockedRefresh.mockReturnValueOnce(new Promise((resolve) => { finishCheck = resolve; }));
    const previous = ensureAuthenticated();
    resetAuthCache();
    mockedRefresh.mockResolvedValueOnce({ authenticated: false });
    await expect(ensureAuthenticated()).resolves.toBe(false);
    finishCheck({ authenticated: true });
    await expect(previous).resolves.toBe(false);
    mockedRefresh.mockResolvedValueOnce({ authenticated: false });
    await expect(ensureAuthenticated()).resolves.toBe(false);
    expect(mockedRefresh).toHaveBeenCalledTimes(3);
  });

  it("an old rejection cannot invalidate the next authenticated scope", async () => {
    let rejectCheck!: (cause: unknown) => void;
    mockedRefresh.mockReturnValueOnce(new Promise((_resolve, reject) => { rejectCheck = reject; }));
    const previous = ensureAuthenticated();
    resetAuthCache();
    mockedRefresh.mockResolvedValueOnce({ authenticated: true });
    await expect(ensureAuthenticated()).resolves.toBe(true);
    rejectCheck(new ApiError("expired", 401));
    await expect(previous).resolves.toBe(true);
    await expect(ensureAuthenticated()).resolves.toBe(true);
    expect(mockedRefresh).toHaveBeenCalledTimes(2);
  });

  it("lets an authenticated user reach a protected route", async () => {
    mockedRefresh.mockResolvedValue({ authenticated: true });
    await navigate("/s/1");
    expect(router.currentRoute.value.name).toBe("session");
  });

  it("redirects to login on a confirmed refusal (authenticated:false)", async () => {
    mockedRefresh.mockResolvedValue({ authenticated: false });
    await navigate("/s/1");
    expect(router.currentRoute.value.name).toBe("login");
    expect(router.currentRoute.value.query.next).toBe("/s/1");
    expect(mockedRefresh).toHaveBeenCalled();
  });

  it("redirects to login on a confirmed 401 even when an empty cache", async () => {
    mockedRefresh.mockRejectedValue(new ApiError("unauthorized", 401));
    await navigate("/s/1");
    expect(router.currentRoute.value.name).toBe("login");
  });

  it("does NOT treat a transient (non-401) error as logout", async () => {
    mockedRefresh.mockRejectedValue(new ApiError("server error", 500));
    await navigate("/s/1");
    // A 5xx must not bounce an already-authenticated user to the login page.
    expect(router.currentRoute.value.name).not.toBe("login");
  });

  it("logout then a protected route lands on login (stale true is dropped)", async () => {
    // Signed in: cache becomes true and a protected route is allowed.
    mockedRefresh.mockResolvedValue({ authenticated: true });
    await navigate("/s/1");
    expect(router.currentRoute.value.name).toBe("session");
    // Logout clears the cached `true`…
    resetAuthCache();
    // …and the next check finds the user is confirmed logged out.
    mockedRefresh.mockResolvedValue({ authenticated: false });
    await navigate("/s/2");
    expect(router.currentRoute.value.name).toBe("login");
    // After the reset the guard really re-checked (did not short-circuit on stale true).
    expect(mockedRefresh).toHaveBeenCalledTimes(2);
  });

  it("resetAuthCache clears a cached failed check so a later login is honoured", async () => {
    // First check: confirmed refusal caches a negative result.
    mockedRefresh.mockResolvedValue({ authenticated: false });
    await expect(ensureAuthenticated()).resolves.toBe(false);
    // The user logs in (server now says yes); the cache must be reset.
    resetAuthCache();
    mockedRefresh.mockResolvedValue({ authenticated: true });
    await expect(ensureAuthenticated()).resolves.toBe(true);
  });

  it("is not short-circuited by a stale positive cache after a confirmed 401", async () => {
    mockedRefresh.mockResolvedValue({ authenticated: true });
    await expect(ensureAuthenticated()).resolves.toBe(true);
    // Confirmed server rejection while the app is open (e.g. realtime onUnauthorized).
    resetAuthCache();
    mockedRefresh.mockRejectedValue(new ApiError("unauthorized", 401));
    await expect(ensureAuthenticated()).resolves.toBe(false);
  });
});
