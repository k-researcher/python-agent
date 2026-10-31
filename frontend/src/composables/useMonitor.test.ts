import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { effectScope, nextTick, ref, type EffectScope } from "vue";
import { flushPromises } from "@vue/test-utils";
import { api } from "../api/client";
import type { MonitorRow, ServerEvent } from "../api/types";
import { useMonitor } from "./useMonitor";

const row: MonitorRow = { session_id: "s1", parent_id: null, project_id: "p1", project: "Project",
  title: "Agent", status: "running", model: "default", mode: "dev", action_kind: "user",
  action_text: "Task", duration_seconds: 10, idle_seconds: 2, cost: "0", has_unknown_cost: false };
const scopes: EffectScope[] = [];
function setup(active = false) {
  const enabled = ref(active);
  const scope = effectScope();
  scopes.push(scope);
  const monitor = scope.run(() => useMonitor(enabled))!;
  return { monitor, enabled, scope };
}
function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>(done => { resolve = done; });
  return { promise, resolve };
}
function event(type: string): ServerEvent {
  return { id: 1, session_id: "other-project-session", type, created_at: "", payload: {} };
}
beforeEach(() => {
  vi.useFakeTimers();
  vi.spyOn(api, "monitor").mockResolvedValue([row]);
  vi.spyOn(api, "stop").mockResolvedValue({ stopped: true });
});
afterEach(() => {
  for (const scope of scopes.splice(0)) scope.stop();
  vi.restoreAllMocks();
  vi.useRealTimers();
});
it("loads on opening and refreshes time only while visible", async () => {
  const { monitor, enabled } = setup();
  expect(api.monitor).not.toHaveBeenCalled();
  enabled.value = true;
  await nextTick(); await flushPromises();
  expect(monitor.rows.value).toEqual([row]);
  vi.mocked(api.monitor).mockResolvedValue([{ ...row, duration_seconds: 15 }]);
  await vi.advanceTimersByTimeAsync(5000);
  expect(monitor.rows.value[0]?.duration_seconds).toBe(15);
  enabled.value = false; await nextTick();
  await vi.advanceTimersByTimeAsync(10000);
  expect(api.monitor).toHaveBeenCalledTimes(2);
  expect(monitor.rows.value).toEqual([]);
});
it.each(["session.status", "session.error", "approval.required", "approval.resolved", "approval.cancelled"])(
  "reloads global activity after %s and merges a burst", async (type) => {
    const { monitor } = setup(true); await flushPromises();
    monitor.onEvent(event(type)); monitor.onEvent(event(type));
    await vi.advanceTimersByTimeAsync(150);
    expect(api.monitor).toHaveBeenCalledTimes(2);
  },
);
it("ignores stream deltas and events while closed", async () => {
  const { monitor, enabled } = setup(true); await flushPromises();
  monitor.onEvent(event("reasoning.delta"));
  await vi.advanceTimersByTimeAsync(200);
  expect(api.monitor).toHaveBeenCalledTimes(1);
  enabled.value = false; await nextTick();
  monitor.onEvent(event("session.status"));
  await vi.advanceTimersByTimeAsync(200);
  expect(api.monitor).toHaveBeenCalledTimes(1);
});
it("rejects late replies after closing and reopening the tab", async () => {
  const old = deferred<MonitorRow[]>();
  vi.mocked(api.monitor).mockReturnValueOnce(old.promise);
  const { monitor, enabled } = setup(true);
  enabled.value = false; await nextTick();
  enabled.value = true; await nextTick(); await flushPromises();
  old.resolve([{ ...row, title: "Old login" }]); await flushPromises();
  expect(monitor.rows.value).toEqual([row]);
});
it("aborts the old request when the tab closes", async () => {
  const pending = deferred<MonitorRow[]>();
  vi.mocked(api.monitor).mockReturnValueOnce(pending.promise);
  const { enabled } = setup(true);
  const signal = vi.mocked(api.monitor).mock.calls[0]?.[0];
  expect(signal).toBeInstanceOf(AbortSignal);
  enabled.value = false; await nextTick();
  expect(signal?.aborted).toBe(true);
  pending.resolve([]); await flushPromises();
});
it("allows Stop again after closing and reopening during the old Stop request", async () => {
  const pending = deferred<{ stopped: boolean }>();
  vi.mocked(api.stop).mockReturnValueOnce(pending.promise);
  const { monitor, enabled } = setup(true); await flushPromises();
  const old = monitor.stop("s1");
  enabled.value = false; await nextTick();
  enabled.value = true; await nextTick(); await flushPromises();
  expect(monitor.stopping.value).toEqual([]);
  await monitor.stop("s1");
  expect(api.stop).toHaveBeenCalledTimes(2);
  pending.resolve({ stopped: true }); await old;
  expect(monitor.stopping.value).toEqual([]);
});
it("keeps one request in flight and repeats after an intervening event", async () => {
  const pending = deferred<MonitorRow[]>();
  vi.mocked(api.monitor).mockReturnValueOnce(pending.promise);
  const { monitor } = setup(true);
  monitor.onEvent(event("session.status"));
  await vi.advanceTimersByTimeAsync(150);
  expect(api.monitor).toHaveBeenCalledTimes(1);
  pending.resolve([{ ...row, title: "Stale" }]); await flushPromises();
  expect(api.monitor).toHaveBeenCalledTimes(2);
  expect(monitor.rows.value).toEqual([row]);
});
it("reports failures and recovers on retry", async () => {
  vi.mocked(api.monitor).mockRejectedValueOnce(new Error("offline"));
  const { monitor } = setup(true); await flushPromises();
  expect(monitor.error.value).toContain("Не удалось загрузить");
  await monitor.refresh();
  expect(monitor.error.value).toBe("");
  expect(monitor.rows.value).toEqual([row]);
});
it("stops once and reloads the server state", async () => {
  const pending = deferred<{ stopped: boolean }>();
  vi.mocked(api.stop).mockReturnValue(pending.promise);
  const { monitor } = setup(true); await flushPromises();
  const first = monitor.stop("s1");
  await monitor.stop("s1");
  expect(api.stop).toHaveBeenCalledTimes(1);
  expect(monitor.stopping.value).toEqual(["s1"]);
  vi.mocked(api.monitor).mockResolvedValue([{ ...row, status: "stopped" }]);
  pending.resolve({ stopped: true }); await first;
  expect(monitor.rows.value[0]?.status).toBe("stopped");
  expect(monitor.stopping.value).toEqual([]);
});
it("shows a stop failure and clears the busy button", async () => {
  const { monitor } = setup(true); await flushPromises();
  vi.mocked(api.stop).mockRejectedValueOnce(new Error("offline"));
  await monitor.stop("s1");
  expect(monitor.error.value).toContain("Не удалось остановить");
  expect(monitor.stopping.value).toEqual([]);
});
it("clears timers and ignores replies after disposal", async () => {
  const pending = deferred<MonitorRow[]>();
  vi.mocked(api.monitor).mockReturnValueOnce(pending.promise);
  const { monitor, scope } = setup(true);
  scope.stop();
  pending.resolve([row]); await flushPromises();
  await vi.advanceTimersByTimeAsync(10000);
  expect(monitor.rows.value).toEqual([]);
  expect(api.monitor).toHaveBeenCalledTimes(1);
});
