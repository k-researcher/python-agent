import { afterEach, expect, it, vi } from "vitest";
import { modelSettings } from "./modelSettings";
import { setCsrfToken } from "./client";

afterEach(() => { setCsrfToken(""); vi.unstubAllGlobals(); });

it("uses distinct delete and reset routes and carries the expected version", async () => {
  setCsrfToken("csrf-value");
  const fetcher = vi.fn().mockResolvedValue({ ok: true, status: 200, json: async () => ({}) });
  vi.stubGlobal("fetch", fetcher);
  await modelSettings.removeProvider("team/gateway", { checksum: "version-1" });
  expect(fetcher).toHaveBeenLastCalledWith("/api/settings/models/providers/team/gateway", expect.objectContaining({
    method: "DELETE", headers: expect.objectContaining({ "If-Match": "version-1", "X-Agent-CSRF": "csrf-value" }),
  }));
  await modelSettings.resetProvider("team/gateway", { checksum: "version-2" });
  expect(fetcher).toHaveBeenLastCalledWith("/api/settings/models/providers/team/gateway/reset", expect.objectContaining({
    method: "POST", headers: expect.objectContaining({ "If-Match": "version-2" }),
  }));
  await modelSettings.resetModel("group/chat", { checksum: "version-2" });
  expect(fetcher).toHaveBeenLastCalledWith("/api/settings/models/models/group/chat/reset", expect.objectContaining({ method: "POST" }));
});

it("marks creation so an occupied ID cannot be overwritten", async () => {
  const fetcher = vi.fn().mockResolvedValue({ ok: true, status: 200, json: async () => ({}) });
  vi.stubGlobal("fetch", fetcher);
  await modelSettings.putModel("chat", { provider: "gateway", model: "chat" }, { checksum: "version-1", create: true });
  expect(fetcher).toHaveBeenLastCalledWith("/api/settings/models/models/chat", expect.objectContaining({
    method: "PUT", headers: expect.objectContaining({ "If-Match": "version-1", "If-None-Match": "*" }),
  }));
});
