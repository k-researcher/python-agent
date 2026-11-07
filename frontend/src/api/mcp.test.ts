import { afterEach, expect, it, vi } from "vitest";
import { request } from "./client";
import { joinCommand, mcpApi, splitCommand } from "./mcp";

vi.mock("./client", () => ({ request: vi.fn() }));

afterEach(() => vi.clearAllMocks());

it("splits a command line like a terminal does", () => {
  expect(splitCommand("  npx -y   server ")).toEqual(["npx", "-y", "server"]);
  expect(splitCommand(`node "/Apps/My Tools/server.js" --name 'a b'`)).toEqual([
    "node",
    "/Apps/My Tools/server.js",
    "--name",
    "a b",
  ]);
  expect(splitCommand(`server --label ""`)).toEqual(["server", "--label", ""]);
  expect(splitCommand("")).toEqual([]);
});

it("refuses a command with an open quote", () => {
  expect(() => splitCommand(`node "/Apps/My Tools/server.js`)).toThrow("Не закрыта кавычка");
});

it("joins the parts so that they split back the same", () => {
  const parts = ["node", "/Apps/My Tools/server.js", "", 'say "hi"', "plain"];
  const line = joinCommand(parts);
  expect(line).toBe(`node "/Apps/My Tools/server.js" "" 'say "hi"' plain`);
  expect(splitCommand(line)).toEqual(parts);
});

it("creation leaves out removed secrets and an automatic folder list", async () => {
  vi.mocked(request).mockResolvedValue({ sandbox: true, file_sources: [], servers: [] });
  await mcpApi.create("docs", {
    command: ["server"],
    env: { TOKEN: "value", OLD: null },
    project_ids: [],
    allowed_modes: ["dev"],
    allow_network: false,
    read_paths: null,
  });
  const [path, init] = vi.mocked(request).mock.calls[0];
  expect(path).toBe("/api/mcp/servers");
  const body = JSON.parse(String(init?.body));
  expect(body).toEqual({
    id: "docs",
    command: ["server"],
    env: { TOKEN: "value" },
    project_ids: [],
    allowed_modes: ["dev"],
    allow_network: false,
  });
});

it("an ID with special characters stays one path part", async () => {
  vi.mocked(request).mockResolvedValue({ sandbox: true, file_sources: [], servers: [] });
  await mcpApi.check("a/b");
  expect(vi.mocked(request).mock.calls[0][0]).toBe("/api/mcp/servers/a%2Fb/check");
});

it("HTTP creation sends headers without command fields or removed secrets", async () => {
  await mcpApi.create("remote", {
    url: "https://example.com/mcp", headers: { Authorization: "Bearer secret", DROP: null },
    http_protocol: "modern", project_ids: [], allowed_modes: ["ask"],
  });
  const body = JSON.parse(String(vi.mocked(request).mock.calls[0][1]?.body));
  expect(body).toEqual({ id: "remote", url: "https://example.com/mcp", headers: { Authorization: "Bearer secret" },
    http_protocol: "modern", project_ids: [], allowed_modes: ["ask"] });
});

it("HTTP updates keep null headers so the server removes them", async () => {
  await mcpApi.update("remote", { headers: { DROP: null, SWAP: "new" }, http_protocol: "legacy" });
  expect(JSON.parse(String(vi.mocked(request).mock.calls[0][1]?.body))).toEqual({
    headers: { DROP: null, SWAP: "new" }, http_protocol: "legacy",
  });
});
