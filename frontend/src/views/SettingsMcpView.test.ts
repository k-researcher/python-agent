import { flushPromises, mount, type VueWrapper } from "@vue/test-utils";
import { createPinia, setActivePinia } from "pinia";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { createMemoryHistory, createRouter } from "vue-router";
import SettingsMcpView from "./SettingsMcpView.vue";
import { api } from "../api/client";
import { mcpApi, type McpList, type McpServerItem } from "../api/mcp";

vi.mock("../api/client", () => ({ api: { projects: vi.fn() } }));
vi.mock("../api/mcp", async (original) => ({
  ...(await original<typeof import("../api/mcp")>()),
  mcpApi: { list: vi.fn(), create: vi.fn(), update: vi.fn(), remove: vi.fn(), check: vi.fn() },
}));

const server = (id: string, extra: Partial<McpServerItem> = {}): McpServerItem => ({
  id,
  command: ["npx", "-y", `${id}-server`],
  url: null,
  header_keys: [],
  http_protocol: "auto",
  env_keys: [],
  project_ids: [],
  allowed_modes: ["dev"],
  allow_network: false,
  read_paths: ["/opt/tools"],
  enabled: true,
  call_timeout: 120,
  last_check: null,
  updated_at: null,
  ...extra,
});

const listOf = (servers: McpServerItem[], extra: Partial<McpList> = {}): McpList => ({
  sandbox: true,
  file_sources: [],
  servers,
  ...extra,
});

const PROJECTS = [
  { id: "p1", name: "app", root_path: "/work/app" },
  { id: "p2", name: "site", root_path: "/work/site" },
];

async function mountView(list: McpList): Promise<VueWrapper> {
  vi.mocked(api.projects).mockResolvedValue(PROJECTS);
  vi.mocked(mcpApi.list).mockResolvedValue(list);
  const router = createRouter({
    history: createMemoryHistory(),
    routes: [
      { path: "/", name: "home", component: { template: "<div/>" } },
      { path: "/settings/models", name: "settings-models", component: { template: "<div/>" } },
      { path: "/settings/speech", name: "settings-speech", component: { template: "<div/>" } },
      { path: "/settings/skills", name: "settings-skills", component: { template: "<div/>" } },
      { path: "/settings/mcp", name: "settings-mcp", component: SettingsMcpView },
    ],
  });
  await router.push("/settings/mcp");
  const wrapper = mount(SettingsMcpView, { global: { plugins: [router] }, attachTo: document.body });
  await flushPromises();
  return wrapper;
}

const save = (wrapper: VueWrapper) => wrapper.find('[data-testid="mcp-save"]');

beforeEach(() => setActivePinia(createPinia()));
afterEach(() => {
  vi.clearAllMocks();
  document.body.innerHTML = "";
});

it("a card shows the launch, the projects, the roles, the network and the secret names", async () => {
  const wrapper = await mountView(
    listOf([
      server("docs", {
        command: ["node", "/Apps/My Tools/docs.js"],
        env_keys: ["DOCS_TOKEN"],
        project_ids: ["p2"],
        allowed_modes: ["dev", "ask"],
        allow_network: true,
      }),
    ]),
  );
  const card = wrapper.find('[data-server="docs"]');
  expect(card.text()).toContain('node "/Apps/My Tools/docs.js"');
  expect(card.text()).toContain("site · Все роли");
  expect(card.text()).toContain("сеть разрешена");
  expect(card.text()).toContain("Секреты: DOCS_TOKEN");
  expect(card.text()).toContain("Ещё не проверялся");
  expect(wrapper.find('[data-testid="mcp-no-sandbox"]').exists()).toBe(false);
  expect(wrapper.find('[data-testid="mcp-empty"]').exists()).toBe(false);
});

it("without servers the page explains how to add one", async () => {
  const wrapper = await mountView(listOf([], { file_sources: ["files"] }));
  expect(wrapper.find('[data-testid="mcp-empty"]').exists()).toBe(true);
  expect(wrapper.find('[data-testid="mcp-file-sources"]').text()).toContain("files");
});

it("without the OS sandbox the page warns that no server starts", async () => {
  const wrapper = await mountView(listOf([server("docs")], { sandbox: false }));
  expect(wrapper.find('[data-testid="mcp-no-sandbox"]').text()).toContain("не запустятся");
});

it("a failed load can be retried", async () => {
  vi.mocked(api.projects).mockResolvedValue(PROJECTS);
  vi.mocked(mcpApi.list).mockRejectedValueOnce(new Error("сеть"));
  const wrapper = await mountView(listOf([]));
  // The first list call in mountView is replaced by the rejected one above.
  expect(wrapper.find('[role="alert"]').text()).toContain("Не удалось загрузить MCP-серверы: сеть");
  await wrapper.find('[role="alert"] button').trigger("click");
  await flushPromises();
  expect(wrapper.find('[data-testid="mcp-empty"]').exists()).toBe(true);
});

it("a new server is sent with its secret, projects and roles", async () => {
  const wrapper = await mountView(listOf([]));
  await wrapper.find('[data-testid="mcp-add"]').trigger("click");
  await wrapper.find('[data-testid="mcp-id"]').setValue("docs");
  await wrapper.find('[data-testid="mcp-command"]').setValue(`node "/Apps/My Tools/docs.js" --stdio`);
  await wrapper.find('[data-testid="mcp-env-add"]').trigger("click");
  await wrapper.find('[data-testid="mcp-env-key"]').setValue("DOCS_TOKEN");
  await wrapper.find('[data-testid="mcp-env-value"]').setValue("secret-value");
  await wrapper.find('[data-testid="mcp-network"]').trigger("click");
  await wrapper.findAll('input[type="radio"]')[1].setValue(true);
  await wrapper.find('[data-project="p1"]').setValue(true);
  await wrapper.find('[data-profile="ask"]').setValue(true);
  vi.mocked(mcpApi.create).mockResolvedValue(listOf([server("docs")]));

  await wrapper.find('[data-testid="mcp-form"]').trigger("submit");
  await flushPromises();

  expect(mcpApi.create).toHaveBeenCalledWith("docs", {
    command: ["node", "/Apps/My Tools/docs.js", "--stdio"],
    env: { DOCS_TOKEN: "secret-value" },
    project_ids: ["p1"],
    allowed_modes: ["dev", "ask"],
    call_timeout: 120,
    allow_network: true,
    read_paths: null,
  });
  expect(wrapper.find('[data-testid="mcp-form"]').exists()).toBe(false);
  expect(wrapper.find('[data-testid="mcp-status"]').text()).toContain("Сервер «docs» добавлен");
  expect(wrapper.text()).not.toContain("secret-value");
});

it("a taken or wrong ID cannot be saved", async () => {
  const wrapper = await mountView(listOf([server("docs")], { file_sources: ["files"] }));
  await wrapper.find('[data-testid="mcp-add"]').trigger("click");
  await wrapper.find('[data-testid="mcp-command"]').setValue("server");
  const cases: [string, string][] = [
    ["docs", "Сервер с таким ID уже есть"],
    ["files", "уже у сервера из файла настроек"],
    ["Docs", "Строчные латинские буквы"],
  ];
  for (const [id, text] of cases) {
    await wrapper.find('[data-testid="mcp-id"]').setValue(id);
    expect(wrapper.find("#mcp-id-help").text()).toContain(text);
    expect(save(wrapper).attributes("disabled")).toBeDefined();
  }
  await wrapper.find('[data-testid="mcp-id"]').setValue("other");
  expect(save(wrapper).attributes("disabled")).toBeUndefined();
});

it("an open quote, a secret without a value, a relative folder and no roles block saving", async () => {
  const wrapper = await mountView(listOf([]));
  await wrapper.find('[data-testid="mcp-add"]').trigger("click");
  await wrapper.find('[data-testid="mcp-id"]').setValue("docs");
  await wrapper.find('[data-testid="mcp-command"]').setValue(`node "/Apps/My Tools`);
  expect(wrapper.find("#mcp-command-help").text()).toContain("Не закрыта кавычка");
  expect(save(wrapper).attributes("disabled")).toBeDefined();
  await wrapper.find('[data-testid="mcp-command"]').setValue("server");
  expect(save(wrapper).attributes("disabled")).toBeUndefined();

  await wrapper.find('[data-testid="mcp-env-add"]').trigger("click");
  await wrapper.find('[data-testid="mcp-env-key"]').setValue("TOKEN");
  expect(wrapper.text()).toContain("Введите значение переменной TOKEN");
  expect(save(wrapper).attributes("disabled")).toBeDefined();
  await wrapper.find('[data-testid="mcp-env-value"]').setValue("x");

  await wrapper.find('[data-testid="mcp-paths"]').setValue("tools/bin");
  expect(wrapper.text()).toContain("Путь должен быть абсолютным: tools/bin");
  expect(save(wrapper).attributes("disabled")).toBeDefined();
  await wrapper.find('[data-testid="mcp-paths"]').setValue("/opt/tools");

  await wrapper.find('[data-profile="dev"]').setValue(false);
  expect(wrapper.text()).toContain("Выберите хотя бы одну группу ролей");
  expect(save(wrapper).attributes("disabled")).toBeDefined();
  await wrapper.find('[data-profile="dev"]').setValue(true);
  expect(save(wrapper).attributes("disabled")).toBeUndefined();
  expect(mcpApi.create).not.toHaveBeenCalled();
});

it("editing keeps an untouched secret, replaces a typed one and removes a dropped one", async () => {
  const item = server("docs", { env_keys: ["DROP", "KEEP", "SWAP"], project_ids: ["p1"] });
  const wrapper = await mountView(listOf([item]));
  await wrapper.find('[data-testid="mcp-edit"]').trigger("click");
  expect(wrapper.find('[data-testid="mcp-id"]').exists()).toBe(false);
  expect((wrapper.find('[data-testid="mcp-command"]').element as HTMLInputElement).value).toBe("npx -y docs-server");
  const rows = wrapper.findAll('[data-testid="mcp-env-row"]');
  expect(rows.map((row) => (row.find("input").element as HTMLInputElement).value)).toEqual(["DROP", "KEEP", "SWAP"]);
  await rows[0].find("button").trigger("click");
  await rows[2].find('[data-testid="mcp-env-value"]').setValue("new-value");
  await wrapper.find('[data-testid="mcp-paths"]').setValue("/opt/tools\n/opt/more\n");
  vi.mocked(mcpApi.update).mockResolvedValue(listOf([item]));

  await wrapper.find('[data-testid="mcp-form"]').trigger("submit");
  await flushPromises();

  expect(mcpApi.update).toHaveBeenCalledWith("docs", {
    command: ["npx", "-y", "docs-server"],
    env: { DROP: null, SWAP: "new-value" },
    project_ids: ["p1"],
    allowed_modes: ["dev"],
    call_timeout: 120,
    allow_network: false,
    read_paths: ["/opt/tools", "/opt/more"],
  });
  expect(wrapper.find('[data-testid="mcp-status"]').text()).toBe("Сервер «docs» сохранён");
});

it("a save error stays in the dialog", async () => {
  const wrapper = await mountView(listOf([server("docs")]));
  await wrapper.find('[data-testid="mcp-edit"]').trigger("click");
  vi.mocked(mcpApi.update).mockRejectedValue(new Error("Папка открывает данные агента"));
  await wrapper.find('[data-testid="mcp-form"]').trigger("submit");
  await flushPromises();
  expect(wrapper.find('[data-testid="mcp-form-error"]').text()).toContain("Папка открывает данные агента");
  expect(wrapper.find('[data-testid="mcp-form"]').exists()).toBe(true);
});

it("a check shows the tools of a working server", async () => {
  const wrapper = await mountView(listOf([server("docs")]));
  let finish: (value: McpList) => void = () => undefined;
  vi.mocked(mcpApi.check).mockReturnValue(new Promise((resolve) => (finish = resolve)));

  await wrapper.find('[data-testid="mcp-check-run"]').trigger("click");
  expect(wrapper.find('[data-testid="mcp-check"]').text()).toBe("Запускаю сервер…");
  expect(wrapper.find('[data-testid="mcp-check-run"]').attributes("disabled")).toBeDefined();
  finish(
    listOf([
      server("docs", {
        last_check: {
          ok: true,
          server: { name: "docs", version: "1.2" },
          tools: [{ name: "search", description: "Поиск" }, { name: "read" }],
          tool_count: 2,
        },
      }),
    ]),
  );
  await flushPromises();

  expect(mcpApi.check).toHaveBeenCalledWith("docs");
  expect(wrapper.find('[data-testid="mcp-check"]').text()).toBe("Каталог получен · 2 инструмента · docs 1.2");
  expect(wrapper.text()).toContain("но не вызывает их и не проверяет доступ к их внешним сервисам");
  expect(wrapper.find('[data-testid="mcp-details"]').text()).toContain("search");
  // The result line folds the details away.
  await wrapper.find('[data-testid="mcp-check"]').trigger("click");
  expect(wrapper.find('[data-testid="mcp-details"]').exists()).toBe(false);
});

it("a failed check opens the error output", async () => {
  const wrapper = await mountView(listOf([server("docs")]));
  vi.mocked(mcpApi.check).mockResolvedValue(
    listOf([server("docs", { last_check: { ok: false, error: "Сервер завершился", stderr: "module not found" } })]),
  );
  await wrapper.find('[data-testid="mcp-check-run"]').trigger("click");
  await flushPromises();
  expect(wrapper.find('[data-testid="mcp-check"]').classes()).toContain("ui-bad");
  expect(wrapper.find('[data-testid="mcp-check"]').text()).toBe("Сервер завершился");
  expect(wrapper.find('[data-testid="mcp-details"] pre').text()).toBe("module not found");
});

it("the switch turns a server off", async () => {
  const wrapper = await mountView(listOf([server("docs")]));
  vi.mocked(mcpApi.update).mockResolvedValue(listOf([server("docs", { enabled: false })]));
  await wrapper.find('[data-server="docs"] [role="switch"]').trigger("click");
  await flushPromises();
  expect(mcpApi.update).toHaveBeenCalledWith("docs", { enabled: false });
  expect(wrapper.find('[data-server="docs"]').classes()).toContain("ui-card--off");
});

it("removal waits for the confirmation", async () => {
  const wrapper = await mountView(listOf([server("docs")]));
  await wrapper.find('[data-testid="mcp-remove"]').trigger("click");
  const cancel = wrapper.findAll(".ui-btn--text").find((button) => button.text() === "Отмена" && button.isVisible());
  await cancel?.trigger("click");
  expect(wrapper.find('[data-testid="mcp-remove-confirm"]').exists()).toBe(false);
  expect(mcpApi.remove).not.toHaveBeenCalled();

  vi.mocked(mcpApi.remove).mockResolvedValue(listOf([]));
  await wrapper.find('[data-testid="mcp-remove"]').trigger("click");
  await wrapper.find('[data-testid="mcp-remove-confirm"]').trigger("click");
  await flushPromises();
  expect(mcpApi.remove).toHaveBeenCalledWith("docs");
  expect(wrapper.find('[data-testid="mcp-empty"]').exists()).toBe(true);
  expect(wrapper.find('[data-testid="mcp-status"]').text()).toBe("Сервер «docs» удалён");
});

const httpServer = (extra: Partial<McpServerItem> = {}) => server("remote", {
  command: [], url: "https://example.com/mcp", header_keys: ["Authorization"],
  read_paths: [], allow_network: false, ...extra,
});

it("leaving ID does not change the launch help while the transport button is clicked", async () => {
  const wrapper = await mountView(listOf([]));
  await wrapper.find('[data-testid="mcp-add"]').trigger("click");
  const help = wrapper.find("#mcp-command-help").text();
  await wrapper.find('[data-testid="mcp-id"]').setValue("remote");
  await wrapper.find('[data-testid="mcp-id"]').trigger("blur");
  expect(wrapper.find("#mcp-command-help").text()).toBe(help);
  await wrapper.find('[aria-label="Подключение"] button[aria-checked="false"]').trigger("click");
  expect(wrapper.find('[data-testid="mcp-url"]').exists()).toBe(true);
});

it("an HTTP card shows its address, header names and checked protocol without a sandbox warning", async () => {
  const wrapper = await mountView(listOf([httpServer({ last_check: {
    ok: true, protocol: "2026-07-28", tool_count: 1,
  } })], { sandbox: false }));
  const card = wrapper.find('[data-server="remote"]');
  expect(card.find('.ui-badge').text()).toBe("HTTP");
  expect(card.text()).toContain("https://example.com/mcp");
  expect(card.text()).toContain("Заголовки: Authorization");
  expect(card.text()).toContain("2026-07-28");
  expect(card.text()).not.toContain("без сети");
  expect(wrapper.find('[data-testid="mcp-no-sandbox"]').exists()).toBe(false);
});

it("a new HTTP server sends only address fields and the selected protocol", async () => {
  const wrapper = await mountView(listOf([], { sandbox: false }));
  await wrapper.find('[data-testid="mcp-add"]').trigger("click");
  await wrapper.find('[data-testid="mcp-id"]').setValue("remote");
  await wrapper.find('[data-testid="mcp-command"]').setValue("unused");
  await wrapper.find('[data-testid="mcp-env-add"]').trigger("click");
  await wrapper.find('[data-testid="mcp-env-key"]').setValue("UNUSED");
  await wrapper.find('[data-testid="mcp-env-value"]').setValue("unused-secret");
  await wrapper.find('[aria-label="Подключение"] button[aria-checked="false"]').trigger("click");
  expect(wrapper.find('[data-testid="mcp-command"]').exists()).toBe(false);
  expect(wrapper.find('[data-testid="mcp-network"]').exists()).toBe(false);
  expect(wrapper.find('[data-testid="mcp-paths"]').exists()).toBe(false);
  expect(wrapper.find('[data-testid="mcp-no-sandbox"]').exists()).toBe(false);
  expect(wrapper.findAll('[data-testid="mcp-env-row"]')).toHaveLength(0);
  expect(save(wrapper).attributes("disabled")).toBeDefined();
  await wrapper.find('[data-testid="mcp-url"]').setValue("https://example.com/mcp");
  await wrapper.find('[data-testid="mcp-env-add"]').trigger("click");
  await wrapper.find('[data-testid="mcp-env-key"]').setValue("Authorization");
  expect(save(wrapper).attributes("disabled")).toBeDefined();
  await wrapper.find('[data-testid="mcp-env-value"]').setValue("Bearer secret");
  expect(wrapper.find('[data-testid="mcp-env-value"]').attributes("type")).toBe("password");
  const protocols = wrapper.findAll('[aria-label="Протокол HTTP"] button');
  expect(protocols[0].attributes("aria-checked")).toBe("true");
  await protocols[1].trigger("click");
  expect(protocols[1].attributes("aria-checked")).toBe("true");
  vi.mocked(mcpApi.create).mockResolvedValue(listOf([httpServer()]));
  await wrapper.find('[data-testid="mcp-form"]').trigger("submit");
  await flushPromises();
  expect(mcpApi.create).toHaveBeenCalledWith("remote", {
    url: "https://example.com/mcp", headers: { Authorization: "Bearer secret" },
    http_protocol: "modern", project_ids: [], allowed_modes: ["dev"], call_timeout: 120,
  });
  expect(wrapper.text()).not.toContain("Bearer secret");
});

it("editing HTTP keeps untouched headers, replaces and removes secrets and keeps the transport", async () => {
  const item = httpServer({ header_keys: ["DROP", "KEEP", "SWAP"], http_protocol: "legacy" });
  const wrapper = await mountView(listOf([item]));
  await wrapper.find('[data-testid="mcp-edit"]').trigger("click");
  expect(wrapper.findAll('[aria-label="Подключение"] button').every(button => button.attributes("disabled") !== undefined)).toBe(true);
  const rows = wrapper.findAll('[data-testid="mcp-env-row"]');
  expect(rows.every(row => (row.findAll("input")[1].element as HTMLInputElement).value === "")).toBe(true);
  await rows[0].find("button").trigger("click");
  await rows[2].findAll("input")[1].setValue("replacement");
  await wrapper.find('[data-testid="mcp-url"]').setValue("https://other.example.com/mcp");
  vi.mocked(mcpApi.update).mockResolvedValue(listOf([item]));
  await wrapper.find('[data-testid="mcp-form"]').trigger("submit");
  await flushPromises();
  expect(mcpApi.update).toHaveBeenCalledWith("remote", {
    url: "https://other.example.com/mcp", headers: { DROP: null, SWAP: "replacement" },
    http_protocol: "legacy", project_ids: [], allowed_modes: ["dev"], call_timeout: 120,
  });
});

it("an invalid URL and duplicate header names block HTTP saving", async () => {
  const wrapper = await mountView(listOf([httpServer({ header_keys: [] })]));
  await wrapper.find('[data-testid="mcp-edit"]').trigger("click");
  for (const url of ["invalid", "ftp://example.com/mcp", "https://user:pass@example.com/mcp", "https://example.com/#secret"]) {
    await wrapper.find('[data-testid="mcp-url"]').setValue(url);
    expect(save(wrapper).attributes("disabled")).toBeDefined();
  }
  await wrapper.find('[data-testid="mcp-url"]').setValue("https://example.com/mcp");
  for (const key of ["X-Key", "x-key"]) {
    await wrapper.find('[data-testid="mcp-env-add"]').trigger("click");
    const row = wrapper.findAll('[data-testid="mcp-env-row"]').at(-1)!;
    await row.findAll("input")[0].setValue(key);
    await row.findAll("input")[1].setValue("value");
  }
  expect(wrapper.text()).toContain("Имена заголовков повторяются");
  await wrapper.find('[data-testid="mcp-form"]').trigger("submit");
  expect(mcpApi.update).not.toHaveBeenCalled();
});

it("the wait for a tool is kept, limited and sent with the server", async () => {
  const wrapper = await mountView(listOf([server("docs", { call_timeout: 300 })]));
  await wrapper.find('[data-testid="mcp-edit"]').trigger("click");
  const field = wrapper.find('[data-testid="mcp-timeout"]');
  expect((field.element as HTMLInputElement).value).toBe("300");
  for (const value of ["4", "601", "2.5", ""]) {
    await field.setValue(value);
    expect(save(wrapper).attributes("disabled")).toBeDefined();
    expect(wrapper.find("#mcp-timeout-help").classes()).toContain("ui-field__help--error");
  }
  await field.setValue("90");
  expect(save(wrapper).attributes("disabled")).toBeUndefined();
  await wrapper.find('[data-testid="mcp-form"]').trigger("submit");
  await flushPromises();
  expect(vi.mocked(mcpApi.update).mock.calls[0][1].call_timeout).toBe(90);
});
