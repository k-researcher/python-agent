import { expect, type Page, type TestInfo } from "@playwright/test";

export const BASE_URL =
  process.env.UI_E2E_BASE_URL ?? "http://127.0.0.1:8097";

export const widths = [390, 760, 1100, 1440];

export function themeOf(info: TestInfo): "light" | "dark" {
  return info.project.name.endsWith("-dark") ? "dark" : "light";
}

export function viewportWidthOf(info: TestInfo): number {
  const name = info.project.name;
  if (name.startsWith("desktop")) return 1440;
  if (name.startsWith("laptop")) return 1100;
  if (name.startsWith("tablet")) return 760;
  return 390;
}

export interface PanelState {
  left: boolean;
  right: boolean;
}

export function panelStateForWidth(width: number): PanelState {
  if (width >= 1440) return { left: true, right: true };
  if (width >= 1100) return { left: true, right: false };
  return { left: false, right: false };
}

export const mockConfig = {
  llm_configured: true,
  llm_model: "model-a",
  llm_destination: "local",
  knowledge_base_enabled: true,
  network_tools_enabled: true,
  network_allowlist: ["example.com"],
  api_auth_enabled: true,
  execution_mode: "embedded",
  default_llm_profile: "default",
  llm_profiles: [
    { name: "default", model: "model-a", destination: "local", configured: true, reasoning_efforts: ["low", "high"] },
    { name: "fast", model: "model-b", destination: "local", configured: true, reasoning_efforts: ["medium"] },
  ],
};

export const mockProjects = [
  { id: "p1", name: "Demo project", root_path: "/tmp/demo" },
];

const ts = "2024-01-01T00:00:00Z";

export const mockSessions = [
  {
    id: "s1",
    project_id: "p1",
    parent_id: null,
    mode: "dev",
    llm_profile: "default",
    title: "Первая сессия",
    status: "idle",
    error: null,
    archived: false,
    created_at: ts,
    updated_at: ts,
  },
  {
    id: "s2",
    project_id: "p1",
    parent_id: null,
    mode: "ask",
    llm_profile: "fast",
    title: "Вторая сессия",
    status: "completed",
    error: null,
    archived: false,
    created_at: ts,
    updated_at: ts,
  },
];

export const mockContext = {
  approximate_tokens: 500,
  context_window: 10000,
  omitted_messages: 2,
  truncated_tool_results: 0,
  reserved_tokens: 0,
  input_budget: 10000,
  fits: true,
  stage: "stage_1",
  summary_used: false,
  output_tokens: 20,
  safety_margin: 1000,
};

export const mockMessages = [
  {
    id: 11,
    session_id: "s1",
    role: "user",
    content: "Привет",
    tool_call_id: null,
    tool_calls: null,
    token_count: 10,
    skipped: false,
    kind: "user",
    reasoning_content: null,
    reasoning_effort: null,
    created_at: ts,
  },
];

export async function seedApp(
  page: Page,
  theme: "light" | "dark",
  panels: PanelState | null,
): Promise<void> {
  await page.addInitScript(
    ({ theme: t, panels: p }) => {
      try {
        if (window.localStorage.getItem("agent:theme") === null) {
          window.localStorage.setItem("agent:theme", t);
        }
        if (p) {
          window.localStorage.setItem(
            "agent:sidebar",
            JSON.stringify({ leftOpen: p.left, rightOpen: p.right }),
          );
        } else {
          window.localStorage.removeItem("agent:sidebar");
        }
        window.localStorage.setItem("draft:new-session", "");
      } catch {
        return;
      }
      const root = document.documentElement;
      root.setAttribute("data-theme", t);
    },
    { theme, panels },
  );
}

export async function installWsMock(page: Page): Promise<void> {
  const wsExpected = new URL("/ws", BASE_URL);
  await page.routeWebSocket(
    (url) =>
      url.protocol === "ws:" &&
      url.host === wsExpected.host &&
      url.pathname === "/ws",
    (ws) => {
      ws.onMessage(() => {});
    },
  );
}

export const mockModelSettings = {
  providers: [
    {
      id: "gateway",
      kind: "openai",
      base_url: "https://llm.example.com/v1",
      api_key_env: "AGENT_LLM_API_KEY",
      api_key_set: true,
      api_key_hint: "…abcd",
      configured: true,
      disabled: false,
      origin: "file",
    },
  ],
  models: [
    {
      id: "model-a",
      provider: "gateway",
      model: "remote-a",
      context_window: 128000,
      max_tokens: 8192,
      pricing: null,
      disabled: false,
      origin: "file",
    },
  ],
  routing: { default: "model-a", fallback: [], roles: {}, light: null },
  warnings: [],
};

/** The server wraps every session resource in one envelope with the run stamp. */
export function snapshotEnvelope(data: unknown) {
  return { data, run_generation: 0, revision: 1, consistent: true };
}

/** Replace one session resource (session, messages, approvals, context, ...) for this page. */
export async function routeSnapshot(page: Page, sessionId: string, resource: string, data: () => unknown) {
  await page.route(`**/api/sessions/${sessionId}/snapshot/${resource}`, (route) =>
    route.fulfill({ json: snapshotEnvelope(data()) }),
  );
}

async function json(route: import("@playwright/test").Route, status: number, body: unknown) {
  await route.fulfill({
    status,
    contentType: "application/json",
    body: JSON.stringify(body),
  });
}

export async function installRoutes(page: Page): Promise<void> {
  await page.route("**/*", async (route) => {
    const req = route.request();
    const url = new URL(req.url());
    if (url.protocol === "ws:" || url.protocol === "wss:") {
      await route.abort();
      return;
    }
    if (url.origin !== BASE_URL) {
      await route.abort();
      return;
    }
    const path = url.pathname;

    if (path === "/auth/session") {
      await json(route, 200, { authenticated: true, csrf_token: "test-csrf" });
      return;
    }
    if (path === "/auth/logout" || path === "/auth/login") {
      await json(route, 200, { authenticated: true });
      return;
    }
    if (path === "/api/config/public") {
      await json(route, 200, mockConfig);
      return;
    }
    if (path === "/api/projects") {
      await json(route, 200, mockProjects);
      return;
    }
    if (path === "/api/sessions") {
      if (req.method() === "POST") {
        await json(route, 500, { detail: "create failed" });
        return;
      }
      await json(route, 200, mockSessions);
      return;
    }
    if (path === "/ws") {
      await route.abort();
      return;
    }

    if (path === "/api/settings/models" && req.method() === "GET") {
      await json(route, 200, mockModelSettings);
      return;
    }

    const snapshot = /^\/api\/sessions\/([^/]+)\/snapshot\/([^/]+)$/.exec(path);
    if (snapshot) {
      const [, sid, resource] = snapshot;
      if (sid !== "s1" && sid !== "s2") {
        await json(route, 404, { detail: "not found" });
        return;
      }
      const resources: Record<string, unknown> = {
        session: sid === "s1" ? mockSessions[0] : mockSessions[1],
        messages: mockMessages,
        approvals: [],
        children: [],
        audit: [],
        context: mockContext,
      };
      if (resource in resources) await json(route, 200, snapshotEnvelope(resources[resource]));
      else await json(route, 404, { detail: "not found" });
      return;
    }

    const match = /^\/api\/sessions\/([^/]+)(?:\/([^/]+))?$/.exec(path);
    if (match) {
      const sid = match[1];
      const sub = match[2] ?? "";
      if (sid !== "s1" && sid !== "s2") {
        await json(route, 404, { detail: "not found" });
        return;
      }
      const detail = sid === "s1" ? mockSessions[0] : mockSessions[1];
      switch (sub) {
        case "":
          if (req.method() === "POST") {
            await json(route, 405, { detail: "method not allowed" });
          } else {
            await json(route, 200, detail);
          }
          return;
        case "messages":
          if (req.method() === "POST") {
            await json(route, 500, { detail: "send failed" });
          } else {
            await json(route, 200, mockMessages);
          }
          return;
        case "approvals":
          await json(route, 200, []);
          return;
        case "children":
          await json(route, 200, []);
          return;
        case "outbound-audit":
          await json(route, 200, []);
          return;
        case "context":
          await json(route, 200, mockContext);
          return;
        default:
          await json(route, 404, { detail: "not found" });
          return;
      }
    }

    if (path.startsWith("/api/")) {
      await json(route, 404, { detail: "not mocked" });
      return;
    }

    await route.continue();
  });
}

export async function openApp(
  page: Page,
  info: TestInfo,
  options?: { path?: string; cold?: boolean },
): Promise<void> {
  const theme = themeOf(info);
  const width = viewportWidthOf(info);
  const panels = options?.cold ? null : panelStateForWidth(width);
  await seedApp(page, theme, panels);
  await installRoutes(page);
  await installWsMock(page);
  await page.goto(options?.path ?? "/");
}

export function rectsOverlap(
  a: { left: number; right: number; top: number; bottom: number },
  b: { left: number; right: number; top: number; bottom: number },
  epsilon = 1,
): boolean {
  return (
    a.right - epsilon > b.left &&
    b.right - epsilon > a.left &&
    a.bottom - epsilon > b.top &&
    b.bottom - epsilon > a.top
  );
}

export async function assertThemeApplied(page: Page, theme: "light" | "dark"): Promise<void> {
  const applied = await page.evaluate(() => ({
    attr: document.documentElement.getAttribute("data-theme"),
    bg: getComputedStyle(document.body).backgroundColor,
  }));
  if (theme === "dark") {
    expect(applied.attr).toBe("dark");
    // --bg of the warm dark theme: #1b1714.
    expect(applied.bg).toBe("rgb(27, 23, 20)");
  } else {
    // --bg of the warm light theme: #faf6ef.
    expect(applied.bg).toBe("rgb(250, 246, 239)");
  }
}

export async function expectNoHorizontalOverflow(page: Page): Promise<void> {
  const overflow = await page.evaluate(() => ({
    scrollWidth: document.documentElement.scrollWidth,
    innerWidth: window.innerWidth,
  }));
  expect(overflow.scrollWidth).toBeLessThanOrEqual(overflow.innerWidth);
}

export async function assertResponsiveDefaultPanels(
  page: Page,
  width: number,
): Promise<void> {
  const left = page.locator(".app-shell__left");
  const right = page.locator(".app-shell__right");
  if (width >= 1440) {
    await expect(left).toBeVisible();
    await expect(right).toBeVisible();
  } else if (width >= 1100) {
    await expect(left).toBeVisible();
    await expect(right).toBeHidden();
  } else {
    await expect(left).toBeHidden();
    await expect(right).toBeHidden();
  }
}

export async function ensureRightPanelOpen(page: Page): Promise<void> {
  const rightToggle = page.locator('[data-toggle="right"]');
  const rightPanel = page.locator(".app-shell__right");
  const open = await rightPanel.isVisible().catch(() => false);
  if (!open && (await rightToggle.isVisible().catch(() => false))) {
    await rightToggle.click();
  }
  await expect(rightPanel).toBeVisible();
}

export async function ensureLeftPanelOpen(page: Page): Promise<void> {
  const leftToggle = page.locator('[data-toggle="left"]');
  const leftPanel = page.locator(".app-shell__left");
  const open = await leftPanel.isVisible().catch(() => false);
  if (!open && (await leftToggle.isVisible().catch(() => false))) {
    await leftToggle.click();
  }
  await expect(leftPanel).toBeVisible();
}

export async function ensureAgentControlsOpen(page: Page): Promise<void> {
  const button = page.getByRole("button", { name: "Параметры агента", exact: true });
  await expect(button).toBeVisible();
  if (await button.getAttribute("aria-expanded") === "false") {
    await button.click();
  }
  await expect(page.getByTestId("agent-controls")).toBeVisible();
}

export async function assertComposerGeometry(page: Page): Promise<void> {
  await ensureAgentControlsOpen(page);
  const inputEl = page.locator('[data-testid="composer-input"]');
  const sendEl = page.locator('[data-testid="composer-send"]');
  const effortEl = page.locator('[data-testid="composer-effort"]');
  await expect(inputEl).toBeVisible();
  await expect(sendEl).toBeVisible();
  await expect(effortEl).toBeVisible();

  const result = await page.evaluate(() => {
    const rect = (el: Element | null) => {
      if (!el) return null;
      const r = el.getBoundingClientRect();
      return { left: r.left, right: r.right, top: r.top, bottom: r.bottom };
    };
    const inputRow = rect(document.querySelector(".input-row"));
    const container = rect(document.querySelector(".input-container"));
    const composer = rect(document.querySelector("[data-testid='composer']"));
    const controls = rect(document.querySelector('[data-testid="agent-controls"]'));
    const qi = rect(document.querySelector('[data-testid="composer-input"]'));
    const qs = rect(document.querySelector('[data-testid="composer-send"]'));
    const qe = rect(document.querySelector('[data-testid="composer-effort"]'));
    return { inputRow, container, composer, controls, input: qi, send: qs, effort: qe, innerWidth: window.innerWidth };
  });

  expect(result.input).not.toBeNull();
  expect(result.send).not.toBeNull();
  expect(result.effort).not.toBeNull();
  expect(result.inputRow).not.toBeNull();
  expect(result.container).not.toBeNull();
  expect(result.composer).not.toBeNull();
  expect(result.controls).not.toBeNull();
  const input = result.input!;
  const send = result.send!;
  const effort = result.effort!;
  const inputRow = result.inputRow!;
  const container = result.container!;
  const composer = result.composer!;
  const controls = result.controls!;

  expect(rectsOverlap(input, send), "composer input overlaps send").toBe(false);
  expect(rectsOverlap(input, effort), "composer input overlaps effort").toBe(false);
  expect(rectsOverlap(send, effort), "composer send overlaps effort").toBe(false);

  expect(input.left).toBeGreaterThanOrEqual(inputRow.left - 1);
  expect(input.right).toBeLessThanOrEqual(inputRow.right + 1);
  expect(send.left).toBeGreaterThanOrEqual(inputRow.left - 1);
  expect(send.right).toBeLessThanOrEqual(inputRow.right + 1);
  expect(input.left).toBeGreaterThanOrEqual(container.left - 1);
  expect(send.right).toBeLessThanOrEqual(container.right + 1);

  expect(controls.bottom).toBeLessThanOrEqual(inputRow.top + 1);
  expect(effort.left).toBeGreaterThanOrEqual(controls.left - 1);
  expect(effort.right).toBeLessThanOrEqual(controls.right + 1);

  expect(input.right).toBeLessThanOrEqual(composer.right + 1);
  expect(send.right).toBeLessThanOrEqual(composer.right + 1);
}

export async function assertRightTabTextsDoNotOverlap(page: Page): Promise<void> {
  const tabs = page.locator(".right-panel__tab");
  await expect(tabs).toHaveCount(4);
  await expect(tabs.first()).toBeVisible();

  const result = await page.evaluate(() => {
    const tabEls = Array.from(document.querySelectorAll(".right-panel__tab"));
    const ranges = tabEls.map((tab) => {
      const range = document.createRange();
      range.selectNodeContents(tab);
      const br = range.getBoundingClientRect();
      return {
        text: (tab.textContent ?? "").trim(),
        left: br.left,
        right: br.right,
        width: br.width,
      };
    });
    for (let i = 0; i < ranges.length - 1; i++) {
      const a = ranges[i];
      const b = ranges[i + 1];
      if (a.width <= 0 || b.width <= 0) continue;
      if (a.right > b.left + 0.5) {
        return { ok: false, reason: `"${a.text}" right ${a.right} > "${b.text}" left ${b.left}` };
      }
    }
    return { ok: true };
  });

  expect(result.ok, result.reason).toBe(true);
}
