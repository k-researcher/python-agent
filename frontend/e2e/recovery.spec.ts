import { expect, test, type Page } from "@playwright/test";
import {
  openApp,
  seedApp,
  installRoutes,
  ensureLeftPanelOpen,
  panelStateForWidth,
  themeOf,
  viewportWidthOf,
  mockMessages,
  mockSessions,
  routeSnapshot,
} from "./helpers";

/** A user message followed by a partial assistant reply for a running session. */
function runningSessionMessages(): unknown[] {
  return [
    {
      ...mockMessages[0],
      id: 11,
      role: "user",
      content: "Найди баг",
      kind: "normal",
    },
    {
      ...mockMessages[0],
      id: 12,
      role: "assistant",
      content: "Сейчас посмотрю…",
      kind: "assistant",
    },
  ];
}

async function openSessionViaSidebar(
  page: Page,
  info: import("@playwright/test").TestInfo,
  id: string,
): Promise<void> {
  await openApp(page, info);
  await ensureLeftPanelOpen(page);
  await page.locator(`[data-node-key="n:${id}"]`).click();
  await expect(page).toHaveURL(new RegExp(`/s/${id}`));
}

/** Home/session flow without the stock WS mock, so the test owns /ws routing.
 *  `setup` runs after installRoutes so its routes take precedence. */
async function openSessionWithOwnWs(
  page: Page,
  info: import("@playwright/test").TestInfo,
  id: string,
  sockets: import("@playwright/test").WebSocketRoute[],
  setup?: () => void,
): Promise<void> {
  await seedApp(page, themeOf(info), panelStateForWidth(viewportWidthOf(info)));
  await installRoutes(page);
  setup?.();
  // Registered after installRoutes (like helpers' installWsMock) so the WS
  // interception wins over the generic `**/*` route that aborts ws.
  await page.routeWebSocket((url) => url.pathname === "/ws", (ws) => {
    sockets.push(ws);
  });
  await page.goto("/");
  await ensureLeftPanelOpen(page);
  await page.locator(`[data-node-key="n:${id}"]`).click();
  await expect(page).toHaveURL(new RegExp(`/s/${id}`));
}

test.describe("session state recovery", () => {
  test("reload during a running session keeps messages, status and the busy indicator", async ({ page }, testInfo) => {
    await openSessionViaSidebar(page, testInfo, "s1");
    await routeSnapshot(page, "s1", "session", () => ({ ...mockSessions[0], status: "running" }));
    await routeSnapshot(page, "s1", "messages", () => (runningSessionMessages()));

    await page.reload();

    await expect(page.getByTestId("session-status")).toHaveText("Выполняется");
    await expect(page.getByTestId("session-status")).toHaveAttribute("data-status", "running");
    await expect(page.getByText("Найди баг", { exact: true })).toBeVisible();
    await expect(page.getByText("Сейчас посмотрю…", { exact: true })).toBeVisible();
    // Running state keeps the composer busy and shows the stop control as the
    // work indicator.
    await expect(page.getByTestId("composer-stop")).toBeVisible();
    await expect(page.getByTestId("composer-input")).toBeDisabled();
  });

  test(
    "awaiting_confirmation approval survives reload and its agree button posts approve",
    async ({ page }, testInfo) => {
      await openSessionViaSidebar(page, testInfo, "s1");
      const decisions: unknown[] = [];
      await page.route("**/api/approvals/ap1", (route) => {
        decisions.push(route.request().postDataJSON());
        return route.fulfill({ json: { session_id: "s1", decision: "approve" } });
      });
      await routeSnapshot(page, "s1", "session", () => ({ ...mockSessions[0], status: "awaiting_confirmation" }));
      await routeSnapshot(page, "s1", "messages", () => ([
            {
              ...mockMessages[0],
              id: 21,
              role: "assistant",
              content: null,
              kind: "assistant",
              tool_calls: [
                {
                  id: "call_1",
                  type: "function",
                  function: { name: "shell", arguments: '{"command":"ls"}' },
                },
              ],
            },
          ]));
      await routeSnapshot(page, "s1", "approvals", () => ([
            {
              id: "ap1",
              tool_call_id: "call_1",
              tool_name: "shell",
              arguments: {},
              risk_level: "low",
              status: "pending",
              comment: null,
            },
          ]));

      await page.reload();

      await expect(page.getByTestId("approval")).toBeVisible();
      await expect(page.getByTestId("session-status")).toHaveText("Ждёт подтверждения");

      await page.getByTestId("approve-btn").click();
      await expect.poll(() => decisions).toEqual([{ decision: "approve", comment: null }]);
    },
  );

  test("reconnects after a WebSocket drop and reloads missed messages via the API", async ({ page }, testInfo) => {
    // Own the /ws routing: collect every connection so the first can be dropped.
    const sockets: import("@playwright/test").WebSocketRoute[] = [];

    let messageCalls = 0;

    await openSessionWithOwnWs(page, testInfo, "s1", sockets, () => {
      // Registered after installRoutes so it shadows the generic `**/*` mock.
      void routeSnapshot(page, "s1", "messages", () => {
        messageCalls += 1;
        return messageCalls === 1
          ? [{ ...mockMessages[0], id: 11, content: "старое сообщение" }]
          : [
              { ...mockMessages[0], id: 11, content: "старое сообщение" },
              {
                ...mockMessages[0],
                id: 12,
                role: "assistant",
                kind: "assistant",
                content: "пропущенный ответ",
              },
            ];
      });
    });

    // Initial payload is shown before the drop.
    await expect(page.getByText("старое сообщение", { exact: true })).toBeVisible();
    await expect.poll(() => sockets.length).toBeGreaterThanOrEqual(1);

    // Drop the live connection; the client must reconnect on its own.
    sockets[0].close();

    await expect.poll(() => sockets.length).toBeGreaterThanOrEqual(2);
    // The reconnect reopens the session and the missed reply shows up from the API.
    await expect(page.getByText("пропущенный ответ", { exact: true })).toBeVisible();
  });

  test("preserves a typed draft in the composer across a reload", async ({ page }, testInfo) => {
    await openSessionViaSidebar(page, testInfo, "s1");
    const input = page.getByTestId("composer-input");
    await expect(input).toBeVisible();
    const draft = "черновик, который переживёт перезагрузку";
    await input.fill(draft);
    // Composer persists via a post-flush watcher; wait until it lands.
    await expect
      .poll(() => page.evaluate(() => window.localStorage.getItem("draft:s1")))
      .toBe(draft);

    await page.reload();

    await expect(page.getByTestId("composer-input")).toBeVisible();
    await expect(input).toHaveValue(draft);
  });

  test("shows the final status and answer when a session completed while the tab was closed", async ({ page }, testInfo) => {
    await openSessionViaSidebar(page, testInfo, "s1");
    await routeSnapshot(page, "s1", "session", () => ({ ...mockSessions[0], status: "completed" }));
    await routeSnapshot(page, "s1", "messages", () => ([{ ...mockMessages[0], id: 31, content: "Итоговый ответ агента" }]));

    await page.reload();

    await expect(page.getByTestId("session-status")).toHaveText("Завершена");
    await expect(page.getByTestId("session-status")).toHaveAttribute("data-status", "completed");
    await expect(page.getByText("Итоговый ответ агента", { exact: true })).toBeVisible();
  });
});
