import { describe, expect, it, vi } from "vitest";
import { mount, flushPromises } from "@vue/test-utils";
import RightPanel from "./RightPanel.vue";
import type { MonitorRow } from "../../api/types";

const context = {
  approximate_tokens: 20,
  context_window: 100,
  omitted_messages: 3,
};

function mountPanel(props: Record<string, unknown> = {}) {
  return mount(RightPanel, {
    props: {
      context,
      children: [
        { id: "c1", title: "Дочерний", status: "running" },
        { id: "c2", title: "Второй", status: "completed" },
      ],
      audit: [
        { id: 1, operation: "http_get", destination: "https://api.example.com", status: "ok", created_at: "2026-10-06T10:00:00Z" },
      ],
      tools: [
        { name: "shell", enabled: true },
        { name: "read_file", enabled: false },
      ],
      ...props,
    },
  });
}

const monitorRow: MonitorRow = { session_id: "global", parent_id: null, project_id: "other",
  project: "Другой проект", title: "Другая задача", status: "running", model: "default", mode: "dev",
  action_kind: "tool", action_text: "read_file", duration_seconds: 65, idle_seconds: 1,
  cost: "0.03", has_unknown_cost: false };

it("shows global activity and keeps Stop separate from opening a session", async () => {
  const wrapper = mountPanel({ monitorRows: [monitorRow] });
  await wrapper.get('[data-tab="monitor"]').trigger("click");
  expect(wrapper.emitted("monitorActive")?.at(-1)).toEqual([true]);
  const row = wrapper.get('[data-monitor-id="global"]');
  expect(row.text()).toContain("Другой проект");
  expect(row.text()).toContain("Выполняется");
  expect(row.text()).toContain("Шаг: read_file");
  expect(row.text()).toContain("1 м 5 с");
  expect(row.text()).toContain("$0.03");
  await row.get(".monitor-stop").trigger("click");
  expect(wrapper.emitted("stopSession")).toEqual([["global"]]);
  expect(wrapper.emitted("openSession")).toBeUndefined();
  await row.get(".monitor-open").trigger("click");
  expect(wrapper.emitted("openSession")).toEqual([["global"]]);
  await wrapper.get('[data-tab="context"]').trigger("click");
  expect(wrapper.emitted("monitorActive")?.at(-1)).toEqual([false]);
});

it.each([["assistant", "отвечает"], ["user", "ждёт модель"], ["none", "—"]] as const)(
  "labels %s activity and hides zero cost", async (kind, label) => {
    const wrapper = mountPanel({ monitorRows: [{ ...monitorRow, action_kind: kind, cost: "0.000", status: "completed" }] });
    await wrapper.get('[data-tab="monitor"]').trigger("click");
    expect(wrapper.get(".monitor-action").text()).toBe(`Шаг: ${label}`);
    expect(wrapper.find(".monitor-cost").exists()).toBe(false);
    expect(wrapper.find(".monitor-stop").exists()).toBe(false);
  },
);
it("shows loading, empty and error states and emits retry", async () => {
  const wrapper = mountPanel({ monitorLoading: true });
  await wrapper.get('[data-tab="monitor"]').trigger("click");
  expect(wrapper.text()).toContain("Загрузка агентов");
  await wrapper.setProps({ monitorLoading: false });
  expect(wrapper.text()).toContain("Нет активных или недавних");
  await wrapper.setProps({ monitorError: "Ошибка сети" });
  await wrapper.get('[role="alert"] button').trigger("click");
  expect(wrapper.emitted("refreshMonitor")).toHaveLength(1);
});
it("disables Stop during its request and marks partial known cost", async () => {
  const wrapper = mountPanel({ monitorRows: [{ ...monitorRow, has_unknown_cost: true }], monitorStopping: ["global"] });
  await wrapper.get('[data-tab="monitor"]').trigger("click");
  expect(wrapper.get(".monitor-stop").attributes("disabled")).toBeDefined();
  expect(wrapper.get(".monitor-cost").text()).toContain("неизвестная часть");
});

describe("RightPanel — tabs", () => {
  it("renders a tablist with the five tabs", () => {
    const wrapper = mountPanel();
    const tablist = wrapper.find('[role="tablist"]');
    expect(tablist.exists()).toBe(true);
    expect(tablist.findAll('[role="tab"]').map((tab) => tab.text())).toEqual([
      "Контекст",
      "Инструменты",
      "Агенты",
      "Аудит",
      "Работает",
    ]);
  });

  it("marks the active tab with aria-selected", () => {
    const wrapper = mountPanel();
    const tabs = wrapper.findAll('[role="tab"]');
    expect(tabs[0].attributes("aria-selected")).toBe("true");
    expect(tabs[1].attributes("aria-selected")).toBe("false");
    expect(wrapper.find('[data-panel="context"]').attributes("role")).toBe("tabpanel");
  });

  it("switches the visible panel when a tab is clicked", async () => {
    const wrapper = mountPanel();
    expect(wrapper.find('[data-panel="context"]').exists()).toBe(true);

    await wrapper.findAll('[role="tab"]')[1].trigger("click");
    expect(wrapper.find('[data-panel="tools"]').exists()).toBe(true);
    expect(wrapper.find('[data-panel="context"]').exists()).toBe(false);
    expect(wrapper.findAll('[role="tab"]')[1].attributes("aria-selected")).toBe("true");
  });

  it("links each panel to its tab via aria-labelledby", async () => {
    const wrapper = mountPanel();
    const tabIds = ["context", "tools", "agents", "audit", "monitor"];
    for (const id of tabIds) {
      const tab = wrapper.find(`[data-tab="${id}"]`);
      expect(tab.attributes("id")).toBe(`tab-${id}`);
      expect(tab.attributes("aria-controls")).toBe(`panel-${id}`);
    }
    expect(wrapper.find('[data-panel="context"]').attributes("aria-labelledby")).toBe("tab-context");
    await wrapper.findAll('[role="tab"]')[1].trigger("click");
    expect(wrapper.find('[data-panel="tools"]').attributes("aria-labelledby")).toBe("tab-tools");
    await wrapper.findAll('[role="tab"]')[2].trigger("click");
    expect(wrapper.find('[data-panel="agents"]').attributes("aria-labelledby")).toBe("tab-agents");
    await wrapper.findAll('[role="tab"]')[3].trigger("click");
    expect(wrapper.find('[data-panel="audit"]').attributes("aria-labelledby")).toBe("tab-audit");
  });

  it("gives the active tab tabindex 0 and others -1 on mount", () => {
    const wrapper = mountPanel();
    const tabs = wrapper.findAll('[role="tab"]');
    expect(tabs[0].attributes("tabindex")).toBe("0");
    expect(tabs[1].attributes("tabindex")).toBe("-1");
    expect(tabs[2].attributes("tabindex")).toBe("-1");
    expect(tabs[3].attributes("tabindex")).toBe("-1");
    expect((tabs[0].element as HTMLElement).id).toBe("tab-context");
  });

  it("updates tabindex when the active tab changes", async () => {
    const wrapper = mountPanel();
    await wrapper.findAll('[role="tab"]')[2].trigger("click");
    const tabs = wrapper.findAll('[role="tab"]');
    expect(tabs[2].attributes("tabindex")).toBe("0");
    expect(tabs[0].attributes("tabindex")).toBe("-1");
    expect(tabs[1].attributes("tabindex")).toBe("-1");
    expect(tabs[3].attributes("tabindex")).toBe("-1");
  });

  it("moves focus to the next tab on ArrowRight and wraps from the last", async () => {
    const wrapper = mountPanel();
    const tabs = wrapper.findAll('[role="tab"]');
    await tabs[0].trigger("keydown", { key: "ArrowRight" });
    expect(wrapper.findAll('[role="tab"]')[1].attributes("aria-selected")).toBe("true");
    expect(wrapper.find('[data-panel="tools"]').exists()).toBe(true);

    const last = wrapper.findAll('[role="tab"]')[4];
    await last.trigger("keydown", { key: "ArrowRight" });
    expect(wrapper.findAll('[role="tab"]')[0].attributes("aria-selected")).toBe("true");
    expect(wrapper.find('[data-panel="context"]').exists()).toBe(true);
  });

  it("moves focus to the previous tab on ArrowLeft and wraps from the first", async () => {
    const wrapper = mountPanel();
    const tabs = wrapper.findAll('[role="tab"]');
    await tabs[0].trigger("keydown", { key: "ArrowLeft" });
    expect(wrapper.findAll('[role="tab"]')[4].attributes("aria-selected")).toBe("true");
    expect(wrapper.find('[data-panel="monitor"]').exists()).toBe(true);

    const last = wrapper.findAll('[role="tab"]')[4];
    await last.trigger("keydown", { key: "ArrowLeft" });
    expect(wrapper.findAll('[role="tab"]')[3].attributes("aria-selected")).toBe("true");
    expect(wrapper.find('[data-panel="audit"]').exists()).toBe(true);
  });

  it("jumps to the first tab on Home and the last on End", async () => {
    const wrapper = mountPanel();
    await wrapper.findAll('[role="tab"]')[2].trigger("keydown", { key: "Home" });
    expect(wrapper.findAll('[role="tab"]')[0].attributes("aria-selected")).toBe("true");
    expect(wrapper.find('[data-panel="context"]').exists()).toBe(true);

    await wrapper.findAll('[role="tab"]')[0].trigger("keydown", { key: "End" });
    expect(wrapper.findAll('[role="tab"]')[4].attributes("aria-selected")).toBe("true");
    expect(wrapper.find('[data-panel="monitor"]').exists()).toBe(true);
  });

  it("calls preventDefault for recognised navigation keys", async () => {
    const wrapper = mountPanel();
    const tabs = wrapper.findAll('[role="tab"]');
    for (const key of ["ArrowRight", "ArrowLeft", "Home", "End"]) {
      const event = { key, preventDefault: vi.fn() };
      await tabs[0].trigger("keydown", event);
      expect(event.preventDefault).toHaveBeenCalled();
    }
  });

  it("does not call preventDefault for unrecognised keys", async () => {
    const wrapper = mountPanel();
    const tabs = wrapper.findAll('[role="tab"]');
    const event = { key: "Tab", preventDefault: vi.fn() };
    await tabs[0].trigger("keydown", event);
    expect(event.preventDefault).not.toHaveBeenCalled();
    expect(wrapper.findAll('[role="tab"]')[0].attributes("aria-selected")).toBe("true");
  });

  it("focuses the newly selected tab after keyboard navigation", async () => {
    const wrapper = mount(RightPanel, {
      props: {
        context,
        children: [],
        audit: [],
        tools: [],
      },
      attachTo: document.body,
    });
    const tabs = wrapper.findAll('[role="tab"]');
    await tabs[0].trigger("keydown", { key: "ArrowRight" });
    await flushPromises();
    const active = wrapper.findAll('[role="tab"]')[1];
    expect(active.attributes("aria-selected")).toBe("true");
    const activeId = (active.element as HTMLElement).id;
    expect(activeId).toBe("tab-tools");
    expect(document.activeElement?.id).toBe("tab-tools");
    wrapper.unmount();
  });

  it("keeps the active tab focused when mounting with attachTo", () => {
    const wrapper = mount(RightPanel, {
      props: {
        context,
        children: [],
        audit: [],
        tools: [],
      },
      attachTo: document.body,
    });
    const firstTab = wrapper.findAll('[role="tab"]')[0];
    expect(firstTab.attributes("aria-selected")).toBe("true");
    expect(firstTab.attributes("tabindex")).toBe("0");
    expect((firstTab.element as HTMLElement).id).toBe("tab-context");
    wrapper.unmount();
  });
});

describe("RightPanel — context", () => {
  it("computes the exact fill percentage", () => {
    const wrapper = mountPanel();
    expect(wrapper.find('[data-context-percent]').text()).toBe("20%");
    expect(wrapper.find('[data-context-detail]').text()).toContain("20");
    expect(wrapper.find('[data-context-detail]').text()).toContain("100");
  });

  it("clamps the percentage at 100", () => {
    const wrapper = mountPanel({
      context: { approximate_tokens: 250, context_window: 100, omitted_messages: 0 },
    });
    expect(wrapper.find('[data-context-percent]').text()).toBe("100%");
  });

  it("applies the warning tone from 80 %", () => {
    const normal = mountPanel({
      context: { approximate_tokens: 79, context_window: 100, omitted_messages: 0 },
    });
    expect(normal.find(".context-bar").attributes("data-tone")).toBe("normal");

    const atLimit = mountPanel({
      context: { approximate_tokens: 80, context_window: 100, omitted_messages: 0 },
    });
    expect(atLimit.find(".context-bar").attributes("data-tone")).toBe("--warn");
    expect(atLimit.find(".context-bar").classes()).toContain("context-bar--warn");

    const above = mountPanel({
      context: { approximate_tokens: 95, context_window: 100, omitted_messages: 0 },
    });
    expect(above.find(".context-bar").classes()).toContain("context-bar--warn");
    expect(above.find('[data-context-percent]').text()).toBe("95%");
  });

  it("renders 0 % for a zero context window (no division by zero)", () => {
    const wrapper = mountPanel({
      context: { approximate_tokens: 50, context_window: 0, omitted_messages: 2 },
    });
    expect(wrapper.find('[data-context-percent]').text()).toBe("0%");
  });

  it("renders 0 % for negative tokens from the database", () => {
    const wrapper = mountPanel({
      context: { approximate_tokens: -5, context_window: 100, omitted_messages: 0 },
    });
    expect(wrapper.find('[data-context-percent]').text()).toBe("0%");
  });

  it("renders with the exact fill width", () => {
    const wrapper = mountPanel();
    const fill = wrapper.find(".context-bar__fill");
    expect(fill.attributes("style")).toContain("width: 20%");
  });

  it("shows the omitted-messages count", () => {
    const wrapper = mountPanel();
    expect(wrapper.find('[data-omitted]').text()).toContain("3");
    expect(wrapper.find('[data-omitted]').classes()).toContain("context-block__omitted--warning");
  });

  it("handles a missing context (null) gracefully", () => {
    const wrapper = mountPanel({ context: null });
    expect(wrapper.find('[data-context-empty]').text()).toContain("Нет данных");
    expect(wrapper.find('[data-context-percent]').exists()).toBe(false);
  });

  it("renders 0 % (not NaN%) when approximate_tokens is NaN or missing", () => {
    for (const badTokens of [undefined, NaN, Infinity, -Infinity]) {
      const wrapper = mountPanel({
        context: {
          approximate_tokens: badTokens as never,
          context_window: 100,
          omitted_messages: 0,
        },
      });
      expect(wrapper.find('[data-context-percent]').text()).toBe("0%");
      expect(wrapper.find('[data-context-percent]').text()).not.toContain("NaN");
      expect(wrapper.find(".context-bar__fill").attributes("style")).toContain("width: 0%");
    }
  });

  it("renders 0 % when the context_window itself is NaN/Infinity", () => {
    for (const badWindow of [NaN, Infinity, -Infinity]) {
      const wrapper = mountPanel({
        context: {
          approximate_tokens: 50,
          context_window: badWindow as never,
          omitted_messages: 0,
        },
      });
      expect(wrapper.find('[data-context-percent]').text()).toBe("0%");
      expect(wrapper.find('[data-context-percent]').text()).not.toContain("NaN");
    }
  });

  it("uses input_budget as the meter denominator when positive and finite", () => {
    const wrapper = mountPanel({
      context: {
        approximate_tokens: 50,
        context_window: 100,
        input_budget: 200,
        omitted_messages: 0,
      },
    });
    expect(wrapper.find('[data-context-percent]').text()).toBe("25%");
    expect(wrapper.find('[data-context-detail]').text()).toContain("50 из 200 токенов");
  });

  it("falls back to context_window when input_budget is missing or non-finite", () => {
    for (const badBudget of [undefined, 0, -5, NaN, Infinity]) {
      const wrapper = mountPanel({
        context: {
          approximate_tokens: 50,
          context_window: 100,
          input_budget: badBudget as never,
          omitted_messages: 0,
        },
      });
      expect(wrapper.find('[data-context-percent]').text()).toBe("50%");
    }
  });

  it("shows an explicit error when fits === false", () => {
    const wrapper = mountPanel({
      context: { approximate_tokens: 90, context_window: 100, omitted_messages: 2, fits: false },
    });
    const errorBox = wrapper.find('[data-context-error]');
    expect(errorBox.exists()).toBe(true);
    expect(errorBox.find("span").text()).toContain("не помещается");
  });

  it("shows the server-provided error message when fits === false with error", () => {
    const wrapper = mountPanel({
      context: {
        approximate_tokens: 90,
        context_window: 100,
        omitted_messages: 2,
        fits: false,
        error: "Input too large for model",
      },
    });
    expect(wrapper.find('[data-context-error]').text()).toContain("Input too large for model");
  });

  it("does not render an error box when fits is true or undefined", () => {
    const fitting = mountPanel({
      context: { approximate_tokens: 30, context_window: 100, omitted_messages: 0, fits: true },
    });
    expect(fitting.find('[data-context-error]').exists()).toBe(false);

    const unknown = mountPanel({
      context: { approximate_tokens: 30, context_window: 100, omitted_messages: 0 },
    });
    expect(unknown.find('[data-context-error]').exists()).toBe(false);
  });

  it("renders the extra context metadata when present", () => {
    const wrapper = mountPanel({
      context: {
        approximate_tokens: 30,
        context_window: 100,
        omitted_messages: 0,
        stage: "summarize",
        summary_used: true,
        would_summarize: false,
        output_tokens: 120,
        safety_margin: 400,
      },
    });
    expect(wrapper.find('[data-context-stage]').text()).toBe("Старые сообщения заменены сводкой");
    expect(wrapper.find(".context-block__meta").text()).toContain("использована");
    expect(wrapper.find(".context-block__meta").text()).toContain("120");
    expect(wrapper.find(".context-block__meta").text()).toContain("400");
  });

  it("treats negative omitted_messages as zero (no 'нет' false positive)", () => {
    const wrapper = mountPanel({
      context: { approximate_tokens: 30, context_window: 100, omitted_messages: -1 },
    });
    expect(wrapper.find('[data-omitted]').text()).toBe("Пропущенных сообщений нет");
    expect(wrapper.find('[data-omitted]').classes()).not.toContain("context-block__omitted--warning");
  });

  it("limits the wrapping layout to the compression metadata row", () => {
    const wrapper = mountPanel({
      context: {
        ...context,
        stage: "stage_1",
        summary_used: false,
        output_tokens: 20,
        safety_margin: 1000,
      },
    });
    const rows = wrapper.findAll(".context-block__meta > div");
    expect(rows).toHaveLength(4);
    expect(rows[0].classes()).toContain("context-block__compression");
    expect(rows[0].get("dt").text()).toBe("Сжатие контекста");
    for (const row of rows.slice(1)) {
      expect(row.classes()).not.toContain("context-block__compression");
    }
    expect(rows.slice(1).map((row) => row.get("dt").text())).toEqual([
      "Сводка", "Выходные токены", "Запас",
    ]);
  });

  it.each([
    ["stage_1", "Сокращены результаты инструментов"],
    ["stage_2", "Свёрнуты вызовы инструментов"],
    ["summary", "Старые сообщения заменены сводкой"],
    ["summarize", "Старые сообщения заменены сводкой"],
    ["deterministic", "История сокращена без модели"],
  ])("localizes context stage %s", (stage, label) => {
    const wrapper = mountPanel({
      context: { approximate_tokens: 30, context_window: 100, omitted_messages: 0, stage },
    });
    expect(wrapper.get("[data-context-stage]").text()).toBe(label);
    expect(wrapper.get(".context-block__meta dt").text()).toBe("Сжатие контекста");
  });

  it.each([undefined, "", "none", "future_stage", "constructor", "__proto__"])("omits an absent or unknown context stage (%s)", (stage) => {
    const wrapper = mountPanel({
      context: { approximate_tokens: 30, context_window: 100, omitted_messages: 0, stage },
    });
    expect(wrapper.find("[data-context-stage]").exists()).toBe(false);
    expect(wrapper.find(".context-block__meta").exists()).toBe(false);
  });

  it("removes an unknown compression row without hiding other context metadata", async () => {
    const stats = {
      approximate_tokens: 30,
      context_window: 100,
      omitted_messages: 0,
      stage: "stage_1",
      summary_used: true,
      output_tokens: 120,
    };
    const wrapper = mountPanel({ context: stats });
    expect(wrapper.get("[data-context-stage]").text()).toBe("Сокращены результаты инструментов");
    await wrapper.setProps({ context: { ...stats, stage: "future_stage" } });
    expect(wrapper.find("[data-context-stage]").exists()).toBe(false);
    const metadata = wrapper.get(".context-block__meta");
    expect(metadata.text()).not.toContain("Сжатие контекста");
    expect(metadata.text()).not.toContain("Неизвестный этап");
    expect(metadata.text()).toContain("использована");
    expect(metadata.text()).toContain("120");
  });
});

describe("RightPanel — tools, agents, audit", () => {
  it("lists tools read-only with enabled state", async () => {
    const wrapper = mountPanel();
    await wrapper.findAll('[role="tab"]')[1].trigger("click");
    const tools = wrapper.findAll('[data-tool]');
    expect(tools.map((tool) => tool.text())).toEqual([
      "shellвкл",
      "read_fileвыкл",
    ]);
    expect(tools[0].find('[data-enabled="on"]').exists()).toBe(true);
    expect(tools[1].find('[data-enabled="off"]').exists()).toBe(true);
  });

  it("lists child agents and emits openSession on click", async () => {
    const wrapper = mountPanel();
    await wrapper.findAll('[role="tab"]')[2].trigger("click");
    const agents = wrapper.findAll(".agent-row");
    expect(agents).toHaveLength(2);
    expect(agents[0].text()).toContain("Дочерний");
    expect(agents[0].get("[data-agent-status]").text()).toBe("Выполняется");

    await agents[1].find('[data-agent-id="c2"]').trigger("click");
    expect(wrapper.emitted("openSession")).toEqual([["c2"]]);
  });

  it("shows the empty state for children and tools", async () => {
    const wrapper = mountPanel({ children: [], tools: [] });
    await wrapper.findAll('[role="tab"]')[1].trigger("click");
    expect(wrapper.text()).toContain("Инструменты не настроены");
    await wrapper.findAll('[role="tab"]')[2].trigger("click");
    expect(wrapper.text()).toContain("Нет дочерних агентов");
  });

  it("lists audit entries with operation, destination and status", async () => {
    const wrapper = mountPanel();
    await wrapper.findAll('[role="tab"]')[3].trigger("click");
    const entries = wrapper.findAll('[data-audit]');
    expect(entries).toHaveLength(1);
    expect(entries[0].text()).toContain("http_get");
    expect(entries[0].text()).toContain("https://api.example.com");
    expect(entries[0].find('[data-audit-status]').text()).toBe("ok");
  });

  it("shows the audit time in the local time zone, not the raw server value", async () => {
    const wrapper = mountPanel();
    await wrapper.findAll('[role="tab"]')[3].trigger("click");
    const time = wrapper.get(".audit-row__time");
    expect(time.text()).not.toContain("T10:00:00Z");
    expect(time.text()).toContain(
      new Date("2026-10-06T10:00:00Z").toLocaleTimeString("ru-RU", { hour: "2-digit", minute: "2-digit", second: "2-digit" }),
    );
    expect(time.attributes("datetime")).toBe("2026-10-06T10:00:00Z");
  });

  it("shows an empty audit state", async () => {
    const wrapper = mountPanel({ audit: [] });
    await wrapper.findAll('[role="tab"]')[3].trigger("click");
    expect(wrapper.text()).toContain("Записей аудита нет");
  });
});
