import { flushPromises, mount } from "@vue/test-utils";
import { describe, expect, it } from "vitest";
import ChatTimeline from "./ChatTimeline.vue";
import ToolCallCard from "./ToolCallCard.vue";
import type { ChatMessage, TimelineItem, ToolCallView } from "./types";

function message(partial: Partial<ChatMessage>): ChatMessage {
  return {
    id: "m1",
    role: "assistant",
    content: "текст",
    reasoning_content: null,
    reasoning_effort: null,
    created_at: "2026-01-01T10:00:00Z",
    kind: "assistant",
    ...partial,
  };
}

function pendingCall(id: string, approvalId: string): ToolCallView {
  return {
    id,
    name: "shell",
    arguments: { command: "ls" },
    argumentsRaw: '{"command":"ls"}',
    result: null,
    approval: {
      id: approvalId,
      tool_call_id: id,
      tool_name: "shell",
      arguments: {},
      risk_level: "low",
      status: "awaiting",
      comment: null,
    },
    status: "awaiting_approval",
  };
}

function toolCall(partial: Partial<ToolCallView> = {}): ToolCallView {
  return {
    id: "t1",
    name: "shell",
    arguments: { command: "ls" },
    argumentsRaw: '{"command":"ls"}',
    result: null,
    approval: null,
    status: "completed",
    ...partial,
  };
}

function mountTimeline(items: TimelineItem[]) {
  return mount(ChatTimeline, { props: { items } });
}

describe("ChatTimeline", () => {
  // Mounted into the document so real scroll events can be dispatched on the
  // scroller element (happy-dom ignores events on detached nodes).
  function mountAttached(items: TimelineItem[]) {
    return mount(ChatTimeline, { props: { items }, attachTo: document.body });
  }

  it("renders user and assistant messages", () => {
    const wrapper = mountTimeline([
      { kind: "user", message: message({ role: "user", kind: "user", content: "привет" }) },
      { kind: "assistant", message: message({ content: "здравствуйте" }) },
    ]);
    const messages = wrapper.findAll('[data-testid="message"]');
    expect(messages).toHaveLength(2);
    expect(messages[0].text()).toContain("привет");
    expect(messages[1].text()).toContain("здравствуйте");
  });

  it("renders tool calls by kind", () => {
    const wrapper = mountTimeline([{ kind: "tool", call: toolCall() }]);
    expect(wrapper.find('[data-testid="tool-call"]').exists()).toBe(true);
    expect(wrapper.get('[data-testid="tool-name"]').text()).toBe("shell");
  });

  it("renders the streaming item for kind=streaming", () => {
    const wrapper = mountTimeline([
      { kind: "streaming", content: "частичный текст", reasoning: "размышления" },
    ]);
    expect(wrapper.find('[data-testid="streaming-item"]').exists()).toBe(true);
    expect(wrapper.find('[data-testid="reasoning-toggle"]').exists()).toBe(true);
  });

  it("does not add a blank cursor row during reasoning-only streaming", async () => {
    const wrapper = mountTimeline([{ kind: "streaming", content: "", reasoning: "размышления" }]);
    const item = wrapper.get('[data-testid="streaming-item"]');
    expect(item.find(".streaming-cursor").exists()).toBe(false);
    expect(item.find('[data-testid="markdown-body"]').exists()).toBe(false);
    expect(item.text().match(/…/g)).toHaveLength(1);
    await wrapper.setProps({ items: [{ kind: "streaming", content: "Ответ", reasoning: "готово" }] });
    await flushPromises();
    expect(item.find(".streaming-cursor").exists()).toBe(false);
    expect(item.get('[data-testid="markdown-body"]').text()).toBe("Ответ");
  });

  it("keeps one waiting cursor before either stream starts", () => {
    const wrapper = mountTimeline([{ kind: "streaming", content: "", reasoning: "" }]);
    expect(wrapper.findAll(".streaming-cursor")).toHaveLength(1);
    expect(wrapper.find('[data-testid="reasoning-toggle"]').exists()).toBe(false);
  });

  it("renders a summary card collapsed by default", () => {
    const wrapper = mountTimeline([
      { kind: "summary", message: message({ content: "длинная сводка" }) },
    ]);
    const card = wrapper.get('[data-testid="summary-card"]');
    expect(card.text()).toContain("Сводка ранней истории");
    expect(wrapper.find('[data-testid="summary-body"]').exists()).toBe(false);
  });

  it("expands summary text on toggle", async () => {
    const wrapper = mountTimeline([
      { kind: "summary", message: message({ content: "полезная сводка" }) },
    ]);
    await wrapper.get('[data-testid="summary-toggle"]').trigger("click");
    const body = wrapper.find('[data-testid="summary-body"]');
    expect(body.exists()).toBe(true);
    expect(body.text()).toContain("полезная сводка");
  });

  it("shows a hint when there are no items", () => {
    const wrapper = mountTimeline([]);
    expect(wrapper.find('[data-testid="timeline-empty"]').exists()).toBe(true);
  });

  it("auto-scrolls to the bottom when the user is already at the bottom", async () => {
    const wrapper = mountAttached([
      { kind: "assistant", message: message({ content: "первое" }) },
    ]);
    try {
      const scroller = wrapper.get('[data-testid="timeline"]').element as HTMLElement;
      // Simulate a real scrollable pane (happy-dom reports 0 metrics for a plain div).
      Object.defineProperty(scroller, "scrollHeight", { value: 300, configurable: true });
      Object.defineProperty(scroller, "clientHeight", { value: 100, configurable: true });
      await wrapper.setProps({
        items: [
          { kind: "assistant", message: message({ content: "первое" }) },
          { kind: "assistant", message: message({ content: "второе" }) },
        ],
      });
      expect(scroller.scrollTop).toBe(300);
    } finally {
      wrapper.unmount();
    }
  });

  it("does not yank the view when the user scrolled up", async () => {
    const wrapper = mountAttached([
      { kind: "assistant", message: message({ content: "a" }) },
    ]);
    try {
      const scroller = wrapper.get('[data-testid="timeline"]').element as HTMLElement;
      Object.defineProperty(scroller, "scrollHeight", { value: 300, configurable: true });
      Object.defineProperty(scroller, "clientHeight", { value: 100, configurable: true });
      // User scrolls such that they are no longer pinned to the bottom.
      scroller.scrollTop = 10; // distance = 300 - 10 - 100 = 190 >= 80
      scroller.dispatchEvent(new Event("scroll"));
      await wrapper.setProps({
        items: [
          { kind: "assistant", message: message({ content: "a" }) },
          { kind: "assistant", message: message({ content: "b" }) },
        ],
      });
      expect(scroller.scrollTop).toBe(10);
    } finally {
      wrapper.unmount();
    }
  });

  it("enables hotkeys only on the first tool call awaiting approval", () => {
    const wrapper = mountTimeline([
      { kind: "tool", call: pendingCall("t1", "ap1") },
      { kind: "tool", call: pendingCall("t2", "ap2") },
      { kind: "tool", call: { ...pendingCall("t3", "ap3"), status: "completed", approval: null } },
    ]);
    const cards = wrapper.findAllComponents(ToolCallCard);
    expect(cards).toHaveLength(3);
    expect(cards[0].props("hotkeys")).toBe(true);
    expect(cards[1].props("hotkeys")).toBe(false);
    expect(cards[2].props("hotkeys")).toBe(false);
  });

  it("forwards approve and reject decisions of a tool card", () => {
    const wrapper = mountTimeline([{ kind: "tool", call: pendingCall("t1", "ap1") }]);
    const card = wrapper.findComponent(ToolCallCard);
    card.vm.$emit("approve", "ap1");
    card.vm.$emit("reject", { approvalId: "ap1", comment: "не нужно" });
    expect(wrapper.emitted("approve")).toEqual([["ap1"]]);
    expect(wrapper.emitted("reject")).toEqual([[{ approvalId: "ap1", comment: "не нужно" }]]);
  });

  it("disables hotkeys everywhere when no call awaits approval", () => {
    const wrapper = mountTimeline([
      { kind: "tool", call: { ...pendingCall("t1", "ap1"), status: "completed", approval: null } },
      { kind: "tool", call: { ...pendingCall("t2", "ap2"), status: "error", approval: null } },
    ]);
    const cards = wrapper.findAllComponents(ToolCallCard);
    for (const card of cards) {
      expect(card.props("hotkeys")).toBe(false);
    }
  });

  it("moves hotkeys to the new first pending call when the earlier one resolves", async () => {
    const wrapper = mountTimeline([
      { kind: "tool", call: pendingCall("t1", "ap1") },
      { kind: "tool", call: pendingCall("t2", "ap2") },
    ]);
    expect(wrapper.findAllComponents(ToolCallCard)[0].props("hotkeys")).toBe(true);

    await wrapper.setProps({
      items: [
        { kind: "tool", call: { ...pendingCall("t1", "ap1"), status: "completed", approval: null } },
        { kind: "tool", call: pendingCall("t2", "ap2") },
      ],
    });
    const cards = wrapper.findAllComponents(ToolCallCard);
    expect(cards[0].props("hotkeys")).toBe(false);
    expect(cards[1].props("hotkeys")).toBe(true);
  });

  it("keeps pinned autoscroll when content height grows after render (highlighting)", async () => {
    // Stub ResizeObserver so we can drive the height-change callback manually
    // (happy-dom defines it but never fires on size changes).
    const sink: { cb: (() => void) | null } = { cb: null };
    class MockResizeObserver {
      constructor(cb: () => void) {
        sink.cb = cb;
      }
      disconnect() {}
      observe() {}
      unobserve() {}
    }
    const RealRO = globalThis.ResizeObserver;
    globalThis.ResizeObserver = MockResizeObserver as unknown as typeof ResizeObserver;

    const wrapper = mountAttached([
      { kind: "assistant", message: message({ content: "```js\nconst a = 1;\n```" }) },
    ]);
    try {
      const scroller = wrapper.get('[data-testid="timeline"]').element as HTMLElement;
      let height = 300;
      Object.defineProperty(scroller, "scrollHeight", { value: height, configurable: true });
      Object.defineProperty(scroller, "clientHeight", { value: 100, configurable: true });
      await flushPromises();

      // Content grows once shiki highlighting injects taller blocks.
      height = 700;
      Object.defineProperty(scroller, "scrollHeight", { value: height, configurable: true });
      sink.cb?.();
      await new Promise((r) => setTimeout(r, 20)); // let rAF run
      // Pinned to the bottom: the view falls through to the new bottom.
      expect(scroller.scrollTop).toBe(height);
    } finally {
      wrapper.unmount();
      globalThis.ResizeObserver = RealRO;
    }
  });
});
