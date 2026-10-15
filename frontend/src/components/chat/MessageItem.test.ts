import { mount } from "@vue/test-utils";
import { describe, expect, it } from "vitest";
import MessageItem from "./MessageItem.vue";
import type { ChatMessage } from "./types";

function makeMessage(partial: Partial<ChatMessage>): ChatMessage {
  return {
    id: "m1",
    role: "assistant",
    content: "ответ",
    reasoning_content: null,
    reasoning_effort: null,
    created_at: "2026-01-01T10:00:00Z",
    kind: "assistant",
    ...partial,
  };
}

describe("MessageItem", () => {
  it("renders the assistant message on the left", () => {
    const wrapper = mount(MessageItem, { props: { message: makeMessage({}) } });
    expect(wrapper.get('[data-testid="message"]').classes()).not.toContain("user");
    expect(wrapper.text()).toContain("ответ");
  });

  it("renders the user message right-aligned", () => {
    const wrapper = mount(MessageItem, {
      props: { message: makeMessage({ role: "user", kind: "user", content: "вопрос" }) },
    });
    expect(wrapper.get('[data-testid="message"]').classes()).toContain("user");
  });

  it("renders a reasoning block when reasoning_content is present", () => {
    const wrapper = mount(MessageItem, {
      props: { message: makeMessage({ reasoning_content: "этапы хода" }) },
    });
    expect(wrapper.find('[data-testid="reasoning-toggle"]').exists()).toBe(true);
  });

  it("shows time and effort metadata under the assistant message", () => {
    const wrapper = mount(MessageItem, {
      props: { message: makeMessage({ reasoning_effort: "high" }) },
    });
    const meta = wrapper.get('[data-testid="message-meta"]');
    const expectedTime = new Date("2026-01-01T10:00:00Z").toLocaleTimeString("en-GB", {
      hour: "2-digit",
      minute: "2-digit",
    });
    expect(meta.text()).toContain(expectedTime);
    expect(meta.text()).toContain("Рассуждения: Высокий");
  });

  it("omits assistant metadata when the values are missing", () => {
    const wrapper = mount(MessageItem, {
      props: { message: makeMessage({ reasoning_effort: null, created_at: "" }) },
    });
    expect(wrapper.find('[data-testid="message-meta"]').text()).not.toContain("Рассуждения");
  });

  it("shows none as a human label without a redundant prefix", () => {
    const wrapper = mount(MessageItem, {
      props: { message: makeMessage({ reasoning_effort: "none" }) },
    });
    expect(wrapper.get(".effort").text()).toBe("Без рассуждений");
  });

  it("keeps an unknown effort visible", () => {
    const wrapper = mount(MessageItem, {
      props: { message: makeMessage({ reasoning_effort: "future_effort" }) },
    });
    expect(wrapper.get(".effort").text()).toBe("Рассуждения: future_effort");
  });

  it("does not remove English text from the actual answer", () => {
    const content = "Let me think about this example. Here is the result.";
    const wrapper = mount(MessageItem, { props: { message: makeMessage({ content }) } });
    expect(wrapper.get('[data-testid="markdown-body"]').text()).toBe(content);
  });
});
