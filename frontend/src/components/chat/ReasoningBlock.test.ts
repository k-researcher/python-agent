import { mount } from "@vue/test-utils";
import { describe, expect, it } from "vitest";
import ReasoningBlock from "./ReasoningBlock.vue";

describe("ReasoningBlock", () => {
  it("is collapsed by default and shows the character count", () => {
    const wrapper = mount(ReasoningBlock, { props: { text: "abc", streaming: false } });
    expect(wrapper.find('[data-testid="reasoning-body"]').exists()).toBe(false);
    expect(wrapper.get('[data-testid="reasoning-toggle"]').text()).toContain("Рассуждения");
    expect(wrapper.text()).toContain("3");
  });

  it("expands on click and shows the reasoning text", async () => {
    const wrapper = mount(ReasoningBlock, { props: { text: "подумаю", streaming: false } });
    await wrapper.get('[data-testid="reasoning-toggle"]').trigger("click");
    const body = wrapper.get('[data-testid="reasoning-body"]');
    expect(body.text()).toContain("подумаю");
  });

  it("shows 'Думаю…' while streaming and disables the toggle", () => {
    const wrapper = mount(ReasoningBlock, { props: { text: "частично", streaming: true } });
    expect(wrapper.get('[data-testid="reasoning-toggle"]').text()).toContain("Думаю…");
    expect(wrapper.get('[data-testid="reasoning-toggle"]').text().match(/…/g)).toHaveLength(1);
    expect((wrapper.get('[data-testid="reasoning-toggle"]').element as HTMLButtonElement).disabled).toBe(true);
  });

  it("handles an empty text without errors", () => {
    const wrapper = mount(ReasoningBlock, { props: { text: "", streaming: false } });
    expect(wrapper.get('[data-testid="reasoning-toggle"]').text()).toContain("0");
  });

  it("pluralizes the symbol counter at 1 / 2 / 5 boundaries", () => {
    expect(mount(ReasoningBlock, { props: { text: "a" } }).get('[data-testid="reasoning-toggle"]').text()).toContain(
      "1 символ",
    );
    expect(mount(ReasoningBlock, { props: { text: "ab" } }).get('[data-testid="reasoning-toggle"]').text()).toContain(
      "2 символа",
    );
    expect(
      mount(ReasoningBlock, { props: { text: "abcde" } }).get('[data-testid="reasoning-toggle"]').text(),
    ).toContain("5 символов");
  });
});
