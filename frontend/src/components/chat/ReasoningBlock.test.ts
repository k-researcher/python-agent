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

  it("shows 'Думаю…' with the character count while streaming", () => {
    const wrapper = mount(ReasoningBlock, { props: { text: "частично", streaming: true } });
    const toggle = wrapper.get('[data-testid="reasoning-toggle"]');
    expect(toggle.text()).toContain("Думаю…");
    expect(toggle.text().match(/…/g)).toHaveLength(1);
    expect(toggle.text()).toContain("8 символов");
    expect((toggle.element as HTMLButtonElement).disabled).toBe(false);
  });

  it("expands while streaming and shows the text as it arrives", async () => {
    const wrapper = mount(ReasoningBlock, { props: { text: "первая", streaming: true } });
    await wrapper.get('[data-testid="reasoning-toggle"]').trigger("click");
    expect(wrapper.get('[data-testid="reasoning-body"]').text()).toContain("первая");
    await wrapper.setProps({ text: "первая мысль, вторая" });
    expect(wrapper.get('[data-testid="reasoning-body"]').text()).toContain("вторая");
  });

  it("keeps the live text scrolled to the end unless the user scrolled up", async () => {
    const wrapper = mount(ReasoningBlock, { props: { text: "a", streaming: true }, attachTo: document.body });
    await wrapper.get('[data-testid="reasoning-toggle"]').trigger("click");
    const body = wrapper.get('[data-testid="reasoning-body"]').element as HTMLElement;
    let height = 500;
    Object.defineProperty(body, "scrollHeight", { configurable: true, get: () => height });
    Object.defineProperty(body, "clientHeight", { configurable: true, get: () => 100 });

    await wrapper.setProps({ text: "ab" });
    expect(body.scrollTop).toBe(500);

    body.scrollTop = 50;
    await wrapper.get('[data-testid="reasoning-body"]').trigger("scroll");
    height = 900;
    await wrapper.setProps({ text: "abc" });
    expect(body.scrollTop).toBe(50);

    body.scrollTop = 800;
    await wrapper.get('[data-testid="reasoning-body"]').trigger("scroll");
    await wrapper.setProps({ text: "abcd" });
    expect(body.scrollTop).toBe(900);
    wrapper.unmount();
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
