import { mount } from "@vue/test-utils";
import { describe, expect, it } from "vitest";
import EmptyState from "./EmptyState.vue";

describe("EmptyState", () => {
  it("shows the 'Что будем делать?' heading and four cards", () => {
    const wrapper = mount(EmptyState);
    expect(wrapper.get('[data-testid="empty-state"]').text()).toContain("Что будем делать?");
    const cards = wrapper.findAll('[data-testid="suggestion-card"]');
    expect(cards).toHaveLength(4);
    const titles = cards.map((c) => c.get(".card-title").text());
    expect(titles).toEqual(["Новая идея", "Составить план", "Разобрать код", "Найти ошибку"]);
  });

  it("emits the template text when a card is clicked", async () => {
    const wrapper = mount(EmptyState);
    const cards = wrapper.findAll('[data-testid="suggestion-card"]');
    await cards[1].trigger("click"); // «Составить план»
    expect(wrapper.emitted("pick")?.[0]).toEqual(["Составь пошаговый план реализации этой задачи."]);
  });

  it("always emits a non-empty template for every card", async () => {
    const wrapper = mount(EmptyState);
    for (const card of wrapper.findAll('[data-testid="suggestion-card"]')) {
      await card.trigger("click");
    }
    const picks = (wrapper.emitted("pick") ?? []).map((args) => args[0] as string);
    expect(picks).toHaveLength(4);
    for (const pick of picks) {
      expect(typeof pick).toBe("string");
      expect(pick.trim().length).toBeGreaterThan(0);
    }
  });
});
