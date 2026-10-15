import { describe, expect, it } from "vitest";
import { mount } from "@vue/test-utils";
import AppShell from "./AppShell.vue";

function mountShell(props: Record<string, unknown> = {}) {
  return mount(AppShell, {
    props,
    slots: {
      topbar: '<div class="slot-topbar">topbar</div>',
      left: '<div class="slot-left">left</div>',
      main: '<div class="slot-main">main</div>',
      right: '<div class="slot-right">right</div>',
    },
  });
}

describe("AppShell", () => {
  it("renders all four slots", () => {
    const wrapper = mountShell();
    expect(wrapper.find(".slot-topbar").exists()).toBe(true);
    expect(wrapper.find(".slot-left").exists()).toBe(true);
    expect(wrapper.find(".slot-main").exists()).toBe(true);
    expect(wrapper.find(".slot-right").exists()).toBe(true);
  });

  it("applies the spec column widths (264 / flexible / 320) via CSS variables", () => {
    const wrapper = mountShell({ leftOpen: true, rightOpen: true });
    const style = wrapper.find(".app-shell__body").attributes("style") ?? "";
    expect(style).toContain("--left-w: 264px");
    expect(style).toContain("--right-w: 320px");
  });

  it("collapses a closed column to zero width", () => {
    const wrapper = mountShell({ leftOpen: false, rightOpen: false });
    const style = wrapper.find(".app-shell__body").attributes("style") ?? "";
    expect(style).toContain("--left-w: 0px");
    expect(style).toContain("--right-w: 0px");
    // v-show hides the closed columns via an inline display:none.
    expect((wrapper.find(".app-shell__left").attributes("style") ?? "").toLowerCase()).toContain("display: none");
    expect((wrapper.find(".app-shell__right").attributes("style") ?? "").toLowerCase()).toContain("display: none");
  });

  it("hides only the closed column", () => {
    const wrapper = mountShell({ leftOpen: false, rightOpen: true });
    expect((wrapper.find(".app-shell__left").attributes("style") ?? "").toLowerCase()).toContain("display: none");
    expect((wrapper.find(".app-shell__right").attributes("style") ?? "").toLowerCase()).not.toContain("display: none");
  });

  it("emits update:leftOpen when the left toggle is clicked", async () => {
    const wrapper = mountShell({ leftOpen: true });
    await wrapper.find('[data-toggle="left"]').trigger("click");
    expect(wrapper.emitted("update:leftOpen")).toEqual([[false]]);
  });

  it("emits update:rightOpen when the right toggle is clicked", async () => {
    const wrapper = mountShell({ rightOpen: true });
    await wrapper.find('[data-toggle="right"]').trigger("click");
    expect(wrapper.emitted("update:rightOpen")).toEqual([[false]]);
  });

  it("emits the opposite value when the panel is already closed", async () => {
    const wrapper = mountShell({ leftOpen: false });
    await wrapper.find('[data-toggle="left"]').trigger("click");
    expect(wrapper.emitted("update:leftOpen")).toEqual([[true]]);
  });

  it("reflects the open state in the toggle aria-expanded attribute", () => {
    const wrapper = mountShell({ leftOpen: true, rightOpen: false });
    expect(wrapper.find('[data-toggle="left"]').attributes("aria-expanded")).toBe("true");
    expect(wrapper.find('[data-toggle="right"]').attributes("aria-expanded")).toBe("false");
  });
});
