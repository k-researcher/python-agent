import { afterEach, describe, expect, it } from "vitest";
import { flushPromises, mount, type VueWrapper } from "@vue/test-utils";
import AgentSettings from "./AgentSettings.vue";

const wrappers: VueWrapper[] = [];

function mountSettings() {
  const wrapper = mount(AgentSettings, {
    attachTo: document.body,
    props: { controlsId: "test-settings", modelLabel: "model-a", roleLabel: "Разработчик", effort: "auto" },
  });
  wrappers.push(wrapper);
  return wrapper;
}

afterEach(() => {
  for (const wrapper of wrappers.splice(0)) wrapper.unmount();
  document.body.innerHTML = "";
});

describe("Параметры агента", () => {
  it("показывает одну сводку и обновляет её вместе с параметрами", async () => {
    const wrapper = mountSettings();
    const button = wrapper.get("button");
    expect(button.text()).toBe("model-a · Разработчик · Авто");
    expect(button.attributes("aria-controls")).toBe("test-settings");
    expect(button.attributes("aria-expanded")).toBe("false");
    await wrapper.setProps({ modelLabel: "model-b", roleLabel: "Консультант", effort: "high" });
    expect(button.text()).toBe("model-b · Консультант · Высокий");
    expect(button.attributes("title")).toBe(button.text());
  });

  it("раскрывает настройки и переводит фокус на первое доступное поле", async () => {
    const wrapper = mountSettings();
    const panel = wrapper.get("#test-settings").element as HTMLElement;
    const unavailable = document.createElement("select");
    unavailable.disabled = true;
    const field = document.createElement("select");
    panel.append(unavailable, field);
    expect(panel.hidden).toBe(true);
    await wrapper.get("button").trigger("click");
    await flushPromises();
    expect(panel.hidden).toBe(false);
    expect(document.activeElement).toBe(field);
  });

  it("закрывается по Escape и возвращает фокус на кнопку", async () => {
    const wrapper = mountSettings();
    const field = document.createElement("select");
    wrapper.get("#test-settings").element.appendChild(field);
    const button = wrapper.get("button");
    await button.trigger("click");
    await flushPromises();
    field.dispatchEvent(new KeyboardEvent("keydown", { key: "Escape", bubbles: true, cancelable: true }));
    await flushPromises();
    expect(button.attributes("aria-expanded")).toBe("false");
    expect(document.activeElement).toBe(button.element);
  });

  it("не закрывается внутри панели, но закрывается при клике снаружи", async () => {
    const wrapper = mountSettings();
    const button = wrapper.get("button");
    await button.trigger("click");
    wrapper.get("#test-settings").element.dispatchEvent(new PointerEvent("pointerdown", { bubbles: true }));
    await flushPromises();
    expect(button.attributes("aria-expanded")).toBe("true");
    document.body.dispatchEvent(new PointerEvent("pointerdown", { bubbles: true }));
    await flushPromises();
    expect(button.attributes("aria-expanded")).toBe("false");
  });

  it("закрывается при выходе фокуса и сохраняет фокус снаружи", async () => {
    const wrapper = mountSettings();
    const outside = document.createElement("input");
    document.body.appendChild(outside);
    const field = document.createElement("select");
    wrapper.get("#test-settings").element.appendChild(field);
    const button = wrapper.get("button");
    await button.trigger("click");
    await flushPromises();
    outside.focus();
    await flushPromises();
    expect(button.attributes("aria-expanded")).toBe("false");
    expect(document.activeElement).toBe(outside);
  });

  it("повторное нажатие закрывает панель без удаления её содержимого", async () => {
    const wrapper = mountSettings();
    const panel = wrapper.get("#test-settings").element;
    panel.appendChild(document.createElement("select"));
    const button = wrapper.get("button");
    await button.trigger("click");
    await button.trigger("click");
    expect(button.attributes("aria-expanded")).toBe("false");
    expect(panel.childElementCount).toBe(1);
  });

  it("закрывается при потере фокуса без следующего элемента", async () => {
    const wrapper = mountSettings();
    const field = document.createElement("select");
    wrapper.get("#test-settings").element.appendChild(field);
    const button = wrapper.get("button");
    await button.trigger("click");
    await flushPromises();
    field.blur();
    await flushPromises();
    expect(button.attributes("aria-expanded")).toBe("false");
  });
});
