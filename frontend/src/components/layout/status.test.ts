import { describe, expect, it } from "vitest";
import { auditStatusLabel, sessionStatusLabel, statusDotClass, statusTone } from "./status";

describe("statusDotClass", () => {
  it("maps the spec statuses to their dedicated dot classes", () => {
    expect(statusDotClass("running")).toBe("status-dot--running");
    expect(statusDotClass("awaiting_confirmation")).toBe("status-dot--awaiting");
    expect(statusDotClass("completed")).toBe("status-dot--completed");
    expect(statusDotClass("error")).toBe("status-dot--error");
  });

  it("falls back to the grey idle dot for any other status", () => {
    // Covers nullish/empty/unknown statuses from the database.
    expect(statusDotClass("")).toBe("status-dot--idle");
    expect(statusDotClass("pending")).toBe("status-dot--idle");
    expect(statusDotClass("stopped")).toBe("status-dot--idle");
    expect(statusDotClass("interrupted")).toBe("status-dot--idle");
    expect(statusDotClass("   ")).toBe("status-dot--idle");
  });
});

describe("sessionStatusLabel", () => {
  it.each([
    ["idle", "Готова"],
    ["pending", "В очереди"],
    ["starting", "Запуск"],
    ["running", "Выполняется"],
    ["awaiting_confirmation", "Ждёт подтверждения"],
    ["completed", "Завершена"],
    ["error", "Ошибка"],
    ["stopped", "Остановлена"],
    ["interrupted", "Прервана"],
  ])("localizes %s", (status, label) => {
    expect(sessionStatusLabel(status)).toBe(label);
  });

  it.each(["", "future_status", "constructor", "__proto__"])("handles unknown %s", (status) => {
    expect(sessionStatusLabel(status)).toBe("Неизвестный статус");
  });
});

describe("auditStatusLabel", () => {
  it("translates the outbound request statuses and keeps an unknown one as it is", () => {
    expect(auditStatusLabel("started")).toBe("Идёт");
    expect(auditStatusLabel("completed")).toBe("Выполнено");
    expect(auditStatusLabel("error")).toBe("Ошибка");
    expect(auditStatusLabel("cancelled")).toBe("Отменено");
    expect(auditStatusLabel("ok")).toBe("ok");
  });
});

describe("statusTone", () => {
  it("gives each known status its colour and leaves the others grey", () => {
    expect(statusTone("running")).toEqual({ "--status-color": "var(--info)" });
    expect(statusTone("completed")).toEqual({ "--status-color": "var(--success)" });
    expect(statusTone("error")).toEqual({ "--status-color": "var(--danger)" });
    expect(statusTone("awaiting_confirmation")).toEqual({ "--status-color": "var(--warning)" });
    expect(statusTone("stopped")).toEqual({});
    expect(statusTone("constructor")).toEqual({});
  });
});
