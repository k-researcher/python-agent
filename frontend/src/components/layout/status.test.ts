import { describe, expect, it } from "vitest";
import { sessionStatusLabel, statusDotClass } from "./status";

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
