import { describe, expect, it } from "vitest";
import { formatLocalTime } from "./time";

const now = new Date("2026-10-29T12:30:00Z");

describe("formatLocalTime", () => {
  it("shows a server UTC time in the given time zone", () => {
    expect(formatLocalTime("2026-10-29T12:09:50.413619Z", { now, timeZone: "Asia/Krasnoyarsk" })).toBe("19:09");
    expect(formatLocalTime("2026-10-29T12:09:50Z", { now, timeZone: "Europe/Moscow" })).toBe("15:09");
  });

  it("adds seconds on request", () => {
    expect(formatLocalTime("2026-10-29T12:09:50Z", { now, timeZone: "Asia/Krasnoyarsk", seconds: true })).toBe(
      "19:09:50",
    );
  });

  it("adds the date when the time is not on the current local day", () => {
    expect(formatLocalTime("2026-10-28T12:09:00Z", { now, timeZone: "Asia/Krasnoyarsk" })).toBe("28.10 19:09");
    // 18:00 UTC is already the next local day in UTC+7.
    expect(formatLocalTime("2026-10-29T18:00:00Z", { now, timeZone: "Asia/Krasnoyarsk" })).toBe("30.10 01:00");
  });

  it("adds the year when the time is in another year", () => {
    expect(formatLocalTime("2025-12-31T10:00:00Z", { now, timeZone: "Asia/Krasnoyarsk" })).toBe("31.12.2025 17:00");
  });

  it("reads a time without a zone as UTC", () => {
    expect(formatLocalTime("2026-10-29T12:09:50", { now, timeZone: "Asia/Krasnoyarsk" })).toBe("19:09");
    expect(formatLocalTime("2026-10-29 12:09:50", { now, timeZone: "Asia/Krasnoyarsk" })).toBe("19:09");
  });

  it("returns an empty string for a missing or invalid value", () => {
    expect(formatLocalTime("", { now })).toBe("");
    expect(formatLocalTime(null, { now })).toBe("");
    expect(formatLocalTime("не время", { now })).toBe("");
  });
});
