import { describe, expect, it } from "vitest";
import { MODE_LABELS, SESSION_MODES, modeLabel } from "./modes";

describe("session modes", () => {
  it("contains exactly eight modes", () => {
    expect(SESSION_MODES).toEqual(["dev", "ask", "arch", "pm", "qa", "devops", "cybersec", "analyst"]);
    expect(SESSION_MODES).toHaveLength(8);
  });

  it("gives each mode a non-empty unique label", () => {
    const labels = SESSION_MODES.map((mode) => MODE_LABELS[mode]);
    expect(labels.every((label) => label.length > 0)).toBe(true);
    expect(new Set(labels).size).toBe(labels.length);
  });

  it("maps an empty/undefined mode to the dev label", () => {
    expect(modeLabel(undefined)).toBe("Разработчик");
    expect(modeLabel(null)).toBe("Разработчик");
    expect(modeLabel("")).toBe("Разработчик");
  });

  it("maps a known mode to its label", () => {
    expect(modeLabel("ask")).toBe("Консультант");
  });

  it("returns an unknown mode as-is", () => {
    expect(modeLabel("unknown")).toBe("unknown");
  });

  it("does not treat inherited object names as modes", () => {
    expect(modeLabel("toString")).toBe("toString");
    expect(modeLabel("constructor")).toBe("constructor");
  });
});
