import { describe, expect, it } from "vitest";
import { languageForPath, languageForOutput } from "./highlight";

describe("languageForPath", () => {
  it.each([
    ["src/agent/repetition.py", "python"],
    ["frontend/src/App.vue", "vue"],
    ["a/b.ts", "typescript"],
    ["config/models.example.yaml", "yaml"],
    ["README.md", "markdown"],
    ["scripts/run.sh", "bash"],
    ["Dockerfile", "docker"],
    ["pyproject.toml", "toml"],
  ])("maps %s to %s", (path, lang) => {
    expect(languageForPath(path)).toBe(lang);
  });

  it("returns null for an unknown or missing path", () => {
    expect(languageForPath("notes.unknownext")).toBeNull();
    expect(languageForPath("")).toBeNull();
    expect(languageForPath(null)).toBeNull();
  });
});

describe("languageForOutput", () => {
  it("uses ANSI colours only when the output has escape codes", () => {
    expect(languageForOutput("\u001b[32mok\u001b[0m")).toBe("ansi");
    expect(languageForOutput("plain text")).toBeNull();
  });
});
