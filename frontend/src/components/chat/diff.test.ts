import { describe, expect, it } from "vitest";
import { diffLines, diffMarker } from "./diff";

describe("diffLines", () => {
  it("empty inputs produce no lines", () => {
    expect(diffLines("", "")).toEqual([]);
  });

  it("whitespace-only inputs produce a single context line", () => {
    expect(diffLines("  ", "  ")).toEqual([{ type: "context", text: "  " }]);
  });

  it("adds every line when the old text is empty", () => {
    const out = diffLines("", "a\nb\nc");
    expect(out.map((l) => [l.type, l.text])).toEqual([
      ["add", "a"],
      ["add", "b"],
      ["add", "c"],
    ]);
  });

  it("removes every line when the new text is empty", () => {
    const out = diffLines("a\nb", "");
    expect(out.map((l) => [l.type, l.text])).toEqual([
      ["remove", "a"],
      ["remove", "b"],
    ]);
  });

  it("marks identical text as context", () => {
    expect(diffLines("x", "x")).toEqual([{ type: "context", text: "x" }]);
  });

  it("is order-stable on a tie: removals before additions", () => {
    // old: ["a","b"], new: ["b","a"] — LCS length 1 has two alternative walks;
    // the deterministic tie-break must emit remove, context, add.
    const out = diffLines("a\nb", "b\na");
    expect(out.map((l) => [l.type, l.text])).toEqual([
      ["remove", "a"],
      ["context", "b"],
      ["add", "a"],
    ]);
  });

  it("treats an empty lines as a plain (possibly empty) token", () => {
    const out = diffLines("a\n\nb", "a\n\nc");
    // 'a' and '' are common; the last line changes.
    expect(out.some((l) => l.type === "remove" && l.text === "b")).toBe(true);
    expect(out.some((l) => l.type === "add" && l.text === "c")).toBe(true);
    expect(out.filter((l) => l.type === "context").length).toBe(2);
  });

  it("treats a single trailing newline as not producing a phantom empty line", () => {
    // "a\n" and "a" are the same single line; no phantom "" token.
    const out = diffLines("a\n", "a");
    expect(out.map((l) => [l.type, l.text])).toEqual([["context", "a"]]);
  });

  it("preserves a real interior blank line", () => {
    const out = diffLines("a\n\nb", "a\n\nb");
    expect(out.map((l) => [l.type, l.text])).toEqual([
      ["context", "a"],
      ["context", ""],
      ["context", "b"],
    ]);
  });

  it("full rewrite is deterministic: all removals then all additions (large input)", () => {
    const oldText = Array.from({ length: 500 }, (_, i) => `old-${i}`).join("\n");
    const newText = Array.from({ length: 500 }, (_, i) => `new-${i}`).join("\n");
    const out = diffLines(oldText, newText);
    // No common lines -> LCS 0 -> the tie-break emits removals then additions.
    expect(out.filter((l) => l.type === "remove")).toHaveLength(500);
    expect(out.filter((l) => l.type === "add")).toHaveLength(500);
    expect(out.map((l) => l.type)).toEqual(
      Array.from({ length: 500 }, () => "remove").concat(Array.from({ length: 500 }, () => "add")),
    );
    // Deterministic: re-running yields the same sequence.
    expect(out.map((l) => l.type)).toEqual(diffLines(oldText, newText).map((l) => l.type));
  });

  it("removes and adds the exact changed region in the middle", () => {
    const out = diffLines("start\nmid-old\nend", "start\nmid-new\nend");
    expect(out.map((l) => [l.type, l.text])).toEqual([
      ["context", "start"],
      ["remove", "mid-old"],
      ["add", "mid-new"],
      ["context", "end"],
    ]);
  });
});

describe("diffMarker", () => {
  it("maps line types to +/-/space markers", () => {
    expect(diffMarker("remove")).toBe("-");
    expect(diffMarker("add")).toBe("+");
    expect(diffMarker("context")).toBe(" ");
  });
});
