import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

async function storageModule() {
  vi.resetModules();
  return import("./draftStorage");
}

let values: Map<string, string>;

beforeEach(() => {
  values = new Map();
  vi.stubGlobal("localStorage", {
    get length() { return values.size; },
    key: (index: number) => [...values.keys()][index] ?? null,
    getItem: (key: string) => values.get(key) ?? null,
    setItem: (key: string, value: string) => values.set(key, value),
    removeItem: (key: string) => values.delete(key),
  });
});

afterEach(() => vi.unstubAllGlobals());

describe("draft access lifetime", () => {
  it("reads and writes only in the current generation", async () => {
    const storage = await storageModule();
    storage.writeDraft("draft:session", "private", storage.draftGeneration());
    expect(storage.readDraft("draft:session")).toBe("private");
  });

  it("clears session and retry drafts while preserving theme and sidebar preferences", async () => {
    const storage = await storageModule();
    values.set("draft:session", "private");
    values.set("draft:new-session:r1", "private retry");
    values.set("agent:theme", "dark");
    values.set("agent:sidebar", "preference");
    storage.clearDrafts();
    expect([...values]).toEqual([["agent:theme", "dark"], ["agent:sidebar", "preference"]]);
  });

  it("an old writer cannot restore private content after the boundary", async () => {
    const storage = await storageModule();
    const previous = storage.draftGeneration();
    storage.clearDrafts();
    storage.writeDraft("draft:session", "private", previous);
    expect(values.has("draft:session")).toBe(false);
  });

  it("an old completion cannot remove the next generation's draft", async () => {
    const storage = await storageModule();
    const previous = storage.draftGeneration();
    storage.clearDrafts();
    storage.writeDraft("draft:session", "new", storage.draftGeneration());
    storage.removeDraft("draft:session", previous);
    expect(storage.readDraft("draft:session")).toBe("new");
  });

  it("failed cleanup blocks later reads even if storage becomes accessible again", async () => {
    const storage = await storageModule();
    vi.stubGlobal("localStorage", { get length() { throw new Error("blocked"); } });
    expect(() => storage.clearDrafts()).not.toThrow();
    vi.stubGlobal("localStorage", { getItem: () => "private", setItem: vi.fn() });
    expect(storage.readDraft("draft:session")).toBe("");
    storage.writeDraft("draft:session", "private", storage.draftGeneration());
    expect(localStorage.setItem).not.toHaveBeenCalled();
  });

  it("storage failures and empty keys do not throw", async () => {
    const storage = await storageModule();
    vi.stubGlobal("localStorage", {
      getItem: () => { throw new Error("blocked"); },
      setItem: () => { throw new Error("blocked"); },
      removeItem: () => { throw new Error("blocked"); },
    });
    expect(storage.readDraft("draft:session")).toBe("");
    expect(storage.readDraft("")).toBe("");
    expect(() => storage.writeDraft("draft:session", "private", storage.draftGeneration())).not.toThrow();
    expect(() => storage.removeDraft("draft:session", storage.draftGeneration())).not.toThrow();
  });
});
