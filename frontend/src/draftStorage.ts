let generation = 0;
let storageBlocked = false;

export function draftGeneration(): number {
  return generation;
}

export function readDraft(key: string): string {
  if (!key || storageBlocked) return "";
  try {
    return window.localStorage.getItem(key) ?? "";
  } catch {
    return "";
  }
}

export function writeDraft(key: string, value: string, expectedGeneration: number): void {
  if (!key || storageBlocked || expectedGeneration !== generation) return;
  try {
    window.localStorage.setItem(key, value);
  } catch {
    return;
  }
}

export function removeDraft(key: string, expectedGeneration: number): void {
  if (!key || storageBlocked || expectedGeneration !== generation) return;
  try {
    window.localStorage.removeItem(key);
  } catch {
    return;
  }
}

export function clearDrafts(): void {
  generation += 1;
  try {
    const storage = window.localStorage;
    const keys: string[] = [];
    for (let index = 0; index < storage.length; index += 1) {
      const key = storage.key(index);
      if (key?.startsWith("draft:")) keys.push(key);
    }
    for (const key of keys) storage.removeItem(key);
  } catch {
    storageBlocked = true;
  }
}
