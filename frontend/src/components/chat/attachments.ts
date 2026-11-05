// Text files attached to a message: reading, limits and display helpers.
// The limits repeat the server checks so that most mistakes show before sending.

import type { AttachedFile } from "./types";

export const MAX_FILES = 10;
export const MAX_FILE_BYTES = 200 * 1024;

/** Read a file as UTF-8 text, or return the reason why it cannot be attached. */
export async function readTextFile(file: File): Promise<AttachedFile | string> {
  if (file.size > MAX_FILE_BYTES) return `Файл ${file.name} больше 200 КБ.`;
  const bytes = new Uint8Array(await file.arrayBuffer());
  if (bytes.includes(0)) return `Файл ${file.name} не текстовый.`;
  try {
    const text = new TextDecoder("utf-8", { fatal: true }).decode(bytes);
    return { name: file.name, text, size: bytes.length };
  } catch {
    return `Файл ${file.name} не в кодировке UTF-8.`;
  }
}

/** The request form of attached files. */
export function toUpload(files: AttachedFile[]): { name: string; text: string }[] {
  return files.map(({ name, text }) => ({ name, text }));
}

/** Add read files to the list; return the new list and the errors to show. */
export async function addFiles(
  current: AttachedFile[],
  files: Iterable<File>,
): Promise<{ files: AttachedFile[]; errors: string[] }> {
  const next = [...current];
  const errors: string[] = [];
  for (const file of files) {
    if (next.length >= MAX_FILES) {
      errors.push(`Можно приложить не больше ${MAX_FILES} файлов.`);
      break;
    }
    const result = await readTextFile(file);
    if (typeof result === "string") errors.push(result);
    else next.push(result);
  }
  return { files: next, errors };
}

export function formatSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} Б`;
  return `${Math.max(1, Math.round(bytes / 1024))} КБ`;
}

/**
 * Return the first ``count`` characters of ``text``. The server counts characters as code
 * points; a JavaScript string index counts UTF-16 units, so emoji take two of them.
 */
export function firstCodePoints(text: string, count: number): string {
  let index = 0;
  for (let seen = 0; seen < count && index < text.length; seen += 1) {
    const point = text.codePointAt(index) ?? 0;
    index += point > 0xffff ? 2 : 1;
  }
  return text.slice(0, index);
}
