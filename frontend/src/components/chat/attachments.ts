// Text files attached to a message: reading, limits and display helpers.
// The limits repeat the server checks so that most mistakes show before sending.

import type { AttachedFile } from "./types";

export const MAX_FILES = 10;
export const MAX_FILE_BYTES = 1024 * 1024;

/** A file that was not attached; ``name`` is null when the reason covers several files. */
export type Rejection = { name: string | null; reason: string };

/** Read a file as UTF-8 text, or return the reason why it cannot be attached. */
export async function readTextFile(file: File): Promise<AttachedFile | string> {
  if (file.size > MAX_FILE_BYTES) {
    const size = formatSize(file.size);
    const limit = formatSize(MAX_FILE_BYTES);
    // Just over the limit both sizes round to the same text; then the size adds nothing.
    const fact = size === limit ? `Файл больше ${limit}` : `Размер ${size}, а можно до ${limit}`;
    return `${fact}. Разделите файл на части.`;
  }
  const bytes = new Uint8Array(await file.arrayBuffer());
  if (bytes.includes(0)) {
    return "Это не текстовый файл. Сейчас можно приложить только текст: код, Markdown, логи, CSV.";
  }
  try {
    const text = new TextDecoder("utf-8", { fatal: true }).decode(bytes);
    return { name: file.name, text, size: bytes.length };
  } catch {
    return "Текст не в кодировке UTF-8. Сохраните файл в UTF-8 и приложите снова.";
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
): Promise<{ files: AttachedFile[]; rejected: Rejection[] }> {
  const next = [...current];
  const rejected: Rejection[] = [];
  for (const file of files) {
    if (next.length >= MAX_FILES) {
      rejected.push({ name: null, reason: `В сообщение помещается до ${MAX_FILES} файлов, остальные не добавлены.` });
      break;
    }
    const result = await readTextFile(file);
    if (typeof result === "string") rejected.push({ name: file.name, reason: result });
    else next.push(result);
  }
  return { files: next, rejected };
}

export function formatSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} Б`;
  if (bytes < 1024 * 1024) return `${Math.max(1, Math.round(bytes / 1024))} КБ`;
  const megabytes = Math.round((bytes / (1024 * 1024)) * 10) / 10;
  return `${String(megabytes).replace(".", ",")} МБ`;
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
