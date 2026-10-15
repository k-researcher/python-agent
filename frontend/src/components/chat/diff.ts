// Minimal line-wise diff used by ToolCallCard to show old -> new edits.
// No external diff library: a simple LCS-based walk over lines.

export type DiffLineType = "context" | "remove" | "add";

export interface DiffLine {
  type: DiffLineType;
  text: string;
}

/** "".split(/\n/) is [""], which would fake a phantom empty line */
function splitLines(text: string): string[] {
  if (text === "") return [];
  const parts = String(text ?? "").split("\n");
  // A trailing newline yields a phantom empty token; drop it so "a\n" and "a"
  // compare as equal. Interior blank lines are preserved.
  if (parts.length > 1 && parts[parts.length - 1] === "") parts.pop();
  return parts;
}

/**
 * Compute a line diff between two texts.
 * Order is deterministic: for a tie the longer common subsequence decides,
 * otherwise removals are emitted before additions.
 */
export function diffLines(oldText: string, newText: string): DiffLine[] {
  const a = splitLines(oldText);
  const b = splitLines(newText);
  const n = a.length;
  const m = b.length;

  // dp[i][j] = length of the LCS of a[i..] and b[j..].
  const dp: number[][] = Array.from({ length: n + 1 }, () => new Array<number>(m + 1).fill(0));
  for (let i = n - 1; i >= 0; i--) {
    for (let j = m - 1; j >= 0; j--) {
      dp[i][j] = a[i] === b[j] ? dp[i + 1][j + 1] + 1 : Math.max(dp[i + 1][j], dp[i][j + 1]);
    }
  }

  const out: DiffLine[] = [];
  let i = 0;
  let j = 0;
  while (i < n && j < m) {
    if (a[i] === b[j]) {
      out.push({ type: "context", text: a[i] });
      i++;
      j++;
    } else if (dp[i + 1][j] >= dp[i][j + 1]) {
      out.push({ type: "remove", text: a[i] });
      i++;
    } else {
      out.push({ type: "add", text: b[j] });
      j++;
    }
  }
  while (i < n) {
    out.push({ type: "remove", text: a[i] });
    i++;
  }
  while (j < m) {
    out.push({ type: "add", text: b[j] });
    j++;
  }
  return out;
}

/** Pretty line marker: " " unchanged, "-" removed, "+" added. */
export function diffMarker(type: DiffLineType): string {
  switch (type) {
    case "remove":
      return "-";
    case "add":
      return "+";
    default:
      return " ";
  }
}
