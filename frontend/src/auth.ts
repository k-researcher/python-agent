import { request, setCsrfToken } from "./api";
import { clearDrafts } from "./draftStorage";

export interface AuthState {
  authenticated: boolean;
  expires_at?: string;
  csrf_token?: string;
}

let generation = 0;

function remember(state: AuthState, expectedGeneration: number): AuthState {
  if (expectedGeneration === generation) setCsrfToken(state.csrf_token ?? "");
  return state;
}

/** Exchange a one-time `#login=<code>` link for a session cookie, then drop it from the URL. */
export async function loginFromFragment(): Promise<AuthState | null> {
  const expectedGeneration = generation;
  const match = /(?:^#|&)login=([^&]+)/.exec(location.hash);
  if (!match) return null;
  history.replaceState(null, "", location.pathname + location.search);
  return remember(
    await request<AuthState>("/auth/login", {
      method: "POST",
      body: JSON.stringify({ code: decodeURIComponent(match[1]) }),
    }),
    expectedGeneration,
  );
}

export async function refreshAuth(): Promise<AuthState> {
  const expectedGeneration = generation;
  return remember(await request<AuthState>("/auth/session"), expectedGeneration);
}

export function clearAuth(): void {
  generation += 1;
  setCsrfToken("");
  clearDrafts();
}

export async function logout(): Promise<void> {
  const expectedGeneration = generation;
  await request("/auth/logout", { method: "POST" });
  if (expectedGeneration === generation) clearAuth();
}
