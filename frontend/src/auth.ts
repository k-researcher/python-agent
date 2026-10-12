import { request, setCsrfToken } from "./api";

export interface AuthState {
  authenticated: boolean;
  expires_at?: string;
  csrf_token?: string;
}

function remember(state: AuthState): AuthState {
  setCsrfToken(state.csrf_token ?? "");
  return state;
}

/** Exchange a one-time `#login=<code>` link for a session cookie, then drop it from the URL. */
export async function loginFromFragment(): Promise<AuthState | null> {
  const match = /(?:^#|&)login=([^&]+)/.exec(location.hash);
  if (!match) return null;
  history.replaceState(null, "", location.pathname + location.search);
  return remember(
    await request<AuthState>("/auth/login", {
      method: "POST",
      body: JSON.stringify({ code: decodeURIComponent(match[1]) }),
    }),
  );
}

export async function refreshAuth(): Promise<AuthState> {
  return remember(await request<AuthState>("/auth/session"));
}

export async function logout(): Promise<void> {
  await request("/auth/logout", { method: "POST" });
  setCsrfToken("");
}
