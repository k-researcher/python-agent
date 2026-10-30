import { createRouter, createWebHistory } from "vue-router";
import { ref } from "vue";
import { refreshAuth } from "./auth";
import { ApiError } from "./api/client";
import LoginView from "./views/LoginView.vue";
import HomeView from "./views/HomeView.vue";
import SessionView from "./views/SessionView.vue";
import SettingsModelsView from "./views/SettingsModelsView.vue";

let authenticated: boolean | null = null;
let authGeneration = 0;

export const newSessionProfile = ref<string | null>(null);

function isConfirmedAuthenticated(): boolean {
  return authenticated === true;
}

function confirmAuthenticated(): void {
  authGeneration += 1;
  authenticated = true;
}

/**
 * Cache only successful checks: once signed in we skip the extra request. A failed
 * check is re-run on the next navigation, so logout/login keeps working.
 *
 * Only a *confirmed* refusal (the server answers 401 / authenticated:false) is
 * treated as "logged out". Transient failures (offline, 5xx) must not bounce an
 * already-signed-in user to /login, so they keep their cached/optimistic state.
 */
async function ensureAuthenticated(): Promise<boolean> {
  if (authenticated === true) return true;
  const expectedGeneration = authGeneration;
  try {
    const state = await refreshAuth();
    if (expectedGeneration !== authGeneration) return isConfirmedAuthenticated();
    authenticated = state.authenticated;
  } catch (cause) {
    if (expectedGeneration !== authGeneration) return isConfirmedAuthenticated();
    if (cause instanceof ApiError && cause.status === 401) {
      authenticated = false;
    } else {
      // Transient error — keep the session working instead of logging the user out.
      return true;
    }
  }
  return authenticated;
}

/**
 * Drop the cached auth result. Call this after logout and after a confirmed loss of
 * authorization so the next navigation re-checks instead of trusting a stale `true`.
 */
function resetAuthCache(): void {
  authGeneration += 1;
  authenticated = null;
}

export const router = createRouter({
  history: createWebHistory(),
  routes: [
    { path: "/", name: "home", component: HomeView },
    { path: "/s/:id", name: "session", component: SessionView, props: true },
    { path: "/settings/models", name: "settings-models", component: SettingsModelsView },
    { path: "/login", name: "login", component: LoginView },
    { path: "/:pathMatch(.*)*", redirect: { name: "home" } },
  ],
});

// Unauthenticated visitors are sent to /login from any route.
router.beforeEach(async (to) => {
  if (to.name === "login") return true;
  const ok = await ensureAuthenticated();
  return ok ? true : { name: "login", query: { next: to.fullPath || "/" } };
});

export { confirmAuthenticated, ensureAuthenticated, resetAuthCache };
