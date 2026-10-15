import { describe, it, expect, beforeEach, vi, afterEach } from "vitest";
import { mount, flushPromises } from "@vue/test-utils";
import { createPinia, setActivePinia } from "pinia";
import App from "./App.vue";
import { api } from "./api/client";
import { loginFromFragment, refreshAuth } from "./auth";

vi.mock("./api/client", () => ({
  api: {
    config: vi.fn(),
    projects: vi.fn(),
    sessions: vi.fn(),
    createProject: vi.fn(),
    createSession: vi.fn(),
    session: vi.fn(),
    messages: vi.fn(),
    children: vi.fn(),
    audit: vi.fn(),
    context: vi.fn(),
    sendMessage: vi.fn(),
    approvals: vi.fn(),
    resolveApproval: vi.fn(),
    stop: vi.fn(),
    archive: vi.fn(),
  },
  ApiError: class ApiError extends Error {
    constructor(message: string, readonly status: number) {
      super(message);
      this.name = "ApiError";
    }
  },
}));

vi.mock("./auth", () => ({
  clearAuth: vi.fn(),
  loginFromFragment: vi.fn().mockResolvedValue({ authenticated: true }),
  logout: vi.fn().mockResolvedValue(undefined),
  refreshAuth: vi.fn().mockResolvedValue({ authenticated: true }),
  resetAuthCache: vi.fn(),
}));

vi.mock("./realtime", () => ({
  Realtime: vi.fn(function MockRealtime() {
    return { connect: vi.fn(), close: vi.fn() };
  }),
}));

vi.mock("./router", () => ({
  confirmAuthenticated: vi.fn(),
  resetAuthCache: vi.fn(),
  newSessionProfile: { value: null },
}));

vi.mock("vue-router", async (importOriginal) => {
  const actual = await importOriginal<typeof import("vue-router")>();
  return {
    ...actual,
    useRouter: vi.fn(() => ({
      push: vi.fn(),
      replace: vi.fn(),
    })),
    useRoute: vi.fn(() => ({ name: "home" })),
  };
});

function mountApp() {
  return mount(App, {
    global: {
      stubs: {
        AppShell: {
          template:
            '<div><slot name="topbar" /><slot name="left" /><slot name="main" /><slot name="right" /></div>',
        },
        Topbar: {
          name: "Topbar",
          template: '<button data-testid="toggle-btn" @click="$emit(\'toggle-theme\')">Toggle</button>',
          emits: ["toggle-theme"],
        },
        LeftSidebar: true,
        RightPanel: {
          props: ["context", "children", "audit", "tools"],
          template: "<div />",
        },
        RouterView: true,
      },
    },
  });
}

describe("App Theme Management", () => {
  const THEME_KEY = "agent:theme";

  let storage: Record<string, string>;

  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(loginFromFragment).mockResolvedValue(null);
    vi.mocked(refreshAuth).mockResolvedValue({ authenticated: true });
    setActivePinia(createPinia());
    storage = {};

    vi.stubGlobal("localStorage", {
      getItem: vi.fn((key: string) => storage[key] || null),
      setItem: vi.fn((key: string, value: string) => { storage[key] = value; }),
      removeItem: vi.fn((key: string) => { delete storage[key]; }),
      clear: vi.fn(() => { storage = {}; }),
    });

    vi.stubGlobal("matchMedia", vi.fn().mockImplementation((query) => ({
      matches: query === "(prefers-color-scheme: dark)",
      media: query,
      onchange: null,
      addListener: vi.fn(),
      removeListener: vi.fn(),
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
      dispatchEvent: vi.fn(),
    })));

    document.documentElement.removeAttribute("data-theme");

    vi.mocked(api.config).mockResolvedValue({
      llm_configured: true,
      llm_model: "test",
      llm_destination: "http://test",
      knowledge_base_enabled: false,
      network_tools_enabled: false,
      network_allowlist: [],
      api_auth_enabled: true,
      execution_mode: "embedded",
      default_llm_profile: "test",
      llm_profiles: [{ name: "test", model: "test", destination: "http://test", configured: true }],
    });
    vi.mocked(api.projects).mockResolvedValue([]);
    vi.mocked(api.sessions).mockResolvedValue([]);
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
    document.documentElement.removeAttribute("data-theme");
  });

  it("sets data-theme='light' on startup when no theme is persisted, even if system is dark", async () => {
    const wrapper = mountApp();
    await flushPromises();
    expect(document.documentElement.getAttribute("data-theme")).toBe("light");
    wrapper.unmount();
  });

  it("sets data-theme='dark' on startup if dark is persisted", async () => {
    storage[THEME_KEY] = "dark";

    const wrapper = mountApp();
    await flushPromises();
    expect(document.documentElement.getAttribute("data-theme")).toBe("dark");
    wrapper.unmount();
  });

  it("toggles theme and updates localStorage correctly", async () => {
    storage[THEME_KEY] = "light";

    const wrapper = mountApp();
    await flushPromises();
    expect(document.documentElement.getAttribute("data-theme")).toBe("light");

    const topbar = wrapper.find('[data-testid="toggle-btn"]');
    expect(topbar.exists()).toBe(true);

    await topbar.trigger("click");
    expect(document.documentElement.getAttribute("data-theme")).toBe("dark");
    expect(storage[THEME_KEY]).toBe("dark");

    await topbar.trigger("click");
    expect(document.documentElement.getAttribute("data-theme")).toBe("light");
    expect(storage[THEME_KEY]).toBe("light");

    wrapper.unmount();
  });

  it("persists theme choice on remount", async () => {
    const wrapper1 = mountApp();
    await flushPromises();
    expect(document.documentElement.getAttribute("data-theme")).toBe("light");
    await wrapper1.find('[data-testid="toggle-btn"]').trigger("click");
    expect(storage[THEME_KEY]).toBe("dark");
    wrapper1.unmount();

    document.documentElement.removeAttribute("data-theme");

    const wrapper2 = mountApp();
    await flushPromises();
    expect(document.documentElement.getAttribute("data-theme")).toBe("dark");
    wrapper2.unmount();
  });
});
