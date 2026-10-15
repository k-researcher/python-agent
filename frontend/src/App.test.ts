import { describe, expect, it, vi, beforeEach } from "vitest";
import { mount, flushPromises } from "@vue/test-utils";
import { createPinia, setActivePinia, type Pinia } from "pinia";
import { nextTick } from "vue";
import App from "./App.vue";
import { api } from "./api/client";

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

vi.mock("./realtime", () => ({
  Realtime: vi.fn(function MockRealtime() {
    return { connect: vi.fn(), close: vi.fn() };
  }),
}));

vi.mock("./auth", () => ({
  clearAuth: vi.fn(),
  loginFromFragment: vi.fn(() => Promise.resolve(null)),
  logout: vi.fn(() => Promise.resolve()),
  refreshAuth: vi.fn(() => Promise.resolve({ authenticated: true })),
}));

vi.mock("./router", () => ({
  newSessionProfile: { value: null },
  confirmAuthenticated: vi.fn(),
  resetAuthCache: vi.fn(),
}));

vi.mock("vue-router", () => ({
  useRouter: vi.fn(() => ({
    push: vi.fn(),
    replace: vi.fn(),
  })),
  useRoute: vi.fn(() => ({ name: "home" })),
}));

const mockConfig = {
  llm_configured: true,
  llm_model: "test-model",
  llm_destination: "http://localhost:8080",
  knowledge_base_enabled: false,
  network_tools_enabled: true,
  network_allowlist: [],
  api_auth_enabled: true,
  execution_mode: "embedded" as const,
  default_llm_profile: "default",
  llm_profiles: [
    { name: "default", model: "test-model", destination: "http://localhost", configured: true },
  ],
};

const mockProject = {
  id: "p-new",
  name: "Test Project",
  root_path: "/tmp/test",
};

let pinia: Pinia;

function mountApp() {
  return mount(App, {
    global: {
      plugins: [pinia],
      stubs: {
        AppShell: {
          template:
            '<div><slot name="topbar" /><slot name="left" /><slot name="main" /><slot name="right" /></div>',
        },
        Topbar: true,
        RightPanel: {
          props: ["context", "children", "audit", "tools"],
          template: "<div />",
        },
        RouterView: true,
      },
    },
  });
}

/** Mount App and wait for the async bootstrap to finish (auth → load → realtime). */
async function mountReadyApp() {
  const wrapper = mountApp();
  await flushPromises();
  await nextTick();
  return wrapper;
}

async function openCreateForm(wrapper: ReturnType<typeof mountApp>) {
  await wrapper.find("[data-new-project]").trigger("click");
  await nextTick();
}

async function fillCreateForm(
  wrapper: ReturnType<typeof mountApp>,
  name: string,
  rootPath: string,
) {
  await wrapper.find("[data-create-name]").setValue(name);
  await wrapper.find("[data-create-path]").setValue(rootPath);
  await nextTick();
}

async function submitCreateForm(wrapper: ReturnType<typeof mountApp>) {
  await wrapper.find("[data-create-form]").trigger("submit");
  await flushPromises();
  await nextTick();
}

beforeEach(() => {
  pinia = createPinia();
  setActivePinia(pinia);
  vi.clearAllMocks();
  (api.config as ReturnType<typeof vi.fn>).mockResolvedValue(mockConfig);
  (api.projects as ReturnType<typeof vi.fn>).mockResolvedValue([]);
  (api.sessions as ReturnType<typeof vi.fn>).mockResolvedValue([]);
  (api.createProject as ReturnType<typeof vi.fn>).mockReset();
});

describe("App — project creation flow", () => {
  it("opens the create form when Plus is clicked", async () => {
    const wrapper = await mountReadyApp();
    expect(wrapper.find("[data-create-form]").exists()).toBe(false);
    await openCreateForm(wrapper);
    expect(wrapper.find("[data-create-form]").exists()).toBe(true);
  });

  it("submits valid values exactly once", async () => {
    (api.createProject as ReturnType<typeof vi.fn>).mockResolvedValue(mockProject);
    (api.projects as ReturnType<typeof vi.fn>).mockResolvedValue([mockProject]);

    const wrapper = await mountReadyApp();
    await openCreateForm(wrapper);
    await fillCreateForm(wrapper, "My Project", "/path/to/project");
    await submitCreateForm(wrapper);
    expect(api.createProject).toHaveBeenCalledTimes(1);
    expect(api.createProject).toHaveBeenCalledWith("My Project", "/path/to/project");
  });

  it("disables submit while pending and prevents duplicate submission", async () => {
    let resolveCreate!: (value: unknown) => void;
    const deferred = new Promise<unknown>((res) => {
      resolveCreate = res;
    });
    (api.createProject as ReturnType<typeof vi.fn>).mockReturnValue(deferred);

    const wrapper = await mountReadyApp();
    await openCreateForm(wrapper);
    await fillCreateForm(wrapper, "Test", "/test");
    await wrapper.find("[data-create-form]").trigger("submit");
    await nextTick();
    const submitBtn = wrapper.find("[data-create-submit]");
    expect(submitBtn.attributes("disabled")).toBeDefined();
    await wrapper.find("[data-create-form]").trigger("submit");
    await nextTick();
    expect(api.createProject).toHaveBeenCalledTimes(1);
    resolveCreate(mockProject);
    await flushPromises();
    await nextTick();

    expect(wrapper.find("[data-create-form]").exists()).toBe(false);
  });

  it("keeps form, values, and shows error on failed creation", async () => {
    (api.createProject as ReturnType<typeof vi.fn>).mockRejectedValue(
      new Error("Failed to create"),
    );

    const wrapper = await mountReadyApp();
    await openCreateForm(wrapper);
    await fillCreateForm(wrapper, "Keep Me", "/keep/this/path");
    await submitCreateForm(wrapper);
    expect(wrapper.find("[data-testid='project-error']").exists()).toBe(true);
    expect(wrapper.find("[data-testid='project-error']").text()).toBe("Failed to create");
    expect(wrapper.find("[data-create-form]").exists()).toBe(true);
    expect((wrapper.find("[data-create-name]").element as HTMLInputElement).value).toBe(
      "Keep Me",
    );
    expect((wrapper.find("[data-create-path]").element as HTMLInputElement).value).toBe(
      "/keep/this/path",
    );
    expect(wrapper.find("[data-create-submit]").attributes("disabled")).toBeUndefined();
  });

  it("closes and resets the form on success", async () => {
    (api.createProject as ReturnType<typeof vi.fn>).mockResolvedValue(mockProject);
    (api.projects as ReturnType<typeof vi.fn>).mockResolvedValue([mockProject]);

    const wrapper = await mountReadyApp();
    await openCreateForm(wrapper);
    await fillCreateForm(wrapper, "Success", "/success");
    await submitCreateForm(wrapper);
    expect(wrapper.find("[data-create-form]").exists()).toBe(false);
    await openCreateForm(wrapper);
    expect(wrapper.find("[data-create-form]").exists()).toBe(true);
    expect((wrapper.find("[data-create-name]").element as HTMLInputElement).value).toBe("");
    expect((wrapper.find("[data-create-path]").element as HTMLInputElement).value).toBe("");
  });

  it("closes and resets the form on cancel", async () => {
    const wrapper = await mountReadyApp();
    await openCreateForm(wrapper);
    await fillCreateForm(wrapper, "Cancel Me", "/cancel/path");
    await wrapper.find("[data-create-cancel]").trigger("click");
    await nextTick();
    await nextTick();
    expect(wrapper.find("[data-create-form]").exists()).toBe(false);
    await openCreateForm(wrapper);
    expect((wrapper.find("[data-create-name]").element as HTMLInputElement).value).toBe("");
    expect((wrapper.find("[data-create-path]").element as HTMLInputElement).value).toBe("");
  });
});
