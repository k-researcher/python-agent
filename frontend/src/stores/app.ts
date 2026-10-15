import { ref } from "vue";
import { defineStore } from "pinia";
import { api } from "../api/client";
import {
  isStreamDeltaEvent,
  type NewSession,
  type Project,
  type PublicConfig,
  type ServerEvent,
  type Session,
  type SessionNode,
} from "../api/types";

export type ConnectionStatus = "connecting" | "live" | "offline";

function byNewest(a: Session, b: Session): number {
  if (a.updated_at !== b.updated_at) return a.updated_at > b.updated_at ? -1 : 1;
  return a.id < b.id ? -1 : 1;
}

export const useAppStore = defineStore("app", () => {
  const config = ref<PublicConfig | null>(null);
  const projects = ref<Project[]>([]);
  const sessions = ref<Session[]>([]);
  const connection = ref<ConnectionStatus>("offline");
  const selectedProjectId = ref<string | null>(null);
  let generation = 0;

  function reset(): void {
    generation += 1;
    config.value = null;
    projects.value = [];
    sessions.value = [];
    selectedProjectId.value = null;
    connection.value = "offline";
  }

  function reconcileProject(): void {
    if (!projects.value.some((project) => project.id === selectedProjectId.value)) {
      selectedProjectId.value = projects.value[0]?.id ?? null;
    }
  }

  // Addressable reloaders: each refreshes exactly one slice of state.
  async function loadProjects(): Promise<void> {
    const expectedGeneration = generation;
    const result = await api.projects();
    if (expectedGeneration !== generation) return;
    projects.value = result;
    reconcileProject();
  }

  async function loadSessions(): Promise<void> {
    const expectedGeneration = generation;
    const result = await api.sessions();
    if (expectedGeneration !== generation) return;
    sessions.value = result;
  }

  /** config + projects + sessions in one round trip; picks a default project. */
  async function load(): Promise<void> {
    const expectedGeneration = generation;
    const [cfg, pr, ss] = await Promise.all([api.config(), api.projects(), api.sessions()]);
    if (expectedGeneration !== generation) return;
    config.value = cfg;
    projects.value = pr;
    sessions.value = ss;
    reconcileProject();
  }

  async function createProject(name: string, rootPath: string): Promise<Project> {
    const expectedGeneration = generation;
    const project = await api.createProject(name, rootPath);
    if (expectedGeneration !== generation) return project;
    // Reload, not push: the list must match server ordering. Only select the new
    // project after the reload succeeds, so a failed reload never leaves a
    // selectedProjectId pointing at an absent project (no half-written state).
    await loadProjects();
    if (expectedGeneration === generation && projects.value.some((item) => item.id === project.id)) {
      selectedProjectId.value = project.id;
    }
    return project;
  }

  async function createSession(input: NewSession): Promise<Session> {
    const expectedGeneration = generation;
    const session = await api.createSession({
      project_id: input.project_id,
      prompt: input.prompt,
      mode: input.mode,
      llm_profile: input.llm_profile,
      reasoning_effort: input.reasoning_effort,
      title: input.title,
      auto_start: input.auto_start,
    });
    if (expectedGeneration === generation) await loadSessions();
    return session;
  }

  /** Root nodes (parent → children) for one project, newest first, archives hidden. */
  function sessionTree(projectId: string | null): SessionNode[] {
    if (projectId == null) return [];
    const own = sessions.value.filter((s) => s.project_id === projectId && !s.archived);
    const childrenByParent = new Map<string | null, Session[]>();
    for (const s of own) {
      const list = childrenByParent.get(s.parent_id);
      if (list) list.push(s);
      else childrenByParent.set(s.parent_id, [s]);
    }
    for (const list of childrenByParent.values()) list.sort(byNewest);
    const build = (parentId: string | null): SessionNode[] =>
      (childrenByParent.get(parentId) ?? []).map((s) => ({
        session: s,
        children: build(s.id),
      }));
    return build(null);
  }

  /** Project/session lifecycle events refresh only what they touch. */
  function applyEvent(event: ServerEvent): void {
    if (isStreamDeltaEvent(event)) return;
    switch (event.type) {
      case "project.created":
        void loadProjects();
        break;
      case "session.status":
      case "session.error":
      case "session.created":
      case "session.archived":
      case "session.child_created":
      case "session.deleted":
        void loadSessions();
        break;
      default:
        break;
    }
  }

  return {
    config,
    projects,
    sessions,
    connection,
    selectedProjectId,
    reset,
    load,
    createProject,
    createSession,
    sessionTree,
    applyEvent,
  };
});
