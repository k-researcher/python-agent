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
  const modelsChecksum = ref("");
  const projects = ref<Project[]>([]);
  const sessions = ref<Session[]>([]);
  const connection = ref<ConnectionStatus>("offline");
  const selectedProjectId = ref<string | null>(null);
  let generation = 0;
  let configVersion = 0;
  let configFlight: Promise<void> | null = null;
  let configDirty = false;

  function reset(): void {
    generation += 1;
    configVersion += 1;
    configFlight = null;
    configDirty = false;
    modelsChecksum.value = "";
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

  // Active and archived sessions together: the sidebar shows the archive as its own group.
  async function fetchSessions(): Promise<Session[]> {
    const [active, archived] = await Promise.all([api.sessions(), api.archivedSessions()]);
    return [...active, ...archived];
  }

  async function loadSessions(): Promise<void> {
    const expectedGeneration = generation;
    const result = await fetchSessions();
    if (expectedGeneration !== generation) return;
    sessions.value = result;
  }

  /** config + projects + sessions in one round trip; picks a default project. */
  async function load(): Promise<void> {
    const expectedGeneration = generation;
    const expectedConfigVersion = ++configVersion;
    const [cfg, pr, ss] = await Promise.all([api.config(), api.projects(), fetchSessions()]);
    if (expectedGeneration !== generation) return;
    if (expectedConfigVersion === configVersion) config.value = cfg;
    projects.value = pr;
    sessions.value = ss;
    reconcileProject();
  }

  /** Read the public config again, for example after the model settings change. */
  function loadConfig(): Promise<void> {
    if (configFlight) { configDirty = true; return configFlight; }
    const expectedGeneration = generation;
    configFlight = (async () => {
      do {
        configDirty = false;
        const version = ++configVersion;
        const cfg = await api.config();
        if (expectedGeneration !== generation) return;
        if (!configDirty && version === configVersion) config.value = cfg;
      } while (configDirty);
    })().finally(() => {
      if (expectedGeneration === generation) configFlight = null;
    });
    return configFlight;
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

  /** Session IDs of a subtree, the session itself first. */
  function subtreeIds(sessionId: string): string[] {
    const ids = [sessionId];
    for (let index = 0; index < ids.length; index += 1) {
      for (const item of sessions.value) if (item.parent_id === ids[index]) ids.push(item.id);
    }
    return ids;
  }

  async function setArchived(sessionId: string, archived: boolean): Promise<void> {
    await (archived ? api.archive(sessionId) : api.unarchive(sessionId));
    await loadSessions();
  }

  async function deleteSession(sessionId: string): Promise<void> {
    await api.deleteSession(sessionId, subtreeIds(sessionId).length > 1);
    await loadSessions();
  }

  async function deleteProject(projectId: string, name: string): Promise<void> {
    await api.deleteProject(projectId, name);
    await Promise.all([loadProjects(), loadSessions()]);
  }

  async function createSession(input: NewSession): Promise<Session> {
    const expectedGeneration = generation;
    const session = await api.createSession({
      project_id: input.project_id,
      prompt: input.prompt,
      mode: input.mode,
      llm_profile: input.llm_profile,
      ...(input.engine_profile ? { engine_profile: input.engine_profile } : {}),
      reasoning_effort: input.reasoning_effort,
      title: input.title,
      auto_start: input.auto_start,
    });
    if (expectedGeneration === generation) await loadSessions();
    return session;
  }

  /** Root nodes (parent → children) for one project, newest first, archived ones included. */
  function sessionTree(projectId: string | null): SessionNode[] {
    if (projectId == null) return [];
    const own = sessions.value.filter((s) => s.project_id === projectId);
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
      case "settings.models.changed": {
        const checksum = event.payload.checksum;
        if (typeof checksum !== "string" || checksum === modelsChecksum.value) break;
        modelsChecksum.value = checksum;
        void loadConfig().catch(() => undefined);
        break;
      }
      case "project.created":
        void loadProjects();
        break;
      case "project.deleted":
        void loadProjects().catch(() => undefined);
        void loadSessions().catch(() => undefined);
        break;
      case "session.status":
      case "session.error":
      case "session.created":
      case "session.archived":
      case "session.unarchived":
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
    modelsChecksum,
    projects,
    sessions,
    connection,
    selectedProjectId,
    reset,
    load,
    loadConfig,
    createProject,
    createSession,
    sessionTree,
    subtreeIds,
    setArchived,
    deleteSession,
    deleteProject,
    applyEvent,
  };
});
