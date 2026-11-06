// Shared and project skills: the list, one skill, changes, import and switches.

import { request } from "./client";

export type SkillScope = "shared" | "project";

export interface SkillItem {
  scope: SkillScope;
  name: string;
  description: string;
  version: string;
  /** The state in the selected project; without a project, the default of a shared skill. */
  enabled: boolean;
  /** The default of a shared skill for all projects; null for a project skill. */
  default_enabled: boolean | null;
  /** The own choice of the project; null when the project follows the default. */
  project_choice: boolean | null;
  /** An enabled project skill with the same name is used instead of this shared skill. */
  replaced: boolean;
  files: number;
  path: string;
}

export interface SkillCatalog {
  shared_dir: string;
  project_dir: string | null;
  skills: SkillItem[];
  errors: { scope: SkillScope; message: string }[];
}

export interface SkillDetail extends SkillItem {
  instructions: string;
  file_list: string[];
  revision: string;
}

export interface SkillDraft {
  description: string;
  instructions: string;
}

const PREFIX = "/api/skills";
const skillPath = (scope: SkillScope, name: string) => `${PREFIX}/${scope}/${encodeURIComponent(name)}`;

function query(values: Record<string, string | null | undefined>): string {
  const params = new URLSearchParams();
  for (const [key, value] of Object.entries(values)) if (value) params.set(key, value);
  const text = params.toString();
  return text ? `?${text}` : "";
}

export const skillsApi = {
  list: (projectId: string | null) => request<SkillCatalog>(`${PREFIX}${query({ project_id: projectId })}`),
  get: (scope: SkillScope, name: string, projectId: string | null) =>
    request<SkillDetail>(`${skillPath(scope, name)}${query({ project_id: projectId })}`),
  create: (scope: SkillScope, projectId: string | null, name: string, draft: SkillDraft) =>
    request<SkillCatalog>(PREFIX, {
      method: "POST",
      body: JSON.stringify({ scope, project_id: projectId, name, ...draft }),
    }),
  update: (scope: SkillScope, name: string, projectId: string | null, draft: SkillDraft, revision: string) =>
    request<SkillCatalog>(skillPath(scope, name), {
      method: "PUT",
      body: JSON.stringify({ project_id: projectId, ...draft, revision }),
    }),
  remove: (scope: SkillScope, name: string, projectId: string | null, revision: string) =>
    request<SkillCatalog>(`${skillPath(scope, name)}${query({ project_id: projectId, revision })}`, {
      method: "DELETE",
    }),
  /** ``enabled`` null removes the own choice of the project, so the default applies again. */
  setSwitch: (scope: SkillScope, name: string, projectId: string | null, enabled: boolean | null) =>
    request<SkillCatalog>(`${skillPath(scope, name)}/switch`, {
      method: "PUT",
      body: JSON.stringify({ project_id: projectId, enabled }),
    }),
  /** Send a SKILL.md file or a zip archive of a skill directory as the raw request body. */
  importFile: (scope: SkillScope, projectId: string | null, file: File) =>
    request<SkillCatalog>(`${PREFIX}/import${query({ scope, project_id: projectId, filename: file.name })}`, {
      method: "POST",
      body: file,
      headers: { "Content-Type": "application/octet-stream" },
    }),
};

/** The same rule as the server: lowercase Latin letters, digits and hyphens, up to 64. */
export const SKILL_NAME = /^[a-z0-9][a-z0-9-]{0,63}$/;
export const MAX_DESCRIPTION = 1024;
export const MAX_IMPORT_BYTES = 5 * 1024 * 1024;
