// Speech recognition: local server profiles and the transcription call.

import { request } from "./client";

export interface SpeechProfile {
  id: string;
  base_url: string;
  model: string;
  language: string;
  disabled: boolean;
  /** The server configuration has a restart command for this profile. */
  restart: boolean;
}

export interface SpeechSettings {
  profiles: SpeechProfile[];
}

export interface SpeechProfilePatch {
  base_url?: string;
  model?: string;
  language?: string;
  disabled?: boolean;
}

export interface SpeechCheck {
  ok: boolean;
  status: number | null;
  seconds: number | null;
  message?: string;
  /** Set only by a restart: false means the restart command itself failed. */
  restarted?: boolean;
}

const PREFIX = "/api/settings/speech";
const profilePath = (id: string) => `${PREFIX}/profiles/${encodeURIComponent(id)}`;

export const speechSettings = {
  get: () => request<SpeechSettings>(PREFIX),
  putProfile: (id: string, patch: SpeechProfilePatch) =>
    request<SpeechSettings>(profilePath(id), { method: "PUT", body: JSON.stringify(patch) }),
  removeProfile: (id: string) => request<SpeechSettings>(profilePath(id), { method: "DELETE" }),
  makePrimary: (id: string) => request<SpeechSettings>(`${profilePath(id)}/primary`, { method: "POST" }),
  testProfile: (id: string) => request<SpeechCheck>(`${profilePath(id)}/test`, { method: "POST" }),
  restartProfile: (id: string) => request<SpeechCheck>(`${profilePath(id)}/restart`, { method: "POST" }),
};

/** Send a recording as the raw request body and get its text. */
export function transcribe(audio: Blob): Promise<{ text: string; profile: string }> {
  return request("/api/speech/transcribe", {
    method: "POST",
    body: audio,
    headers: { "Content-Type": audio.type || "audio/webm" },
  });
}
