import type { ServerEvent } from "./api/types";
import { isStreamDeltaEvent } from "./api/types";

export type ConnectionStatus = "connecting" | "live" | "offline";

/** Minimal WebSocket surface shared by the browser type and test fakes. */
export interface RealtimeSocket {
  readonly readyState: number;
  send(data: string): void;
  close(code?: number, reason?: string): void;
  onopen: (() => void) | null;
  onmessage: ((event: { data: unknown }) => void) | null;
  onclose: ((event: { code?: number; reason?: string }) => void) | null;
  onerror: ((event: unknown) => void) | null;
}

export interface RealtimeOptions {
  onEvent?: (event: ServerEvent) => void;
  onStatus?: (status: ConnectionStatus) => void;
  /** Called instead of reconnecting when the server rejects the session (code 4401). */
  onUnauthorized?: () => void;
  /** ws endpoint; defaults to `(wss|ws)://<host>/ws`. */
  url?: string;
  /** Dependency injection for tests. */
  createSocket?: (url: string) => RealtimeSocket;
  /**
   * Probes whether the current browser session is still authenticated.
   * Returns `true` when it is, `false` when the server definitively rejects it,
   * and `undefined` when the check itself fails (network error / 5xx) — in which
   * case the caller must NOT treat the user as logged out.
   */
  checkSession?: () => Promise<boolean | undefined>;
  setTimer?: (callback: () => void, ms: number) => unknown;
  clearTimer?: (handle: unknown) => void;
}

const MAX_RECONNECT_MS = 30_000;

function defaultWsUrl(): string {
  const protocol = typeof location !== "undefined" && location.protocol === "https:" ? "wss:" : "ws:";
  return `${protocol}//${location.host}/ws`;
}

/**
 * Default auth probe: same-origin GET /auth/session with cookies. The session is an
 * HttpOnly cookie and the CSRF token stays in memory (never logged or persisted here).
 */
async function defaultCheckSession(): Promise<boolean | undefined> {
  try {
    const response = await fetch("/auth/session", { credentials: "same-origin" });
    if (!response.ok) return undefined;
    const body: unknown = await response.json().catch(() => null);
    if (typeof body !== "object" || body === null) return undefined;
    const authenticated = (body as { authenticated?: unknown }).authenticated;
    if (typeof authenticated !== "boolean") return undefined;
    return authenticated;
  } catch {
    return undefined;
  }
}

function parseServerEvent(raw: unknown): ServerEvent | null {
  if (typeof raw !== "string") return null;
  let data: unknown;
  try {
    data = JSON.parse(raw);
  } catch {
    return null;
  }
  if (typeof data !== "object" || data === null) return null;
  const record = data as Record<string, unknown>;
  if (typeof record.type !== "string" || record.type.trim() === "") return null;
  if (record.ephemeral !== undefined && record.ephemeral !== true) return null;
  return data as ServerEvent;
}

/**
 * WebSocket transport with exponential reconnect (1 s, 2 s, 4 s … capped at 30 s).
 * The delay resets after a successful open. A rejected session calls `onUnauthorized`
 * and stops, so an unauthenticated client does not retry forever.
 */
export class Realtime {
  private readonly onEvent?: (event: ServerEvent) => void;
  private readonly onStatus?: (status: ConnectionStatus) => void;
  private readonly onUnauthorized?: () => void;
  private readonly url: string;
  private readonly createSocket: (url: string) => RealtimeSocket;
  private readonly checkSession: () => Promise<boolean | undefined>;
  private readonly setTimer: (callback: () => void, ms: number) => unknown;
  private readonly clearTimer: (handle: unknown) => void;

  private socket: RealtimeSocket | null = null;
  private status: ConnectionStatus = "offline";
  private attempts = 0;
  private timer: unknown = null;
  private stopped = true;
  private everConnected = false;
  /** Bumped on connect/close/stop so stale async probe results can be discarded. */
  private generation = 0;

  constructor(options: RealtimeOptions = {}) {
    this.onEvent = options.onEvent;
    this.onStatus = options.onStatus;
    this.onUnauthorized = options.onUnauthorized;
    this.url = options.url ?? defaultWsUrl();
    this.createSocket =
      options.createSocket ?? ((url) => new WebSocket(url) as unknown as RealtimeSocket);
    this.checkSession = options.checkSession ?? defaultCheckSession;
    this.setTimer = options.setTimer ?? ((cb, ms) => window.setTimeout(cb, ms));
    this.clearTimer = options.clearTimer ?? ((handle) => window.clearTimeout(handle as number));
  }

  get connectionStatus(): ConnectionStatus {
    return this.status;
  }

  /** Open (or reopen) the socket and start the reconnect loop. */
  connect(): void {
    this.stopped = false;
    this.everConnected = false;
    this.generation += 1;
    if (this.timer != null) {
      this.clearTimer(this.timer);
      this.timer = null;
    }
    // A previous live socket would otherwise leak: detach its handlers and close it
    // before opening a new one, so re-connect() never leaves a dangling connection.
    const previous = this.socket;
    this.socket = null;
    if (previous) {
      previous.onopen = null;
      previous.onmessage = null;
      previous.onerror = null;
      previous.onclose = null;
      previous.close(1000);
    }
    this.open();
  }

  /** Stop reconnecting and close the socket (no reconnect, no unauthorized callback). */
  close(): void {
    this.stopped = true;
    this.everConnected = false;
    this.generation += 1;
    if (this.timer != null) {
      this.clearTimer(this.timer);
      this.timer = null;
    }
    const socket = this.socket;
    this.socket = null;
    if (socket) {
      socket.onopen = null;
      socket.onmessage = null;
      socket.onerror = null;
      socket.onclose = null;
      socket.close(1000);
    }
    this.setStatus("offline");
  }

  private open(): void {
    if (this.stopped) return;
    this.setStatus("connecting");
    let socket: RealtimeSocket;
    try {
      socket = this.createSocket(this.url);
    } catch {
      // Constructor failed (bad URL); retry on the normal backoff.
      this.scheduleReconnect();
      return;
    }
    this.socket = socket;
    socket.onopen = () => {
      if (this.socket !== socket) return;
      this.everConnected = true;
      this.attempts = 0;
      this.setStatus("live");
    };
    socket.onmessage = (event) => {
      if (this.socket !== socket) return;
      const parsed = parseServerEvent(event.data);
      if (parsed == null) return;
      this.onEvent?.(parsed);
    };
    socket.onclose = (event) => {
      if (this.socket !== socket) return;
      this.socket = null;
      this.setStatus("offline");
      if (this.stopped) return;
      const generation = this.generation;

      if (event?.code === 4401) {
        this.stopUnauthorized();
        return;
      }

      if (!this.everConnected) {
        this.probeAuth(generation);
        return;
      }

      this.scheduleReconnect();
    };
    socket.onerror = () => {
      // The close event always follows; nothing to do here.
    };
  }

  private async probeAuth(generation: number): Promise<void> {
    let authenticated: boolean | undefined;
    try {
      authenticated = await this.checkSession();
    } catch {
      authenticated = undefined;
    }
    if (generation !== this.generation) return;
    if (authenticated === false) {
      this.stopUnauthorized();
      return;
    }
    this.scheduleReconnect();
  }

  /** Confirmed auth rejection: stop and report exactly once. */
  private stopUnauthorized(): void {
    this.stopped = true;
    this.generation += 1;
    if (this.timer != null) {
      this.clearTimer(this.timer);
      this.timer = null;
    }
    const socket = this.socket;
    this.socket = null;
    if (socket) {
      socket.onopen = null;
      socket.onmessage = null;
      socket.onerror = null;
      socket.onclose = null;
      socket.close(1000);
    }
    this.setStatus("offline");
    this.onUnauthorized?.();
  }

  private scheduleReconnect(): void {
    if (this.stopped || this.timer != null) return;
    const delay = Math.min(MAX_RECONNECT_MS, 1000 * 2 ** this.attempts);
    this.attempts += 1;
    this.timer = this.setTimer(() => {
      this.timer = null;
      this.open();
    }, delay);
  }

  private setStatus(status: ConnectionStatus): void {
    if (this.status === status) return;
    this.status = status;
    this.onStatus?.(status);
  }
}

export { isStreamDeltaEvent };
