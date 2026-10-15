import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { Realtime, type ConnectionStatus, type RealtimeSocket } from "./realtime";

class FakeSocket implements RealtimeSocket {
  readyState = 0;
  onopen: (() => void) | null = null;
  onmessage: ((event: { data: unknown }) => void) | null = null;
  onclose: ((event: { code?: number; reason?: string }) => void) | null = null;
  onerror: ((event: unknown) => void) | null = null;
  closeCode: number | undefined;
  sent: string[] = [];

  constructor(public readonly url: string) {}

  send(data: string): void {
    this.sent.push(data);
  }

  close(code?: number): void {
    this.closeCode = code;
  }

  openFromServer(): void {
    this.readyState = 1;
    this.onopen?.();
  }

  receive(data: unknown): void {
    this.onmessage?.({ data });
  }

  closeFromServer(code?: number): void {
    this.onclose?.({ code });
  }
}

describe("Realtime", () => {
  let sockets: FakeSocket[];
  let realtime: Realtime;
  const events: unknown[] = [];
  const statuses: ConnectionStatus[] = [];
  let unauthorized = 0;
  let checkSession: () => Promise<boolean | undefined>;

  function makeRealtime(overrides: Partial<{ checkSession: () => Promise<boolean | undefined> }> = {}): Realtime {
    sockets = [];
    events.length = 0;
    statuses.length = 0;
    unauthorized = 0;
    checkSession = overrides.checkSession ?? (async () => true);
    return new Realtime({
      createSocket: (url) => {
        const socket = new FakeSocket(url);
        sockets.push(socket);
        return socket;
      },
      onEvent: (event) => events.push(event),
      onStatus: (status) => statuses.push(status),
      onUnauthorized: () => {
        unauthorized += 1;
      },
      checkSession,
    });
  }

  beforeEach(() => {
    vi.useFakeTimers();
    realtime = makeRealtime();
  });

  afterEach(() => {
    realtime.close();
    vi.useRealTimers();
  });

  it("transitions connecting → live on open and resets the backoff", () => {
    realtime.connect();
    expect(statuses).toEqual(["connecting"]);
    sockets[0].openFromServer();
    expect(statuses).toEqual(["connecting", "live"]);
    expect(realtime.connectionStatus).toBe("live");
  });

  it("reconnects with growing delays and resets after a successful open", () => {
    realtime.connect();
    sockets[0].openFromServer();
    sockets[0].closeFromServer(1006);
    expect(statuses.at(-1)).toBe("offline");

    vi.advanceTimersByTime(999);
    expect(sockets.length).toBe(1); // 1 s not reached yet

    vi.advanceTimersByTime(1);
    expect(sockets.length).toBe(2); // first reconnect after 1 s
    sockets[1].openFromServer();
    expect(realtime.connectionStatus).toBe("live");

    sockets[1].closeFromServer(1006);
    vi.advanceTimersByTime(2000);
    expect(sockets.length).toBe(3); // second reconnect after 2 s (backoff grew)
  });

  it("caps the reconnect delay at 30 s after a live session and repeated drops", () => {
    realtime.connect();
    sockets[0].openFromServer(); // live once → everConnected stays true thereafter
    sockets[0].closeFromServer(1006); // network drop → first backoff is 1 s
    expect(unauthorized).toBe(0);

    for (let i = 1; i <= 5; i += 1) {
      vi.advanceTimersByTime(Math.min(30_000, 1000 * 2 ** (i - 1)));
      expect(sockets.length).toBe(i + 1); // sockets[i] just opened
      sockets[i].closeFromServer(1006);
    }
    vi.advanceTimersByTime(29_999);
    expect(sockets.length).toBe(6);
    vi.advanceTimersByTime(1);
    expect(sockets.length).toBe(7); // fired after exactly 30 s
  });

  it("parses and forwards server events", () => {
    realtime.connect();
    sockets[0].receive(
      JSON.stringify({
        type: "message.delta",
        session_id: "s1",
        ephemeral: true,
        payload: { stream_id: "abc", seq: 1, delta: "hi" },
      }),
    );
    expect(events).toHaveLength(1);
    const event = events[0] as { type: string; payload: { delta: string } };
    expect(event.type).toBe("message.delta");
    expect(event.payload.delta).toBe("hi");
  });

  it("ignores malformed websocket payloads", () => {
    realtime.connect();
    sockets[0].receive("not json {");
    sockets[0].receive(JSON.stringify({ payload: {} }));
    expect(events).toHaveLength(0);
  });

  it("stops and calls onUnauthorized on a 4401 close without retrying forever", () => {
    realtime.connect();
    sockets[0].closeFromServer(4401);
    expect(unauthorized).toBe(1);
    expect(statuses.at(-1)).toBe("offline");
    vi.advanceTimersByTime(120_000);
    expect(sockets.length).toBe(1); // no reconnect loop
  });

  it("reconnects after an ambiguous pre-open close when auth is still valid", async () => {
    realtime = makeRealtime({ checkSession: async () => true });
    realtime.connect();
    sockets[0].closeFromServer(1006); // ambiguous pre-open drop
    expect(unauthorized).toBe(0); // no logout while the probe is pending

    await vi.advanceTimersByTimeAsync(0); // let the probe resolve

    vi.advanceTimersByTime(1000);
    expect(sockets.length).toBe(2);
    expect(unauthorized).toBe(0);
  });

  it("stops and calls onUnauthorized exactly once when the probe confirms auth is gone", async () => {
    realtime = makeRealtime({ checkSession: async () => false });
    realtime.connect();
    sockets[0].closeFromServer(1006); // ambiguous pre-open drop
    expect(unauthorized).toBe(0); // not decided yet

    await vi.advanceTimersByTimeAsync(0); // probe resolves to unauthenticated
    expect(unauthorized).toBe(1);
    expect(statuses.at(-1)).toBe("offline");

    vi.advanceTimersByTime(120_000);
    expect(sockets.length).toBe(1); // stopped, no reconnect loop
    expect(unauthorized).toBe(1); // exactly once
  });

  it("retries with backoff (no logout) when the probe fails on the network", async () => {
    realtime = makeRealtime({ checkSession: async () => undefined });
    realtime.connect();
    sockets[0].closeFromServer(1006);
    expect(unauthorized).toBe(0);

    await vi.advanceTimersByTimeAsync(0); // probe resolves to unknown (network failure)
    expect(unauthorized).toBe(0); // no logout on network failure

    vi.advanceTimersByTime(1000);
    expect(sockets.length).toBe(2); // backoff retry
    expect(realtime.connectionStatus).toBe("connecting");
  });

  it("retries with backoff (no logout) when the probe throws", async () => {
    realtime = makeRealtime({
      checkSession: async () => {
        throw new Error("network down");
      },
    });
    realtime.connect();
    sockets[0].closeFromServer(1006);

    await vi.advanceTimersByTimeAsync(0);
    expect(unauthorized).toBe(0); // thrown probe error → not a logout

    vi.advanceTimersByTime(1000);
    expect(sockets.length).toBe(2);
  });

  it("reconnects after a pre-open close with an unistinguished code when auth is valid", async () => {
    realtime = makeRealtime({ checkSession: async () => true });
    realtime.connect();
    sockets[0].closeFromServer(1008); // ambiguous pre-open close
    expect(unauthorized).toBe(0);

    await vi.advanceTimersByTimeAsync(0);
    vi.advanceTimersByTime(1000);
    expect(sockets.length).toBe(2);
    expect(unauthorized).toBe(0);
  });

  it("ignores a stale delayed probe result after close()", async () => {
    realtime = makeRealtime({ checkSession: async () => false });
    realtime.connect();
    sockets[0].closeFromServer(1006); // kicks off a slow probe

    realtime.close(); // user navigated away / manual close
    await vi.advanceTimersByTimeAsync(0); // stale probe resolves
    expect(unauthorized).toBe(0); // must not fire after close()
    expect(realtime.connectionStatus).toBe("offline");
  });

  it("ignores a stale delayed probe result after a new connect()", async () => {
    realtime = makeRealtime({ checkSession: async () => false });
    realtime.connect();
    sockets[0].closeFromServer(1006); // kicks off a slow probe

    realtime.connect(); // reconnect (e.g. after re-login) before the probe resolves
    await vi.advanceTimersByTimeAsync(0); // stale probe resolves
    expect(unauthorized).toBe(0); // new connection is not condemned by the stale probe

    sockets[1].openFromServer();
    expect(realtime.connectionStatus).toBe("live");
  });

  it("does not create duplicate reconnect timers when a probe resolves", async () => {
    realtime = makeRealtime({ checkSession: async () => true });
    realtime.connect();
    sockets[0].closeFromServer(1006);

    await vi.advanceTimersByTimeAsync(0);
    vi.advanceTimersByTime(999);
    expect(sockets.length).toBe(1); // 1 s not reached yet
    vi.advanceTimersByTime(1);
    expect(sockets.length).toBe(2); // exactly one reconnect fired
  });

  it("re-connect() while live closes the previous socket and drops its late events", () => {
    realtime.connect();
    sockets[0].openFromServer();
    expect(realtime.connectionStatus).toBe("live");

    realtime.connect(); // manual reopen (e.g. after re-login)
    expect(sockets[0].closeCode).toBe(1000); // old socket is closed, no leak

    sockets[0].openFromServer(); // stale open must be ignored
    expect(realtime.connectionStatus).toBe("connecting");
    sockets[0].receive(JSON.stringify({ type: "connected" }));
    expect(events).toHaveLength(0); // stale socket events are dropped

    sockets[1].openFromServer();
    expect(realtime.connectionStatus).toBe("live");
  });

  it("reconnects after a network drop once a session has been live", () => {
    realtime.connect();
    sockets[0].openFromServer();
    sockets[0].closeFromServer(1006); // network drop, auth is fine
    expect(unauthorized).toBe(0);
    vi.advanceTimersByTime(1000);
    expect(sockets.length).toBe(2);
  });

  it("close() cancels the reconnect and closes the live socket", () => {
    realtime.connect();
    sockets[0].openFromServer();
    realtime.close();
    expect(sockets[0].closeCode).toBe(1000);
    vi.advanceTimersByTime(120_000);
    expect(sockets.length).toBe(1);
    expect(realtime.connectionStatus).toBe("offline");
  });

  describe("default fetch-based checkSession", () => {
    let fetchSpy: ReturnType<typeof vi.fn>;

    function makeWithDefaultCheck(): Realtime {
      sockets = [];
      events.length = 0;
      statuses.length = 0;
      unauthorized = 0;
      return new Realtime({
        createSocket: (url) => {
          const socket = new FakeSocket(url);
          sockets.push(socket);
          return socket;
        },
        onEvent: (event) => events.push(event),
        onStatus: (status) => statuses.push(status),
        onUnauthorized: () => {
          unauthorized += 1;
        },
      });
    }

    beforeEach(() => {
      fetchSpy = vi.fn();
      vi.stubGlobal("fetch", fetchSpy);
    });

    afterEach(() => {
      vi.unstubAllGlobals();
    });

    function stubResponse(body: unknown, ok = true): void {
      fetchSpy.mockResolvedValue({ ok, json: async () => body } as Response);
    }

    it("uses same-origin credentials on the auth probe", async () => {
      stubResponse({ authenticated: true });
      realtime = makeWithDefaultCheck();
      realtime.connect();
      sockets[0].closeFromServer(1006);
      await vi.advanceTimersByTimeAsync(0);
      expect(fetchSpy).toHaveBeenCalledWith("/auth/session", { credentials: "same-origin" });
      realtime.close();
    });

    it("does not logout or reconnect on HTTP 200 {} body", async () => {
      stubResponse({});
      realtime = makeWithDefaultCheck();
      realtime.connect();
      sockets[0].closeFromServer(1006);
      await vi.advanceTimersByTimeAsync(0);
      expect(unauthorized).toBe(0);
      expect(sockets.length).toBe(1);
      vi.advanceTimersByTime(1000);
      expect(sockets.length).toBe(2);
      expect(realtime.connectionStatus).toBe("connecting");
    });

    it("does not logout or reconnect on HTTP 200 non-boolean authenticated", async () => {
      stubResponse({ authenticated: "false" });
      realtime = makeWithDefaultCheck();
      realtime.connect();
      sockets[0].closeFromServer(1006);
      await vi.advanceTimersByTimeAsync(0);
      expect(unauthorized).toBe(0);
      expect(sockets.length).toBe(1);
      vi.advanceTimersByTime(1000);
      expect(sockets.length).toBe(2);
      expect(realtime.connectionStatus).toBe("connecting");
    });

    it("does not logout or reconnect on HTTP 200 array body", async () => {
      stubResponse([{ authenticated: true }]);
      realtime = makeWithDefaultCheck();
      realtime.connect();
      sockets[0].closeFromServer(1006);
      await vi.advanceTimersByTimeAsync(0);
      expect(unauthorized).toBe(0);
      expect(sockets.length).toBe(1);
      vi.advanceTimersByTime(1000);
      expect(sockets.length).toBe(2);
      expect(realtime.connectionStatus).toBe("connecting");
    });

    it("stops once when the default probe confirms authenticated:false", async () => {
      stubResponse({ authenticated: false });
      realtime = makeWithDefaultCheck();
      realtime.connect();
      sockets[0].closeFromServer(1006);
      await vi.advanceTimersByTimeAsync(0);
      expect(unauthorized).toBe(1);
      expect(statuses.at(-1)).toBe("offline");
      vi.advanceTimersByTime(120_000);
      expect(sockets.length).toBe(1);
      expect(unauthorized).toBe(1);
    });

    it("reconnects with backoff when the default probe confirms authenticated:true", async () => {
      stubResponse({ authenticated: true });
      realtime = makeWithDefaultCheck();
      realtime.connect();
      sockets[0].closeFromServer(1006);
      await vi.advanceTimersByTimeAsync(0);
      expect(unauthorized).toBe(0);
      expect(sockets.length).toBe(1);
      vi.advanceTimersByTime(1000);
      expect(sockets.length).toBe(2);
      expect(realtime.connectionStatus).toBe("connecting");
    });
  });
});
