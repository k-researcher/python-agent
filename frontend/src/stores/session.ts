import { computed, ref } from "vue";
import { defineStore } from "pinia";
import { api } from "../api/client";
import {
  isStreamDeltaEvent,
  type Approval,
  type ContextStats,
  type Message,
  type OutboundAudit,
  type ServerEvent,
  type Session,
  type SnapshotStamp,
  type StreamDeltaEvent,
  type TimelineItem,
  type ToolCallView,
  type ToolResult,
} from "../api/types";

interface StreamingState {
  content: string;
  reasoning: string;
}

function parseToolResult(content: string | null): ToolResult | null {
  if (typeof content !== "string" || content.trim() === "") return null;
  let parsed: unknown;
  try {
    parsed = JSON.parse(content);
  } catch {
    return null;
  }
  if (typeof parsed !== "object" || parsed === null) return null;
  const record = parsed as Record<string, unknown>;
  if (typeof record.success !== "boolean") return null;
  const result: ToolResult = { success: record.success };
  if ("output" in record) result.output = record.output;
  if (typeof record.error === "string") result.error = record.error;
  return result;
}

function parseToolArguments(raw: string): Record<string, unknown> {
  if (raw.trim() === "") return {};
  try {
    const parsed: unknown = JSON.parse(raw);
    if (typeof parsed === "object" && parsed !== null) {
      return parsed as Record<string, unknown>;
    }
  } catch {
    // fall through
  }
  return { _invalid_arguments: raw };
}

function toolStatus(
  approval: Approval | null,
  result: ToolResult | null,
): ToolCallView["status"] {
  if (approval) {
    if (approval.status === "pending") return "awaiting_approval";
    if (approval.status === "rejected" || approval.status === "cancelled") return "rejected";
    // approved → the call keeps running below
  }
  if (result) return result.success ? "completed" : "error";
  return "pending";
}

function buildToolCalls(
  message: Message,
  resultsByCall: Map<string, ToolResult>,
  approvalsByCall: Map<string, Approval>,
): ToolCallView[] {
  const raw = message.tool_calls ?? [];
  const views: ToolCallView[] = [];
  for (const call of raw) {
    if (typeof call !== "object" || call === null) continue;
    const id = typeof call.id === "string" ? call.id : "";
    const fn = call.function;
    const fnRecord =
      typeof fn === "object" && fn !== null ? (fn as Record<string, unknown>) : {};
    const name = typeof fnRecord.name === "string" && fnRecord.name !== "" ? fnRecord.name : "tool";
    const argumentsRaw = typeof fnRecord.arguments === "string" ? fnRecord.arguments : "";
    const approval = approvalsByCall.get(id) ?? null;
    const result = resultsByCall.get(id) ?? null;
    views.push({
      id,
      name,
      arguments: parseToolArguments(argumentsRaw),
      argumentsRaw,
      result,
      approval,
      status: toolStatus(approval, result),
    });
  }
  return views;
}

export const useSessionStore = defineStore("session", () => {
  const sessionId = ref<string | null>(null);
  const session = ref<Session | null>(null);
  const messages = ref<Message[]>([]);
  const approvals = ref<Approval[]>([]);
  const streaming = ref<StreamingState | null>(null);
  const context = ref<ContextStats | null>(null);
  const audit = ref<OutboundAudit[]>([]);
  const children = ref<Session[]>([]);

  let activeStreamId: string | null = null;
  let finishedStreamIds = new Set<string>();
  let maxStreamNo = 0;
  let maxStreamId: string | null = null;
  let contentSeq = 0;
  let reasoningSeq = 0;
  let generation = 0;
  let runGeneration = 0;
  let runOpen = true;
  let epoch = 0;
  let latestRevision = 0;
  let acceptedRevision = 0;
  type Reload = { token: number; dirty: boolean; promise: Promise<void> };
  let reload: Reload | null = null;
  const terminal = new Set(["completed", "stopped", "error", "interrupted"]);

  function resetStream(): void {
    if (activeStreamId != null) finishedStreamIds.add(activeStreamId);
    streaming.value = null;
    activeStreamId = null;
    contentSeq = reasoningSeq = 0;
  }

  function confirmRun(next: number, status: string): boolean {
    if (next < runGeneration) return false;
    if (next > runGeneration) {
      if (status !== "running" && status !== "pending" && !terminal.has(status)) return false;
      epoch += 1;
      resetStream();
      finishedStreamIds = new Set();
      maxStreamNo = 0;
      maxStreamId = null;
      runGeneration = next;
      runOpen = true;
    }
    if (!runOpen && !terminal.has(status)) return false;
    if (terminal.has(status)) {
      runOpen = false;
      resetStream();
    }
    return true;
  }

  function clearSessionData(): void {
    messages.value = [];
    approvals.value = [];
    context.value = null;
    audit.value = [];
    children.value = [];
    session.value = null;
    resetStream();
    finishedStreamIds.clear();
    maxStreamId = null;
    maxStreamNo = runGeneration = latestRevision = acceptedRevision = 0;
    runOpen = true;
    epoch += 1;
  }

  function close(): void {
    generation += 1;
    sessionId.value = null;
    clearSessionData();
  }

  function isCurrent(id: string | null, gen: number): boolean {
    return id != null && sessionId.value === id && generation === gen;
  }

  function reloadAll(): Promise<void> {
    const id = sessionId.value;
    const token = generation;
    if (id == null) return Promise.resolve();
    if (reload?.token === token) {
      reload.dirty = true;
      return reload.promise;
    }
    const state: Reload = { token, dirty: false, promise: Promise.resolve() };
    reload = state;
    state.promise = (async () => {
      try {
        do {
          state.dirty = false;
          const capturedEpoch = epoch;
          const [msgs, appr, kids, aud, ctx, sess] = await Promise.all([
            api.messages(id), api.approvals(id), api.children(id), api.audit(id),
            api.context(id), api.session(id),
          ]);
          if (!isCurrent(id, token)) return;
          const stamps: SnapshotStamp[] = [msgs, appr, kids, aud, ctx, sess] as SnapshotStamp[];
          const versioned = stamps.some((item) => item.revision !== undefined);
          const revision = sess.revision ?? 0;
          const nextRun = sess.run_generation ?? 0;
          if (state.dirty || capturedEpoch !== epoch || (versioned && (
            stamps.some((item) => item.consistent === false || item.revision !== revision || item.run_generation !== nextRun) ||
            revision < latestRevision || revision < acceptedRevision || nextRun < runGeneration
          ))) {
            state.dirty = true;
            continue;
          }
          // Apply one clean package. A reload of a closed run cannot open it again.
          const acceptStatus = confirmRun(nextRun, sess.status);
          for (const identity of (msgs as SnapshotStamp).completed_streams ?? []) {
            applyStream({ type: "stream.reset", session_id: id, ephemeral: true, payload: identity });
          }
          messages.value = msgs;
          approvals.value = appr;
          children.value = kids;
          audit.value = aud;
          context.value = ctx;
          if (acceptStatus) session.value = sess;
          else if (session.value) session.value = { ...sess, status: session.value.status, error: session.value.error };
          acceptedRevision = revision;
          latestRevision = Math.max(latestRevision, revision);
        } while (state.dirty && isCurrent(id, token));
      } finally {
        if (reload === state) reload = null;
      }
    })();
    return state.promise;
  }

  async function open(id: string): Promise<void> {
    generation += 1;
    sessionId.value = id;
    clearSessionData();
    await reloadAll();
  }

  async function send(content: string, reasoningEffort?: string): Promise<void> {
    const id = sessionId.value;
    const gen = generation;
    if (id == null) throw new Error("Нет открытой сессии");
    const created = await api.sendMessage(id, content, reasoningEffort);
    if (!isCurrent(id, gen)) return;
    resetStream();
    // A `message.created` WS event may already have reloaded the list (including the
    // new message) before the POST resolves. Dedup by id so the message is added
    // exactly once (self-check point 4).
    messages.value = [...messages.value.filter((m) => m.id !== created.id), created];
    if (session.value) {
      session.value = { ...session.value, status: "pending", error: null };
    }
  }

  async function resolve(
    approvalId: string,
    decision: "approve" | "reject",
    comment?: string,
  ): Promise<void> {
    const id = sessionId.value;
    const gen = generation;
    await api.resolveApproval(approvalId, decision, comment);
    if (!isCurrent(id, gen)) return;
    await reloadAll();
  }

  async function stop(): Promise<void> {
    const id = sessionId.value;
    const gen = generation;
    if (id == null) return;
    epoch += 1;
    runOpen = false;
    resetStream();
    if (session.value) session.value = { ...session.value, status: "stopped" };
    try {
      await api.stop(id);
    } catch {
      if (isCurrent(id, gen)) await reloadAll().catch(() => {});
    }
  }

  function streamNumber(payload: StreamDeltaEvent["payload"]): number | null {
    if (Number.isInteger(payload.stream_no) && (payload.stream_no ?? 0) > 0) return payload.stream_no!;
    return null;
  }

  function applyStream(event: StreamDeltaEvent): void {
    const payload = event.payload;
    if (typeof payload !== "object" || payload === null || typeof payload.stream_id !== "string") return;
    if (!runOpen || (payload.generation ?? 0) !== runGeneration) return;
    const no = streamNumber(payload);
    if (no === null || no < maxStreamNo) return;
    const { stream_id, seq, delta } = payload;
    if (event.type === "stream.reset") {
      finishedStreamIds.add(stream_id);
      if (no > maxStreamNo) {
        resetStream();
        maxStreamNo = no;
        maxStreamId = stream_id;
      } else if (stream_id === activeStreamId) resetStream();
      return;
    }
    if (typeof delta !== "string" || finishedStreamIds.has(stream_id)) return;
    if (no === maxStreamNo && maxStreamId !== null && maxStreamId !== stream_id) return;
    if (no > maxStreamNo || activeStreamId === null) {
      resetStream();
      maxStreamNo = no;
      maxStreamId = stream_id;
      activeStreamId = stream_id;
      streaming.value = { content: "", reasoning: "" };
    }
    const current = streaming.value ?? { content: "", reasoning: "" };
    if (event.type === "message.delta") {
      if (seq == null || seq > contentSeq) {
        if (seq != null) contentSeq = seq;
        current.content += delta;
      }
    } else if (seq == null || seq > reasoningSeq) {
      if (seq != null) reasoningSeq = seq;
      current.reasoning += delta;
    }
    streaming.value = current;
  }

  /** Fold one server event into the open session state. */
  function applyEvent(event: ServerEvent): void {
    if (event.session_id !== sessionId.value) return;
    if (isStreamDeltaEvent(event)) {
      applyStream(event);
      return;
    }
    const payload = event.payload;
    if (typeof payload !== "object" || payload === null) return;
    latestRevision = Math.max(latestRevision, event.id);
    if (reload?.token === generation) reload.dirty = true;
    const nextRun = typeof payload.generation === "number" ? payload.generation : 0;
    switch (event.type) {
      case "session.status":
      case "session.error": {
        const status = typeof payload.status === "string" ? payload.status : session.value?.status ?? "pending";
        if (!confirmRun(nextRun, status)) break;
        if (session.value) session.value = { ...session.value, status,
          run_generation: runGeneration,
          error: typeof payload.error === "string" ? payload.error : session.value.error,
          error_code: typeof payload.reason === "string" ? payload.reason : session.value.error_code,
        };
        break;
      }
      case "message.created":
        if (nextRun === runGeneration && payload.role === "assistant") {
          if (payload.stream_id === activeStreamId || (runGeneration === 0 && payload.stream_id === undefined)) resetStream();
        }
        void reloadAll().catch(() => {});
        break;
      case "approval.required":
      case "approval.resolved":
      case "approval.cancelled":
      case "tool.completed":
      case "outbound.audit":
      case "context.compacted":
      case "session.child_created":
        void reloadAll().catch(() => {});
        break;
      default:
        break;
    }
  }

  /** Merged feed: messages (minus skipped), tool calls with results and approvals, summary, streaming. */
  const timeline = computed<TimelineItem[]>(() => {
    const resultsByCall = new Map<string, ToolResult>();
    for (const m of messages.value) {
      if (m.skipped || m.role !== "tool" || m.tool_call_id == null) continue;
      const result = parseToolResult(m.content);
      if (result) resultsByCall.set(m.tool_call_id, result);
    }
    const approvalsByCall = new Map<string, Approval>();
    for (const a of approvals.value) {
      approvalsByCall.set(a.tool_call_id, a);
    }

    const items: TimelineItem[] = [];
    for (const m of messages.value) {
      if (m.skipped) continue;
      if (m.role === "user") {
        items.push(
          m.kind === "summary" ? { kind: "summary", message: m } : { kind: "user", message: m },
        );
        continue;
      }
      if (m.role === "assistant") {
        const calls = buildToolCalls(m, resultsByCall, approvalsByCall);
        const hasText = m.content != null && m.content.trim() !== "";
        if (hasText || calls.length === 0) {
          items.push({ kind: "assistant", message: m });
        }
        for (const call of calls) {
          items.push({ kind: "tool", call });
        }
      }
      // role === "tool" messages are folded into their assistant card.
    }
    const stream = streaming.value;
    if (stream && (stream.content !== "" || stream.reasoning !== "")) {
      items.push({ kind: "streaming", content: stream.content, reasoning: stream.reasoning });
    }
    return items;
  });

  return {
    sessionId,
    session,
    messages,
    approvals,
    streaming,
    context,
    audit,
    children,
    close,
    open,
    send,
    resolve,
    stop,
    applyEvent,
    reloadAll,
    timeline,
  };
});
