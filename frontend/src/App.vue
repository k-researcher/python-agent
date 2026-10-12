<script setup lang="ts">
import { computed, onMounted, onUnmounted, ref } from "vue";
import {
  api,
  ApiError,
  type Approval,
  type ContextStats,
  type Message,
  type OutboundAudit,
  type Project,
  type PublicConfig,
  type Session,
} from "./api";
import { loginFromFragment, logout, refreshAuth } from "./auth";

const projects = ref<Project[]>([]);
const sessions = ref<Session[]>([]);
const messages = ref<Message[]>([]);
const approvals = ref<Approval[]>([]);
const children = ref<Session[]>([]);
const audit = ref<OutboundAudit[]>([]);
const contextStats = ref<ContextStats | null>(null);
const config = ref<PublicConfig | null>(null);
const selectedId = ref<string | null>(null);
const connected = ref(false);
const error = ref("");
const authenticated = ref<boolean | null>(null);
const streaming = ref("");

const projectName = ref("");
const projectPath = ref("");
const selectedProjectId = ref("");
const mode = ref("dev");
const selectedLlmProfile = ref("");
const newPrompt = ref("");
const newMessage = ref("");

let socket: WebSocket | null = null;
let reconnectTimer: number | undefined;
let reconnectAttempts = 0;
let disposed = false;

const current = computed(() => sessions.value.find((item) => item.id === selectedId.value));
const canSend = computed(() =>
  current.value && ["completed", "stopped", "error", "interrupted"].includes(current.value.status),
);

async function refresh(): Promise<void> {
  try {
    [config.value, projects.value, sessions.value] = await Promise.all([
      api.config(),
      api.projects(),
      api.sessions(),
    ]);
    if (!selectedProjectId.value && projects.value.length) selectedProjectId.value = projects.value[0].id;
    if (!selectedLlmProfile.value && config.value) {
      selectedLlmProfile.value = config.value.default_llm_profile;
    }
    if (selectedId.value) await refreshCurrent();
  } catch (cause) {
    if (cause instanceof ApiError && cause.status === 401) {
      signedOut();
      return;
    }
    error.value = cause instanceof Error ? cause.message : String(cause);
  }
}

function signedOut(): void {
  authenticated.value = false;
  socket?.close();
}

async function signOut(): Promise<void> {
  await logout().catch(() => undefined);
  signedOut();
}

async function refreshCurrent(): Promise<void> {
  if (!selectedId.value) return;
  [messages.value, approvals.value, children.value, audit.value, contextStats.value] = await Promise.all([
    api.messages(selectedId.value),
    api.approvals(selectedId.value),
    api.children(selectedId.value),
    api.audit(selectedId.value),
    api.context(selectedId.value),
  ]);
}

async function selectSession(id: string): Promise<void> {
  selectedId.value = id;
  await refreshCurrent();
}

async function addProject(): Promise<void> {
  error.value = "";
  try {
    const project = await api.createProject(projectName.value, projectPath.value);
    projectName.value = "";
    projectPath.value = "";
    selectedProjectId.value = project.id;
    await refresh();
  } catch (cause) {
    error.value = cause instanceof Error ? cause.message : String(cause);
  }
}

async function createSession(): Promise<void> {
  if (!selectedProjectId.value || !newPrompt.value.trim()) return;
  error.value = "";
  try {
    const session = await api.createSession(
      selectedProjectId.value,
      newPrompt.value,
      mode.value,
      selectedLlmProfile.value,
    );
    newPrompt.value = "";
    await refresh();
    await selectSession(session.id);
  } catch (cause) {
    error.value = cause instanceof Error ? cause.message : String(cause);
  }
}

async function sendMessage(): Promise<void> {
  if (!selectedId.value || !newMessage.value.trim()) return;
  const content = newMessage.value;
  newMessage.value = "";
  try {
    await api.sendMessage(selectedId.value, content);
    await refresh();
  } catch (cause) {
    newMessage.value = content;
    error.value = cause instanceof Error ? cause.message : String(cause);
  }
}

async function decide(item: Approval, decision: "approve" | "reject"): Promise<void> {
  const comment = decision === "reject" ? window.prompt("Причина отклонения:") ?? "" : "";
  await api.resolveApproval(item.id, decision, comment);
  await refresh();
}

function connect(): void {
  if (reconnectTimer) window.clearTimeout(reconnectTimer);
  const protocol = location.protocol === "https:" ? "wss:" : "ws:";
  // The session cookie authenticates the handshake; no token is sent from JavaScript.
  socket = new WebSocket(`${protocol}//${location.host}/ws`);
  socket.onmessage = (event) => {
    const message = JSON.parse(String(event.data)) as {
      type?: string;
      ephemeral?: boolean;
      session_id?: string;
      payload?: { delta?: string };
    };
    // Streamed text: show it in place, do not reload the state for each fragment.
    if (message.ephemeral) {
      if (message.session_id !== selectedId.value) return;
      if (message.type === "message.delta") streaming.value += message.payload?.delta ?? "";
      if (message.type === "stream.reset") streaming.value = "";
      return;
    }
    if (message.type === "message.created") streaming.value = "";
    if (message.type === "connected") {
      connected.value = true;
      reconnectAttempts = 0;
    }
    void refresh();
  };
  socket.onclose = async (event) => {
    connected.value = false;
    if (disposed || authenticated.value === false) return;
    // A rejected handshake looks like a plain close; ask the server if the login is still valid.
    if (event.code === 4401 || event.code === 1006 || event.code === 1008) {
      const state = await refreshAuth().catch(() => ({ authenticated: false }));
      if (!state.authenticated) {
        signedOut();
        return;
      }
    }
    const delay = Math.min(15_000, 1000 * 2 ** reconnectAttempts);
    reconnectAttempts += 1;
    reconnectTimer = window.setTimeout(connect, delay);
  };
}

onMounted(async () => {
  disposed = false;
  try {
    const state = (await loginFromFragment()) ?? (await refreshAuth());
    authenticated.value = state.authenticated;
  } catch (cause) {
    authenticated.value = false;
    error.value = cause instanceof Error ? cause.message : String(cause);
  }
  if (!authenticated.value) return;
  await refresh();
  connect();
});

onUnmounted(() => {
  disposed = true;
  if (reconnectTimer) window.clearTimeout(reconnectTimer);
  socket?.close();
});
</script>

<template>
  <div v-if="authenticated === false" class="login-required">
    <h1>Нужна ссылка для входа</h1>
    <p>Откройте одноразовую ссылку из консоли агента или выпустите новую командой:</p>
    <pre>uv run agent auth link</pre>
    <p v-if="error" class="error">{{ error }}</p>
  </div>
  <div v-else-if="authenticated" class="layout">
    <aside>
      <div class="brand">
        <span>Python Agent</span>
        <i :class="connected ? 'online' : 'offline'" :title="connected ? 'WebSocket подключён' : 'WebSocket отключён'" />
        <button class="link" type="button" @click="signOut">Выйти</button>
      </div>

      <small v-if="config">
        LLM: {{ config.llm_destination }} · сеть:
        {{ config.network_tools_enabled ? "включена" : "выключена" }} · runtime:
        {{ config.execution_mode }}
      </small>

      <details>
        <summary>Добавить проект</summary>
        <form @submit.prevent="addProject">
          <input v-model="projectName" placeholder="Название" required />
          <input v-model="projectPath" placeholder="Абсолютный путь" required />
          <button>Добавить</button>
        </form>
      </details>

      <details open>
        <summary>Новая сессия</summary>
        <form @submit.prevent="createSession">
          <select v-model="selectedProjectId" required>
            <option disabled value="">Выберите проект</option>
            <option v-for="project in projects" :key="project.id" :value="project.id">
              {{ project.name }}
            </option>
          </select>
          <select v-model="mode">
            <option value="dev">Разработчик</option>
            <option value="ask">Консультант</option>
          </select>
          <select v-model="selectedLlmProfile" required>
            <option
              v-for="profile in config?.llm_profiles ?? []"
              :key="profile.name"
              :value="profile.name"
              :disabled="!profile.configured"
            >
              {{ profile.name }} · {{ profile.model }}
            </option>
          </select>
          <textarea v-model="newPrompt" placeholder="Задача" rows="4" required />
          <button>Создать и запустить</button>
        </form>
      </details>

      <nav>
        <button
          v-for="session in sessions"
          :key="session.id"
          class="session"
          :class="{ selected: session.id === selectedId }"
          @click="selectSession(session.id)"
        >
          <span>{{ session.title }}</span>
          <small :data-status="session.status">{{ session.status }}</small>
        </button>
      </nav>
    </aside>

    <main>
      <p v-if="error" class="error">{{ error }}</p>
      <div v-if="!current" class="empty">Выберите или создайте сессию</div>
      <template v-else>
        <header>
          <div>
            <h1>{{ current.title }}</h1>
            <span>{{ current.mode }} · {{ current.llm_profile }} · {{ current.status }}</span>
          </div>
          <button v-if="['running', 'pending', 'awaiting_confirmation'].includes(current.status)" class="danger" @click="api.stop(current.id)">Остановить</button>
        </header>

        <section v-if="contextStats" class="approvals">
          <article>
            <strong>Контекст и приватность</strong>
            <span>
              ~{{ contextStats.approximate_tokens }} токенов · пропущено сообщений:
              {{ contextStats.omitted_messages }} · LLM: {{ config?.llm_destination }}
            </span>
          </article>
        </section>

        <section v-if="children.length" class="approvals">
          <article>
            <strong>Дочерние агенты</strong>
            <button v-for="child in children" :key="child.id" @click="selectSession(child.id)">
              {{ child.title }} · {{ child.status }}
            </button>
          </article>
        </section>

        <section v-if="approvals.length" class="approvals">
          <article v-for="item in approvals" :key="item.id">
            <strong>Требуется подтверждение: {{ item.tool_name }}</strong>
            <span>Риск: {{ item.risk_level }}</span>
            <pre>{{ JSON.stringify(item.arguments, null, 2) }}</pre>
            <div class="actions">
              <button @click="decide(item, 'approve')">Разрешить</button>
              <button class="danger" @click="decide(item, 'reject')">Отклонить</button>
            </div>
          </article>
        </section>

        <section class="messages">
          <article v-for="message in messages.filter((item) => item.role !== 'system')" :key="message.id" :class="message.role">
            <b>{{ message.role }}</b>
            <pre>{{ message.content }}</pre>
          </article>
          <article v-if="streaming" class="assistant streaming">
            <b>assistant</b>
            <pre>{{ streaming }}</pre>
          </article>
        </section>

        <details v-if="audit.length">
          <summary>Исходящие обращения ({{ audit.length }})</summary>
          <article v-for="item in audit" :key="item.id">
            <b>{{ item.category }} · {{ item.status }}</b>
            <span>{{ item.destination }} · {{ item.payload_bytes }} байт</span>
            <small>SHA-256: {{ item.payload_sha256 }}</small>
          </article>
        </details>

        <form class="composer" @submit.prevent="sendMessage">
          <textarea v-model="newMessage" rows="3" placeholder="Продолжить диалог" :disabled="!canSend" />
          <button :disabled="!canSend || !newMessage.trim()">Отправить</button>
        </form>
      </template>
    </main>
  </div>
</template>
