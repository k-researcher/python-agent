// Keeps the app shell for a fast start and for a start without the network.
// It never stores answers of the server API: they always go to the server.

const SHELL = "shell-v1";
const ASSETS = "assets-v1";
// Built files have a hash in the name; old ones stay only up to this count.
const MAX_ASSETS = 80;
const SERVER_PATHS = ["/api", "/auth", "/ws", "/docs", "/redoc", "/openapi.json"];

self.addEventListener("install", (event) => {
  event.waitUntil(caches.open(SHELL).then((cache) => cache.add("/")).then(() => self.skipWaiting()));
});

self.addEventListener("activate", (event) => {
  const keep = new Set([SHELL, ASSETS]);
  event.waitUntil(
    caches
      .keys()
      .then((names) => Promise.all(names.filter((name) => !keep.has(name)).map((name) => caches.delete(name))))
      .then(() => self.clients.claim()),
  );
});

function isServerPath(pathname) {
  return SERVER_PATHS.some((prefix) => pathname === prefix || pathname.startsWith(`${prefix}/`));
}

async function trimAssets() {
  const cache = await caches.open(ASSETS);
  const keys = await cache.keys();
  await Promise.all(keys.slice(0, Math.max(0, keys.length - MAX_ASSETS)).map((key) => cache.delete(key)));
}

// A page: the server first, so a new version shows at once; the saved shell without the network.
async function page(request) {
  try {
    const response = await fetch(request);
    if (response.ok && response.type === "basic" && (response.headers.get("content-type") || "").includes("text/html")) {
      const cache = await caches.open(SHELL);
      await cache.put("/", response.clone());
    }
    return response;
  } catch (error) {
    const saved = await caches.match("/", { cacheName: SHELL });
    if (saved) return saved;
    throw error;
  }
}

// A built file never changes under its name, so the saved copy is used first.
async function asset(request) {
  const saved = await caches.match(request, { cacheName: ASSETS });
  if (saved) return saved;
  const response = await fetch(request);
  if (response.ok && response.type === "basic") {
    const cache = await caches.open(ASSETS);
    await cache.put(request, response.clone());
    await trimAssets();
  }
  return response;
}

self.addEventListener("fetch", (event) => {
  const request = event.request;
  if (request.method !== "GET") return;
  const url = new URL(request.url);
  if (url.origin !== self.location.origin || isServerPath(url.pathname)) return;
  if (request.mode === "navigate") event.respondWith(page(request));
  else if (url.pathname.startsWith("/assets/")) event.respondWith(asset(request));
});
