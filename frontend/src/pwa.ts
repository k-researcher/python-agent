// The installed app: the service worker for a fast start, and the window color of the theme.

/** Register the service worker in a built app on a secure page (HTTPS or this computer). */
export function registerServiceWorker(production = import.meta.env.PROD): void {
  if (!production || !window.isSecureContext || !("serviceWorker" in navigator)) return;
  window.addEventListener("load", () => {
    navigator.serviceWorker.register("/sw.js").catch(() => {
      // Without the worker the app works the same; only the fast start is lost.
    });
  });
}

/** Give the window frame of the installed app the background color of the current theme. */
export function syncThemeColor(): void {
  const color = getComputedStyle(document.documentElement).getPropertyValue("--bg").trim();
  if (!color) return;
  const metas = [...document.querySelectorAll<HTMLMetaElement>('meta[name="theme-color"]')];
  // The page starts with one color per system theme; after the choice one color is enough.
  for (const extra of metas.slice(1)) extra.remove();
  const meta = metas[0] ?? document.head.appendChild(Object.assign(document.createElement("meta"), { name: "theme-color" }));
  meta.removeAttribute("media");
  meta.content = color;
}
