import { afterEach, expect, it, vi } from "vitest";
import { registerServiceWorker, syncThemeColor } from "./pwa";

const register = vi.fn(() => Promise.resolve());

function secure(value: boolean): void {
  Object.defineProperty(window, "isSecureContext", { value, configurable: true });
  Object.defineProperty(navigator, "serviceWorker", { value: { register }, configurable: true });
}

afterEach(() => {
  vi.clearAllMocks();
  document.head.innerHTML = "";
  document.documentElement.removeAttribute("style");
});

it("a built app on a secure page registers the worker after the load", () => {
  secure(true);
  registerServiceWorker(true);
  expect(register).not.toHaveBeenCalled();
  window.dispatchEvent(new Event("load"));
  expect(register).toHaveBeenCalledWith("/sw.js");
});

it("the development server and an insecure page get no worker", () => {
  const listen = vi.spyOn(window, "addEventListener");
  secure(true);
  registerServiceWorker(false);
  secure(false);
  registerServiceWorker(true);
  expect(listen).not.toHaveBeenCalledWith("load", expect.anything());
  listen.mockRestore();
});

it("the window color follows the chosen theme", () => {
  document.head.innerHTML = `
    <meta name="theme-color" content="#faf6ef" media="(prefers-color-scheme: light)">
    <meta name="theme-color" content="#1b1714" media="(prefers-color-scheme: dark)">`;
  document.documentElement.style.setProperty("--bg", "#1b1714");
  syncThemeColor();
  const metas = document.querySelectorAll<HTMLMetaElement>('meta[name="theme-color"]');
  expect(metas).toHaveLength(1);
  expect(metas[0].content).toBe("#1b1714");
  expect(metas[0].hasAttribute("media")).toBe(false);
});

it("a page without the color tag gets one", () => {
  document.documentElement.style.setProperty("--bg", "#faf6ef");
  syncThemeColor();
  expect(document.querySelector<HTMLMetaElement>('meta[name="theme-color"]')?.content).toBe("#faf6ef");
});
