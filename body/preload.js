"use strict";
/** Preload for the visible windows (sandboxed): exposes `window.riya`, a narrow IPC surface to the main process.
 *  The main process owns the core connection; the mode arrives as a `--riya-mode=` argument. */
const { contextBridge, ipcRenderer } = require("electron");

const modeArg = process.argv.find((a) => a.startsWith("--riya-mode="));

contextBridge.exposeInMainWorld("riya", {
  mode: modeArg ? modeArg.slice("--riya-mode=".length) : "buddy",
  /** cb(coreMessage) for every message from the core (the last hello/settings/state/expression are replayed on load). */
  onCoreMessage: (cb) => ipcRenderer.on("core:message", (_event, msg) => cb(msg)),
  /** cb(connected: boolean) whenever the core connection opens or drops (the current state is replayed on load). */
  onConnection: (cb) => ipcRenderer.on("core:status", (_event, connected) => cb(connected)),
  /** Send a body -> core protocol message, e.g. { type: "user_text", text }. */
  send: (msg) => ipcRenderer.send("core:send", msg),
  /** Buddy mode: true makes the window ignore the mouse (still forwarding moves); false makes it clickable. */
  setClickThrough: (ignore) => ipcRenderer.send("click-through", Boolean(ignore)),
  dragStart: () => ipcRenderer.send("drag-start"),
  dragEnd: () => ipcRenderer.send("drag-end"),
  /** Resolves to { dir, manifest } with manifest frame paths as file URLs, or null if no pack is usable. */
  /** Resolves to the active talk-hotkey accelerator string, or null if none could be registered. */
  hotkey: () => ipcRenderer.invoke("hotkey:get"),
  /** Resolves to the active don't-look hotkey accelerator string, or null if none could be registered. */
  dontLookHotkey: () => ipcRenderer.invoke("hotkey:dontlook"),
  getPack: () => ipcRenderer.invoke("pack:get"),
});
