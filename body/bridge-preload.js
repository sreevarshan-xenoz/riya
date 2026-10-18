"use strict";
/** Preload for the hidden core-bridge window only: the narrow channel between its WebSocket and the main process. */
const { contextBridge, ipcRenderer } = require("electron");

contextBridge.exposeInMainWorld("riyaBridge", {
  message: (msg) => ipcRenderer.send("bridge:message", msg),
  status: (connected) => ipcRenderer.send("bridge:status", Boolean(connected)),
  onSend: (cb) => ipcRenderer.on("bridge:send", (_event, msg) => cb(msg)),
});
