"use strict";
/* Runs in the hidden bridge window. Electron 33's main process has no WebSocket, so the single core connection lives
   here and is relayed to the main process, which fans it out to the visible windows. */

const CORE_URL = "ws://127.0.0.1:8765/ws"; // keep in sync with the CSP in bridge.html

const bridge = window.riyaBridge;
const client = new RiyaCore.CoreClient({ url: CORE_URL, WebSocketImpl: WebSocket });
for (const type of RiyaCore.CORE_TYPES) client.on(type, (msg) => bridge.message(msg)); // every core -> body type
client.on("open", () => bridge.status(true));
client.on("close", () => bridge.status(false));
bridge.onSend((msg) => client.send(msg));
client.connect();
