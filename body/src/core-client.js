"use strict";
/** Reconnecting WebSocket client for the Riya core protocol. Never throws on network trouble.
 *  Loads in Node (tests) and as a plain script in a renderer (the hidden core-bridge window). */

const BACKOFF_MS = [500, 1000, 2000, 4000, 8000, 10000];
/** Every core -> body message type (docs/protocol.schema.json core_to_body; a test keeps them equal). */
const CORE_TYPES = ["hello", "settings", "state", "expression", "say", "say_end", "mouth", "memories", "error", "look",
  "eyes", "agents", "confirm_request", "confirm_done", "reminders", "actions", "learning", "health", "setup_result",
  "setup_confirm", "voices", "agent_activity", "presence", "agent_window"];
/** Body -> core types that only mean something now (main-logic.js NOW_ONLY; a test keeps them equal): never queued while
 *  the socket isn't open, and dropped from the queue when it closes, so a stale setup_request or setup_action can't
 *  reach the next connection. */
const NOW_ONLY = ["talk_hotkey", "dont_look", "quit_core", "confirm", "actions_clear", "reminder_cancel", "note_delete",
  "note_delete_style", "twin_reset", "learning_reset", "setup_request", "setup_action", "onboarding_done", "focus_agent"];
const nowOnly = (data) => {
  try {
    const msg = JSON.parse(data);
    return Boolean(msg) && NOW_ONLY.includes(msg.type);
  } catch (e) {
    return false;
  }
};

class CoreClient {
  constructor({ url, WebSocketImpl, setTimeoutFn = (fn, ms) => setTimeout(fn, ms) }) {
    this.url = url;
    this.WebSocketImpl = WebSocketImpl;
    this.setTimeoutFn = setTimeoutFn;
    this.log = (...args) => console.warn("[core-client]", ...args);
    this.handlers = new Map();
    this.queue = [];
    this.ws = null;
    this.attempt = 0;
    this.closed = false;
  }

  on(type, fn) {
    if (!this.handlers.has(type)) this.handlers.set(type, []);
    this.handlers.get(type).push(fn);
  }

  connect() {
    if (this.closed) return;
    let ws;
    try {
      ws = new this.WebSocketImpl(this.url);
    } catch (e) {
      this.log("connect failed", e.message);
      this._scheduleReconnect();
      return;
    }
    this.ws = ws;
    ws.onopen = () => {
      if (this.ws !== ws) return;
      this.attempt = 0;
      this._flush();
      this._emit("open");
    };
    ws.onmessage = (ev) => {
      if (this.ws !== ws) return;
      let msg;
      try {
        msg = JSON.parse(ev.data);
      } catch (e) {
        this.log("ignoring bad JSON from core");
        return;
      }
      if (msg && typeof msg.type === "string") this._emit(msg.type, msg);
    };
    ws.onerror = () => {};
    ws.onclose = () => {
      if (this.ws !== ws) return;
      this.ws = null;
      this.queue = this.queue.filter((data) => !nowOnly(data)); // say, one whose send threw just before the drop
      this._emit("close");
      this._scheduleReconnect();
    };
  }

  send(obj) {
    if (obj && NOW_ONLY.includes(obj.type) && !this._isOpen()) {
      this.log(`dropped ${obj.type}: the core isn't connected`);
      return;
    }
    this.queue.push(JSON.stringify(obj));
    this._flush();
  }

  close() {
    this.closed = true;
    const ws = this.ws;
    this.ws = null;
    if (ws) {
      try { ws.close(); } catch (e) { /* already gone */ }
    }
  }

  _isOpen() {
    return this.ws && this.ws.readyState === (this.WebSocketImpl.OPEN ?? 1);
  }

  _flush() {
    while (this.queue.length && this._isOpen()) {
      try {
        this.ws.send(this.queue[0]);
      } catch (e) {
        this.log("send failed", e.message);
        return;
      }
      this.queue.shift();
    }
  }

  _scheduleReconnect() {
    if (this.closed) return;
    const delay = BACKOFF_MS[Math.min(this.attempt, BACKOFF_MS.length - 1)];
    this.attempt += 1;
    this.setTimeoutFn(() => this.connect(), delay);
  }

  _emit(type, payload) {
    for (const fn of this.handlers.get(type) || []) {
      try {
        fn(payload);
      } catch (e) {
        this.log(`handler for ${type} threw`, e.message);
      }
    }
  }
}

if (typeof module !== "undefined" && module.exports) module.exports = { CoreClient, BACKOFF_MS, CORE_TYPES, NOW_ONLY };
else globalThis.RiyaCore = { CoreClient, BACKOFF_MS, CORE_TYPES, NOW_ONLY };
