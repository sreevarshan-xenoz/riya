"use strict";
/** Reconnecting WebSocket client for the Riya core protocol. Never throws on network trouble. */

const BACKOFF_MS = [500, 1000, 2000, 4000, 8000, 10000];

class CoreClient {
  constructor({ url, WebSocketImpl, setTimeoutFn = setTimeout }) {
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
      this._emit("close");
      this._scheduleReconnect();
    };
  }

  send(obj) {
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

module.exports = { CoreClient, BACKOFF_MS };
