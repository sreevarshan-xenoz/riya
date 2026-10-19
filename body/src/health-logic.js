"use strict";
/** Pure helpers for the Health section of the settings window: the checklist rows and their one fix button each, the
 *  online-key editor (the key is never kept here), the Yes/No confirm gate and the voice list. Everything the core says
 *  is shown through textContent by the caller. Loads in Node (tests) and as a plain script in the settings page. */

const REFRESH_MS = 30000;
const MAX_KEY = 200;
const DETAIL_MAX = 120;
const CONFIRM_EXPIRY_MS = 60000; // the core's nonce lifetime: a confirm left open this long is closed, unanswered
const MAX_NONCE = 64;
const CONFIRM_GATED = ["online_test", "eyes_test", "hooks_connect", "hooks_disconnect"];

// One `key: "Label"` pair per line: tests/core/test_health_body_sync.py reads these blocks against the core's rows.
const ROW_LABELS = {
  local: "Local brain",
  online: "Online brain",
  voice: "Voice",
  mic: "Mic",
  look: "Look",
  eyes: "Eyes",
  claude_code: "Claude Code",
  colab: "Colab logs",
  memory: "Memory",
  autostart: "Start with Windows",
};
const FIX_LABELS = {
  online_test: "Test",
  eyes_test: "Test",
  voice_sample: "Play sample",
  mic_check: "Check mic",
  hooks_connect: "Connect",
  hooks_disconnect: "Disconnect",
};
const CONFIRM_TITLES = {
  online_test: "Test the online key?",
  eyes_test: "Test her eyes?",
  hooks_connect: "Connect Claude Code?",
  hooks_disconnect: "Disconnect Claude Code?",
};
const STATUS_TEXT = { ok: "ok", warn: "needs a look", off: "off" };
const KEY_SOURCES = {
  app: "Saved in Riya",
  ".env": "Set via .env (edit that file to remove it)",
  env: "Set via the RIYA_ONLINE_API_KEY environment variable",
};
const NO_KEY = "No key yet";

const has = (obj, key) => typeof key === "string" && Object.prototype.hasOwnProperty.call(obj, key);
const isObject = (x) => x !== null && typeof x === "object";

/** The rows of a `health` message in the core's order: known keys with a known status and a string detail; others are
 *  skipped. detail is cut to 120 characters; fix is a string or null. */
function healthRows(msg) {
  const items = isObject(msg) && Array.isArray(msg.items) ? msg.items : [];
  const rows = [];
  for (const item of items) {
    if (!isObject(item) || !has(ROW_LABELS, item.key) || !has(STATUS_TEXT, item.status) || typeof item.detail !== "string") continue;
    rows.push({
      key: item.key,
      label: ROW_LABELS[item.key],
      status: item.status,
      detail: item.detail.slice(0, DETAIL_MAX),
      fix: typeof item.fix === "string" ? item.fix : null,
    });
  }
  return rows;
}

/** The button for a row's fix: { label, confirm, message }, or null when the row has none (or none we show here). The
 *  Start-with-Windows button flips the state it shows. */
function fixRequest(row, settings) {
  const fix = row && row.fix;
  if (typeof fix !== "string" || !(has(FIX_LABELS, fix) || fix === "autostart")) return null;
  let label = FIX_LABELS[fix];
  let args = {};
  if (fix === "voice_sample") {
    args = isObject(settings) && typeof settings.voice === "string" && settings.voice ? { voice: settings.voice } : {}; // {} samples the current voice
  } else if (fix === "autostart") {
    const turnOn = row.status !== "ok";
    label = turnOn ? "Turn on" : "Turn off";
    args = { on: turnOn };
  }
  return { label, confirm: CONFIRM_GATED.includes(fix), message: { type: "setup_request", action: fix, args } };
}

/** The `online_key_set` message for what was typed: trimmed, non-empty, at most 200 characters, no whitespace or
 *  control characters; else null. */
function keyRequest(raw) {
  if (typeof raw !== "string") return null;
  const key = raw.trim();
  if (!key || key.length > MAX_KEY || /[\s\p{C}]/u.test(key)) return null;
  return { type: "setup_request", action: "online_key_set", args: { key } };
}

/** Send the key in `field` once, and only while the core is connected (a setup_request is dropped while it is away, so
 *  the field keeps the key until a send really happened); then empty the field. True when it was sent. Keeps no copy. */
function submitKey(field, send, connected) {
  if (connected !== true || !field || typeof send !== "function") return false;
  const msg = keyRequest(field.value);
  if (!msg) return false;
  send(msg);
  field.value = "";
  return true;
}

/** Every Save-key button is disabled unless the core is connected. */
const saveKeyEnabled = (connected) => connected === true;

/** Remove is only for a key Riya saved herself; one from .env or the environment is edited there. */
const clearKeyEnabled = (source) => source === "app";

function keySourceText(source) {
  return has(KEY_SOURCES, source) ? KEY_SOURCES[source] : NO_KEY;
}

/** Remembers the gated actions asked for from this window, so a confirm meant for another window (the welcome) is
 *  ignored here, and builds the answer. */
class ConfirmGate {
  constructor() {
    this.pending = new Set();
  }

  request(action) {
    if (CONFIRM_GATED.includes(action)) this.pending.add(action);
  }

  /** A setup_confirm whose action was asked for here, with a usable nonce and text; consumes that request. */
  accepts(msg) {
    if (!isObject(msg) || typeof msg.action !== "string" || !this.pending.has(msg.action)) return false;
    if (typeof msg.nonce !== "string" || msg.nonce.length < 1 || msg.nonce.length > MAX_NONCE || typeof msg.text !== "string") return false;
    this.pending.delete(msg.action);
    return true;
  }

  /** A `setup_result` for `action` arrived. Only a failure can be the answer to a request that never got a confirm (the
   *  core refused to ask); a success is another window's (results go to every window), so it leaves our request alone. */
  release(action, ok) {
    if (ok === false) this.pending.delete(action);
  }

  answer(nonce, yes) {
    return { type: "setup_action", nonce, yes: yes === true };
  }

  clear() {
    this.pending.clear();
  }
}

/** The confirm modal's state: which confirm is on screen, whether it may still be answered, and when it is closed
 *  without an answer. Every open gets its own token; a close or answer for any other token does nothing, so a replaced
 *  dialog's late close event can never answer the newer nonce. The caller shows the dialog and times out `expire`. */
class ConfirmFlow {
  constructor() {
    this.gate = new ConfirmGate();
    this.current = null; // { token, action, nonce } while a modal may be answered
    this.tokens = 0;
  }

  request(action) {
    this.gate.request(action);
  }

  /** A `setup_confirm`: a token for the modal to open (replacing any open one, which can no longer answer), or null when
   *  this window didn't ask for it. */
  open(msg) {
    if (!this.gate.accepts(msg)) return null;
    this.current = { token: ++this.tokens, action: msg.action, nonce: msg.nonce };
    return this.current.token;
  }

  isOpen(token) {
    return this.current !== null && this.current.token === token;
  }

  /** The user's Yes or No for the modal opened as `token`: the `setup_action` to send, once; null if that modal was
   *  replaced, expired, closed by the core or already answered. */
  answer(token, yes) {
    if (!this.isOpen(token)) return null;
    const { nonce } = this.current;
    this.current = null;
    return this.gate.answer(nonce, yes);
  }

  /** The modal for `token` ran out of time: true when it was still open (the caller closes the dialog, sending nothing). */
  expire(token) {
    if (!this.isOpen(token)) return false;
    this.current = null;
    return true;
  }

  /** A `setup_result`: true when the open modal was for that action (the core has resolved it, so close without sending). */
  result(msg) {
    if (!isObject(msg) || typeof msg.action !== "string") return false;
    this.gate.release(msg.action, msg.ok);
    if (this.current === null || this.current.action !== msg.action) return false;
    this.current = null;
    return true;
  }

  /** The core went away: nothing is pending and nothing can be answered. True when a modal was open. */
  disconnect() {
    const wasOpen = this.current !== null;
    this.current = null;
    this.gate.clear();
    return wasOpen;
  }
}

/** The voice names of a `voices` message (non-strings dropped) and the current one. */
function voiceItems(msg) {
  if (!isObject(msg) || msg.type !== "voices" || !Array.isArray(msg.items)) return { items: [], current: null };
  return { items: msg.items.filter((v) => typeof v === "string"), current: typeof msg.current === "string" ? msg.current : null };
}

/** The line shown under the rows after an action finishes. */
function resultText(msg) {
  if (!isObject(msg) || typeof msg.detail !== "string") return "";
  return msg.ok === true ? msg.detail : `Didn't work: ${msg.detail}`;
}

/** Refresh the checklist only while the page is showing and the core is there to answer. */
const shouldRefresh = ({ hidden, connected }) => !hidden && Boolean(connected);

const api = {
  REFRESH_MS, MAX_KEY, CONFIRM_EXPIRY_MS, CONFIRM_GATED, ROW_LABELS, FIX_LABELS, CONFIRM_TITLES, STATUS_TEXT, healthRows,
  fixRequest, keyRequest, submitKey, saveKeyEnabled, clearKeyEnabled, keySourceText, ConfirmGate, ConfirmFlow, voiceItems, resultText,
  shouldRefresh,
};
if (typeof module !== "undefined" && module.exports) module.exports = api;
else globalThis.RiyaHealth = api;
