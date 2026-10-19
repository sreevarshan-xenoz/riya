"use strict";
/** Pure helpers for the settings window: badge labels, date text, setting patches, quiet-hours validation (mirroring
 *  the core's HH:MM-HH:MM), and routing a core error to the control that caused it.
 *  Loads in Node (tests) and as a plain script in the settings page. */

const Agents = typeof require === "function" ? require("./agents-logic") : globalThis.RiyaAgents;
const Hands = typeof require === "function" ? require("./hands-logic") : globalThis.RiyaHands;

const MODES = ["buddy", "panel", "window"];
const VOICES = ["alba", "marius", "javert", "jean", "fantine", "cosette", "eponine", "azelma"];
const BOOL_KEYS = ["mute_voice", "mute_mic", "autostart", "online_enabled", "online_consolidation", "may_start_conversations",
  "eyes_enabled", "hands_online"];
const PATTERN_KEYS = ["eyes_deny_titles", "eyes_deny_processes"];
const INT_RANGES = { glance_daily_cap: [1, 1000], glance_min_gap_s: [10, 3600] }; // tighter than the core's limits
const MAX_PATTERNS = 100; // the core's limits
const MAX_PATTERN_LEN = 300;
const HANDS_KEYS = ["tools_enabled", "tools_ask", "apps", "commands", "weather_place"];
const SETTING_KEYS = ["mode", "voice", "volume", "quiet_hours", ...BOOL_KEYS, ...Object.keys(INT_RANGES), ...PATTERN_KEYS, "agent_logs",
  ...HANDS_KEYS];
const QUIET_RE = /^([01]\d|2[0-3]):([0-5]\d)-([01]\d|2[0-3]):([0-5]\d)$/;
const QUIET_ERROR = "Use HH:MM-HH:MM (24-hour), like 23:30-08:00, or leave it empty to turn quiet hours off.";
const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
const SOURCES = {
  user_said: { label: "user said", tone: "user" },
  inferred: { label: "inferred", tone: "inferred" },
  looked_up: { label: "looked up", tone: "lookup" },
};
const NO_HOTKEY = "none — another app is using the shortcuts";

/** Badge for a fact: conflicting facts say so whatever their source. */
function sourceLabel(fact) {
  if (fact && fact.status === "conflict") return { label: "conflicting", tone: "conflict" };
  const source = fact && fact.source;
  if (SOURCES[source]) return { ...SOURCES[source] };
  return { label: typeof source === "string" && source ? source : "unknown", tone: "inferred" };
}

/** "18 Oct 2026, 14:05" from epoch seconds (local time; `utc` for tests), or "unknown". */
function formatWhen(seconds, { utc = false } = {}) {
  if (typeof seconds !== "number" || !Number.isFinite(seconds)) return "unknown";
  const d = new Date(seconds * 1000);
  if (Number.isNaN(d.getTime())) return "unknown";
  const get = (local, universal) => (utc ? d[universal]() : d[local]());
  const pad = (n) => String(n).padStart(2, "0");
  const date = `${get("getDate", "getUTCDate")} ${MONTHS[get("getMonth", "getUTCMonth")]} ${get("getFullYear", "getUTCFullYear")}`;
  return `${date}, ${pad(get("getHours", "getUTCHours"))}:${pad(get("getMinutes", "getUTCMinutes"))}`;
}

/** { ok: true, value: "HH:MM-HH:MM" | null } (blank turns it off) or { ok: false, error }. */
function validateQuietHours(raw) {
  const text = typeof raw === "string" ? raw.trim() : "";
  if (!text) return { ok: true, value: null };
  return QUIET_RE.test(text) ? { ok: true, value: text } : { ok: false, error: QUIET_ERROR };
}

/** { ok: true, patterns } from textarea text, one regex per line (blanks and repeats dropped), each checked as a JS
 *  regex (the core re-validates), or { ok: false, error } naming the first bad line. */
function parsePatternLines(raw) {
  const patterns = [];
  const lines = typeof raw === "string" ? raw.split(/\r?\n/) : [];
  for (let i = 0; i < lines.length; i++) {
    const pattern = lines[i].trim();
    if (!pattern || patterns.includes(pattern)) continue;
    if (pattern.length > MAX_PATTERN_LEN) return { ok: false, error: `Line ${i + 1} is longer than ${MAX_PATTERN_LEN} characters.` };
    try {
      new RegExp(pattern);
    } catch (e) {
      return { ok: false, error: `Line ${i + 1} is not a valid regular expression: ${pattern}` };
    }
    patterns.push(pattern);
  }
  if (patterns.length > MAX_PATTERNS) return { ok: false, error: `Use at most ${MAX_PATTERNS} patterns.` };
  return { ok: true, patterns };
}

/** A whole number within the range of `key`, from a number or numeric text; null if it isn't one. */
function rangedInt(key, raw) {
  const [lo, hi] = INT_RANGES[key];
  const n = typeof raw === "string" ? (raw.trim() ? Number(raw) : NaN) : typeof raw === "number" ? raw : NaN;
  return Number.isInteger(n) && n >= lo && n <= hi ? n : null;
}

/** A validated one-key `set_settings` patch for a control: { ok: true, patch } or { ok: false, error }. */
function settingPatch(key, raw) {
  if (BOOL_KEYS.includes(key)) return { ok: true, patch: { [key]: Boolean(raw) } };
  if (key === "mode") return MODES.includes(raw) ? { ok: true, patch: { mode: raw } } : { ok: false, error: "Pick buddy, panel or window." };
  if (key === "voice") {
    const voice = typeof raw === "string" ? raw.trim() : "";
    return voice ? { ok: true, patch: { voice } } : { ok: false, error: "Enter a voice name." };
  }
  if (key === "volume") {
    const n = typeof raw === "string" && !raw.trim() ? NaN : Number(raw);
    if (!Number.isFinite(n)) return { ok: false, error: "Volume must be a number from 0 to 1." };
    return { ok: true, patch: { volume: Math.round(Math.min(1, Math.max(0, n)) * 100) / 100 } };
  }
  if (key === "quiet_hours") {
    const r = validateQuietHours(raw);
    return r.ok ? { ok: true, patch: { quiet_hours: r.value } } : r;
  }
  if (key in INT_RANGES) {
    const n = rangedInt(key, raw);
    const [lo, hi] = INT_RANGES[key];
    return n === null ? { ok: false, error: `Enter a whole number from ${lo} to ${hi}.` } : { ok: true, patch: { [key]: n } };
  }
  if (PATTERN_KEYS.includes(key)) {
    const r = parsePatternLines(raw);
    return r.ok ? { ok: true, patch: { [key]: r.patterns } } : r;
  }
  if (key === "agent_logs") {
    const r = Agents.validateAgentLogs(raw);
    return r.ok ? { ok: true, patch: { agent_logs: r.logs } } : r;
  }
  if (key === "weather_place") {
    const place = typeof raw === "string" ? raw.trim() : null;
    if (place === null) return { ok: false, error: "Enter a place name." };
    return place.length > Hands.MAX_PLACE ? { ok: false, error: `Use at most ${Hands.MAX_PLACE} characters.` }
      : { ok: true, patch: { weather_place: place || null } };
  }
  if (key === "apps") {
    const r = Hands.validateApps(raw);
    return r.ok ? { ok: true, patch: { apps: r.apps } } : r;
  }
  if (key === "commands") {
    const r = Hands.validateCommands(raw);
    return r.ok ? { ok: true, patch: { commands: r.commands } } : r;
  }
  return { ok: false, error: `${key} can't be changed here.` };
}

/** Options for the voice <select>: the presets, led by the current voice marked "(custom)" if it isn't one. */
function voiceOptions(current) {
  const presets = VOICES.map((v) => ({ value: v, label: v }));
  if (typeof current !== "string" || !current || VOICES.includes(current)) return presets;
  return [{ value: current, label: `${current} (custom)` }, ...presets];
}

/** Holds the latest value offered while a control has focus (rebuilding it would close an open dropdown); `take()`
 *  hands it back on blur. `offer` returns `{ value }` to apply now, or null when it was deferred. */
class DeferWhileFocused {
  constructor() {
    this.pending = null;
  }

  offer(value, focused) {
    if (focused) {
      this.pending = { value };
      return null;
    }
    this.pending = null;
    return { value };
  }

  take() {
    const pending = this.pending;
    this.pending = null;
    return pending;
  }
}

/** "12 of 150 today" from an `eyes` message, or a dash before the core has said. */
function eyesCountText(eyes) {
  if (!eyes || !Number.isFinite(eyes.glances_today) || !Number.isFinite(eyes.cap)) return "—";
  return `${eyes.glances_today} of ${eyes.cap} today`;
}

/** One line about why she isn't looking, from an `eyes` message ("" when she is free to). */
function eyesStatusText(eyes) {
  if (!eyes) return "";
  if (!eyes.enabled) return "Her eyes are off.";
  return eyes.dont_look ? "Don't look is on: she is covering her eyes." : "";
}

/** The glance notes as { id, text, meta } rows (time, app, activity); entries without a numeric id are skipped. */
function glanceRows(msg, opts) {
  const list = msg && Array.isArray(msg.glances) ? msg.glances : [];
  const str = (v) => (typeof v === "string" ? v.trim() : "");
  return list.filter((g) => g && Number.isInteger(g.id)).map((g) => ({
    id: g.id,
    text: str(g.activity) || "(no description)",
    meta: `${formatWhen(g.at, opts)} · ${str(g.app) || "unknown app"}`,
  }));
}

function hotkeyText(accelerator) {
  return typeof accelerator === "string" && accelerator ? accelerator : NO_HOTKEY;
}

function episodeText(episode) {
  const summary = typeof episode.summary === "string" ? episode.summary.trim() : "";
  return summary || "(no summary)";
}

function dueText(thread, opts) {
  return `due ${formatWhen(thread.due_at, opts)}`;
}

/** The lists a `memories` message carries; older cores omit episodes, threads and notes. */
function memoryLists(msg) {
  const list = (v) => (Array.isArray(v) ? v : []);
  return {
    facts: list(msg.facts), episodes: list(msg.episodes), threads: list(msg.threads), days: list(msg.transcript_days),
    notes: list(msg.notes),
  };
}

/** Errors carry only a message, so attribute each one: to the setting it names, else to the latest unanswered action. */
class ErrorRouter {
  constructor() {
    this.pending = [];
  }

  /** An action was sent; `target` is a key such as "setting:mode" or "fact:3". */
  sent(target) {
    this.pending.push(target);
  }

  /** The core answered a request with a settings or memories message. */
  settle() {
    this.pending = [];
  }

  /** The target for an error message ("global" if nothing fits); consumes the pending actions. */
  route(message) {
    const text = typeof message === "string" ? message : "";
    const named = /(?:setting|pattern in) '(\w+)'/.exec(text);
    let target = named && SETTING_KEYS.includes(named[1]) ? `setting:${named[1]}` : null;
    if (!target && /\bagent_logs\b/.test(text)) target = "setting:agent_logs";
    if (!target && /autostart/i.test(text)) target = "setting:autostart";
    if (!target && this.pending.length) target = this.pending[this.pending.length - 1];
    this.pending = [];
    return target || "global";
  }
}

const api = {
  MODES, VOICES, QUIET_ERROR, NO_HOTKEY, sourceLabel, formatWhen, validateQuietHours, settingPatch, voiceOptions, hotkeyText,
  episodeText, dueText, memoryLists, ErrorRouter, DeferWhileFocused, parsePatternLines, eyesCountText, eyesStatusText, glanceRows,
};
if (typeof module !== "undefined" && module.exports) module.exports = api;
else globalThis.RiyaSettings = api;
