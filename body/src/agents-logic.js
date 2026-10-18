"use strict";
/** Pure helpers for the agents area (panel list, window list, buddy badge) and the settings Agents section. Every string
 *  here ends up in the DOM through textContent only: agent names and details come from hook payloads (untrusted).
 *  Loads in Node (tests) and as a plain script in a renderer. */

const AGE_REFRESH_MS = 30_000;
const DAY_MS = 24 * 3600_000;
const MAX_AGENT_LOGS = 20; // the core's limits
const MAX_LOG_NAME = 40;
const MAX_PATTERN_LEN = 300;
const LOG_KEYS = ["name", "glob", "done_re", "fail_re", "lost_re"];
const PATTERN_KEYS = ["done_re", "fail_re", "lost_re"];
const TOKEN_PATH = "%LOCALAPPDATA%\\Riya\\hook_token"; // shown as text, never opened
const SETUP_COMMAND = "python -m riya.agents.install_hooks"; // shown as text to copy, never run by the body
const TONES = { // tone is a CSS class: calm by default, waiting and failed/lost/stalled stand out
  busy: "busy", running: "busy", idle: "dim", unknown: "dim", ended: "dim", waiting: "alert", finished: "good",
  failed: "bad", lost: "bad", stalled: "bad",
};

/** { label, tone } for a status; anything unrecognised is "unknown" (the label never echoes untrusted text). */
function agentChip(status) {
  return typeof status === "string" && Object.hasOwn(TONES, status)
    ? { label: status, tone: TONES[status] }
    : { label: "unknown", tone: "dim" };
}

/** "just now", "5m", "3h", "2d" for an age in milliseconds; "" when it isn't a finite number. */
function relativeAge(ms) {
  if (typeof ms !== "number" || !Number.isFinite(ms)) return "";
  if (ms < 45_000) return "just now";
  const minutes = Math.floor(ms / 60_000); // whole minutes until a full hour, so "0h" can't happen
  if (minutes < 60) return `${Math.max(1, minutes)}m`;
  const hours = Math.floor(ms / 3600_000);
  return hours < 24 ? `${hours}h` : `${Math.floor(hours / 24)}d`;
}

const isItem = (a) => a !== null && typeof a === "object";

/** How many agents want her user's attention (the buddy badge). */
function badgeCount(items) {
  return Array.isArray(items) ? items.filter((a) => isItem(a) && a.needs_attention === true).length : 0;
}

/** A sorted copy: needs attention first, then live agents, then ended ones (the view dims those); stable in each group. */
function sortAgents(items) {
  const rank = (a) => (a.needs_attention === true ? 0 : a.status === "ended" ? 2 : 1);
  return (Array.isArray(items) ? items.filter(isItem) : []).map((a, i) => [a, i])
    .sort(([a, i], [b, j]) => rank(a) - rank(b) || i - j).map(([a]) => a);
}

/** The settings line about the Claude Code hooks, from the core's `hooks_seen_at` (epoch seconds or null). */
function hookStatus(hooksSeenAt, nowMs, connected = true) {
  if (!connected) return "Events received in the last 24 h: unknown (not connected to her core)";
  const answer = typeof hooksSeenAt !== "number" || !Number.isFinite(hooksSeenAt) ? "never"
    : nowMs - hooksSeenAt * 1000 <= DAY_MS ? "yes" : "no";
  return `Events received in the last 24 h: ${answer}`;
}

/** The receiver's problems from an `agents` message, as settings lines (none while disconnected, or when an older core
 *  sends neither field): the port was busy at start (she retries every 60 s), and hooks turned away for a wrong token
 *  in the last 24 h. */
function receiverProblems(msg, connected) {
  if (!connected || !msg || typeof msg !== "object") return [];
  const out = [];
  if (msg.receiver_ok === false) out.push("receiver not running (port 8766 busy)");
  const n = msg.rejected_recent;
  if (Number.isInteger(n) && n > 0) out.push(`events rejected (wrong token): ${n}, re-run the installer`);
  return out;
}

/** The list to show for a connection state: a dropped core means a stale list would pass for live, so it is emptied
 *  (the core sends the agents again on hello). */
function agentsWhileConnected(items, connected) {
  return connected ? items : [];
}

/** Hover text: the detail, else "name: status". Used as a title property, never as HTML. */
function agentTitle(agent) {
  const detail = typeof agent.detail === "string" ? agent.detail.trim() : "";
  return detail || `${agent.name}: ${agent.status}`;
}

/** Ages refresh at most every 30 s, and only while the agents area is visible and has something to show. */
function ageRefreshDue({ last, now, hidden, count }) {
  if (hidden || !(count > 0)) return false;
  return typeof last !== "number" || now - last >= AGE_REFRESH_MS;
}

/** { ok: true, logs } (name and glob trimmed) or { ok: false, error } for the `agent_logs` setting. Mirrors the core's
 *  checks; patterns are compiled as JS regexes (the core re-validates, as with the eyes deny lists). */
function validateAgentLogs(list) {
  if (!Array.isArray(list)) return { ok: false, error: "Agent logs must be a list." };
  if (list.length > MAX_AGENT_LOGS) return { ok: false, error: `Use at most ${MAX_AGENT_LOGS} log entries.` };
  const logs = [];
  const names = new Set();
  for (let i = 0; i < list.length; i++) {
    const e = list[i];
    const at = `Entry ${i + 1}`;
    if (!e || typeof e !== "object" || Array.isArray(e) || Object.keys(e).length !== LOG_KEYS.length
      || !LOG_KEYS.every((k) => typeof e[k] === "string")) return { ok: false, error: `${at} is not a complete log entry.` };
    const name = e.name.trim();
    const glob = e.glob.trim();
    if (!name) return { ok: false, error: `${at} needs a name.` };
    if (name.length > MAX_LOG_NAME) return { ok: false, error: `${at}: the name is longer than ${MAX_LOG_NAME} characters.` };
    if (names.has(name)) return { ok: false, error: `${at}: the name "${name}" is used twice. Names must be unique.` };
    if (!glob) return { ok: false, error: `${at} needs a glob (the files to watch).` };
    for (const k of ["glob", ...PATTERN_KEYS]) {
      if (e[k].length > MAX_PATTERN_LEN) return { ok: false, error: `${at}: ${k} is longer than ${MAX_PATTERN_LEN} characters.` };
    }
    for (const k of PATTERN_KEYS) {
      try {
        new RegExp(e[k]);
      } catch (err) {
        return { ok: false, error: `${at}: ${k} is not a valid regular expression.` };
      }
    }
    names.add(name);
    logs.push({ name, glob, done_re: e.done_re.trim(), fail_re: e.fail_re.trim(), lost_re: e.lost_re.trim() });
  }
  return { ok: true, logs };
}

const api = {
  AGE_REFRESH_MS, TOKEN_PATH, SETUP_COMMAND, LOG_KEYS, MAX_AGENT_LOGS, MAX_LOG_NAME, MAX_PATTERN_LEN,
  agentChip, relativeAge, badgeCount, sortAgents, hookStatus, agentTitle, ageRefreshDue, validateAgentLogs,
  agentsWhileConnected, receiverProblems,
};
if (typeof module !== "undefined" && module.exports) module.exports = api;
else globalThis.RiyaAgents = api;
