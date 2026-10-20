"use strict";
/** Pure helpers for her hands: the tool list for the settings page, the apps and commands editors (validation mirrors
 *  the core's limits), command-line splitting, and the reminders and action-log text. Every string here ends up in the
 *  DOM through textContent only. Loads in Node (tests) and as a plain script in a renderer. */

const MAX_APPS = 50; // the core's limits (riya/core/settings.py; a test keeps them equal)
const MAX_COMMANDS = 20;
const MAX_HANDS_NAME = 40;
const MAX_TARGET = 500;
const MAX_ARGV = 20;
const MAX_PLACE = 80;
const TIMEOUT_RANGE = [1, 60];
const ACTION_CAP = 200;
const SOON_MIN = 60; // under this many minutes a reminder reads "in N min"
const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
const DEFAULT_OFF = ["run_command", "write_file"];
const TOOLS = [ // a test keeps the names and tiers equal to riya/core/hands/tools.py TIERS
  { name: "time_now", tier: 0, label: "Tell the time" },
  { name: "timer_set", tier: 0, label: "Set timers" },
  { name: "reminder_set", tier: 0, label: "Set reminders" },
  { name: "reminder_list", tier: 0, label: "List reminders" },
  { name: "reminder_cancel", tier: 0, label: "Cancel reminders" },
  { name: "note_add", tier: 0, label: "Take notes" },
  { name: "note_list", tier: 0, label: "Read notes back" },
  { name: "calc", tier: 0, label: "Calculate" },
  { name: "weather", tier: 0, label: "Check the weather" },
  { name: "lookup", tier: 0, label: "Look things up" },
  { name: "open_url", tier: 1, label: "Open a website" },
  { name: "open_app", tier: 1, label: "Open your apps" },
  { name: "media", tier: 1, label: "Play, pause and skip" },
  { name: "volume", tier: 1, label: "Change the volume" },
  { name: "clipboard_copy", tier: 1, label: "Copy to the clipboard" },
  { name: "run_command", tier: 2, label: "Run your named commands" },
  { name: "write_file", tier: 2, label: "Save text to a file" },
];
const BY_NAME = new Map(TOOLS.map((t) => [t.name, t]));
const isObj = (v) => v !== null && typeof v === "object" && !Array.isArray(v);
const str = (v) => (typeof v === "string" ? v : "");

/** The 17 tools as settings rows: `enabled`, and for T1 an `ask` toggle; T2 rows carry the note "always asks". */
function toolRows(settings) {
  const enabled = isObj(settings && settings.tools_enabled) ? settings.tools_enabled : {};
  const ask = Array.isArray(settings && settings.tools_ask) ? settings.tools_ask : [];
  return TOOLS.map((t) => ({
    name: t.name,
    tier: t.tier,
    label: t.label,
    enabled: typeof enabled[t.name] === "boolean" ? enabled[t.name] : !DEFAULT_OFF.includes(t.name),
    askToggle: t.tier === 1,
    ask: t.tier === 1 && ask.includes(t.name),
    note: t.tier === 2 ? "always asks" : "",
  }));
}

/** A `set_settings` patch for one tool control, or null if it isn't allowed. `enabled` is a partial map (the core merges
 *  it); `ask` edits the T1 list (T2 always asks, others can't). */
function toolsPatch(settings, name, field, value) {
  const tool = BY_NAME.get(name);
  if (!tool) return null;
  if (field === "enabled") return { tools_enabled: { [name]: Boolean(value) } };
  if (field !== "ask" || tool.tier !== 1) return null;
  const list = (Array.isArray(settings && settings.tools_ask) ? settings.tools_ask : []).filter((n) => typeof n === "string");
  const without = list.filter((n) => n !== name);
  return { tools_ask: value ? (list.includes(name) ? list : [...without, name]) : without };
}

/** A command line as argv: spaces split, double quotes group (no escapes: a backslash is just a backslash). An
 *  unbalanced quote (or a non-string) gives null. */
function splitCommandLine(line) {
  if (typeof line !== "string") return null;
  const out = [];
  let cur = "";
  let has = false; // a token has started (so "" is an empty argument)
  let quoted = false;
  for (const ch of line) {
    if (ch === '"') {
      quoted = !quoted;
      has = true;
    } else if (!quoted && /\s/.test(ch)) {
      if (has) out.push(cur);
      cur = "";
      has = false;
    } else {
      cur += ch;
      has = true;
    }
  }
  if (quoted) return null;
  if (has) out.push(cur);
  return out;
}

/** The inverse of splitCommandLine for display: arguments with spaces (or none) get double quotes. */
function joinCommandLine(argv) {
  return (Array.isArray(argv) ? argv : []).map((a) => {
    const s = String(a);
    return s === "" || /[\s"]/.test(s) ? `"${s}"` : s;
  }).join(" ");
}

const blank = (v) => typeof v !== "string" || !v.trim();

/** { ok: true, apps } (name and target trimmed) or { ok: false, error }, mirroring the core's `apps` checks. */
function validateApps(list) {
  if (!Array.isArray(list)) return { ok: false, error: "Apps must be a list." };
  if (list.length > MAX_APPS) return { ok: false, error: `Use at most ${MAX_APPS} apps.` };
  const apps = [];
  const names = new Set();
  for (let i = 0; i < list.length; i++) {
    const e = list[i];
    const at = `App ${i + 1}`;
    if (!isObj(e) || Object.keys(e).length !== 2 || !("name" in e) || !("target" in e)) return { ok: false, error: `${at} is not a complete entry.` };
    if (blank(e.name)) return { ok: false, error: `${at} needs a name.` };
    if (blank(e.target)) return { ok: false, error: `${at} needs a target (a program, file or folder).` };
    if (e.name.length > MAX_HANDS_NAME) return { ok: false, error: `${at}: the name is longer than ${MAX_HANDS_NAME} characters.` };
    if (e.target.length > MAX_TARGET) return { ok: false, error: `${at}: the target is longer than ${MAX_TARGET} characters.` };
    const name = e.name.trim();
    if (names.has(name.toLowerCase())) return { ok: false, error: `${at}: the name "${name}" is used twice. Names must be unique.` };
    names.add(name.toLowerCase());
    apps.push({ name, target: e.target.trim() });
  }
  return { ok: true, apps };
}

/** { ok: true, commands } or { ok: false, error }, mirroring the core's `commands` checks. An entry carries `argv` (a list)
 *  or `line` (typed text, split here); `cwd` blank means none; `timeout_s` may be text. */
function validateCommands(list) {
  if (!Array.isArray(list)) return { ok: false, error: "Commands must be a list." };
  if (list.length > MAX_COMMANDS) return { ok: false, error: `Use at most ${MAX_COMMANDS} commands.` };
  const commands = [];
  const names = new Set();
  const [lo, hi] = TIMEOUT_RANGE;
  for (let i = 0; i < list.length; i++) {
    const e = list[i];
    const at = `Command ${i + 1}`;
    if (!isObj(e)) return { ok: false, error: `${at} is not a complete entry.` };
    if (blank(e.name)) return { ok: false, error: `${at} needs a name.` };
    if (e.name.length > MAX_HANDS_NAME) return { ok: false, error: `${at}: the name is longer than ${MAX_HANDS_NAME} characters.` };
    let argv = e.argv;
    if (!Array.isArray(argv)) {
      if (blank(e.line)) return { ok: false, error: `${at} needs a command line.` };
      argv = splitCommandLine(e.line);
      if (argv === null) return { ok: false, error: `${at}: the command line has an unbalanced quote.` };
    }
    if (argv.length < 1) return { ok: false, error: `${at} needs a command line.` };
    if (argv.length > MAX_ARGV) return { ok: false, error: `${at}: use at most ${MAX_ARGV} words in the command line.` };
    if (argv.some(blank)) return { ok: false, error: `${at}: the command line has an empty argument.` };
    if (argv.some((a) => a.length > MAX_TARGET)) return { ok: false, error: `${at}: a word in the command line is longer than ${MAX_TARGET} characters.` };
    const cwd = typeof e.cwd === "string" ? e.cwd.trim() : "";
    if (cwd.length > MAX_TARGET) return { ok: false, error: `${at}: the folder is longer than ${MAX_TARGET} characters.` };
    const raw = e.timeout_s;
    const t = typeof raw === "string" ? (raw.trim() ? Number(raw) : NaN) : typeof raw === "number" ? raw : NaN;
    if (!Number.isFinite(t) || t < lo || t > hi) return { ok: false, error: `${at}: the timeout must be from ${lo} to ${hi} seconds.` };
    const name = e.name.trim();
    if (names.has(name.toLowerCase())) return { ok: false, error: `${at}: the name "${name}" is used twice. Names must be unique.` };
    names.add(name.toLowerCase());
    commands.push({ name, argv: [...argv], cwd: cwd || null, timeout_s: t });
  }
  return { ok: true, commands };
}

/** The reminders of a `reminders` message, with `due_at` (epoch seconds) stored as `dueAtMs` (epoch ms, compared with
 *  Date.now()); rows without an integer id and a finite due time are dropped. */
function reminderItems(msg) {
  const list = msg && Array.isArray(msg.items) ? msg.items : [];
  return list.filter((r) => isObj(r) && Number.isInteger(r.id) && typeof r.due_at === "number" && Number.isFinite(r.due_at))
    .map((r) => ({ id: r.id, kind: r.kind === "timer" ? "timer" : "reminder", text: str(r.text), dueAtMs: r.due_at * 1000 }));
}

function clock12(d) {
  const h = d.getHours();
  return `${h % 12 || 12}:${String(d.getMinutes()).padStart(2, "0")} ${h < 12 ? "am" : "pm"}`;
}

/** "in 5 min", "today 5:00 pm", "tomorrow 9:00 am" or "21 Oct 9:00 am" for `item.dueAtMs` against an epoch-ms `nowMs`
 *  (local time); "due now" once it is due, "" without a usable time. */
function reminderDue(item, nowMs) {
  const due = item && item.dueAtMs;
  if (typeof due !== "number" || !Number.isFinite(due) || !Number.isFinite(nowMs)) return "";
  if (due <= nowMs) return "due now";
  const minutes = Math.ceil((due - nowMs) / 60_000);
  if (minutes < SOON_MIN) return `in ${minutes} min`;
  const d = new Date(due);
  const now = new Date(nowMs);
  const sameDay = (a, b) => a.getFullYear() === b.getFullYear() && a.getMonth() === b.getMonth() && a.getDate() === b.getDate();
  if (sameDay(d, now)) return `today ${clock12(d)}`;
  if (sameDay(d, new Date(now.getFullYear(), now.getMonth(), now.getDate() + 1))) return `tomorrow ${clock12(d)}`;
  return `${d.getDate()} ${MONTHS[d.getMonth()]} ${clock12(d)}`;
}

/** One action-log line: the summary (capped) and how it ended, e.g. `Open "notes" · ran`. */
function actionText(a) {
  const summary = (a && typeof a.summary === "string" ? a.summary.trim() : "").slice(0, ACTION_CAP);
  const outcome = a && typeof a.outcome === "string" ? a.outcome.trim() : "";
  if (!summary) return "(no summary)";
  return outcome ? `${summary} · ${outcome}` : summary;
}

const handsExports = {
  MAX_PLACE, TIMEOUT_RANGE, TOOLS, DEFAULT_OFF, toolRows, toolsPatch, splitCommandLine, joinCommandLine, validateApps,
  validateCommands, reminderItems, reminderDue, actionText,
};
if (typeof module !== "undefined" && module.exports) module.exports = handsExports;
else globalThis.RiyaHands = handsExports;
