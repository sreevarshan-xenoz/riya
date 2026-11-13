"use strict";
/** Pure decisions used by main.js: hotkey fallback, mode fallback, late saved-position repositioning, send filtering. */
const path = require("node:path");
const { boundsFor } = require("./geometry");

const MODES = ["buddy", "panel", "window"];
const REPLAY_TYPES = ["hello", "settings", "state", "expression", "look", "eyes", "agents", "reminders", "actions", "learning"];
const DONT_LOOK_HOTKEYS = ["Control+Alt+H", "Control+Alt+F10"]; // first one that registers wins
const NOW_ONLY = ["talk_hotkey", "dont_look", "quit_core", "confirm", "actions_clear", "reminder_cancel", "note_delete",
  "note_delete_style", "twin_reset", "learning_reset", "setup_request", "setup_action", "onboarding_done", "focus_agent",
  "routine_forget"]; // never queued for replay after a reconnect
const QUIT_FLUSH_MS = 300; // the tray's Quit: time for quit_core to reach the core before the body exits

/** First accelerator for which `register` returns true (a throw counts as a failure), or null. */
function pickHotkey(accelerators, register) {
  for (const accelerator of accelerators) {
    try {
      if (register(accelerator)) return accelerator;
    } catch (e) { /* invalid or unavailable accelerator: try the next */ }
  }
  return null;
}

/** The mode to show: the requested one if valid, else the current one, else buddy, so a window always appears. */
function modeToApply(requested, current) {
  if (MODES.includes(requested)) return requested;
  return MODES.includes(current) ? current : "buddy";
}

/** New bounds for an already-shown buddy window when a saved position arrives late, or null to leave her alone. */
function repositionTarget({ mode, dragging, userDragged, moved, current, savedPos, workArea, size }) {
  if (mode !== "buddy" || dragging || userDragged || moved === true) return null; // moved: she has walked, fallen or been dropped
  if (!Array.isArray(savedPos) || savedPos.length !== 2 || !savedPos.every(Number.isFinite)) return null;
  const target = boundsFor("buddy", workArea, savedPos, size);
  return target.x === current[0] && target.y === current[1] ? null : target;
}

/** A talk, don't-look or confirm press, or quitting, is only meaningful now: never queue it for replay after a reconnect. */
function shouldSendToCore(msg, connected) {
  return connected || !NOW_ONLY.includes(msg.type);
}

/** How long Quit waits after sending quit_core: a connected core gets QUIT_FLUSH_MS to receive it. */
function quitDelay(connected) {
  return connected ? QUIT_FLUSH_MS : 0;
}

const isPos = (pos) => Array.isArray(pos) && pos.length === 2 && pos.every(Number.isFinite);
const scaleOf = (display) => (Number.isFinite(display && display.scaleFactor) && display.scaleFactor > 0 ? display.scaleFactor : 1);

/** The display whose `rectOf` rect holds the point, else the one whose rect is nearest, else null. */
function displayAt(point, displays, rectOf) {
  let best = null;
  let bestDist = Infinity;
  for (const display of Array.isArray(displays) ? displays : []) {
    const { x, y, width, height } = rectOf(display);
    const dx = Math.max(x - point[0], 0, point[0] - (x + width - 1));
    const dy = Math.max(y - point[1], 0, point[1] - (y + height - 1));
    const dist = dx * dx + dy * dy;
    if (dist < bestDist) { best = display; bestDist = dist; }
  }
  return best;
}

/** A converter over Electron's `{ bounds, scaleFactor }` displays with the shape of `screen.dipToScreenPoint` and
 *  `screen.screenToDipPoint`: DIP x scale of the display the point is on. Exact only where all displays share a scale
 *  (the main process uses the real `screen` on Windows, which is exact with mixed DPI). */
function displayConverter(displays) {
  const physical = (d) => {
    const k = scaleOf(d);
    return { x: d.bounds.x * k, y: d.bounds.y * k, width: d.bounds.width * k, height: d.bounds.height * k };
  };
  return {
    dipToScreenPoint({ x, y }) {
      const k = scaleOf(displayAt([x, y], displays, (d) => d.bounds));
      return { x: Math.round(x * k), y: Math.round(y * k) };
    },
    screenToDipPoint({ x, y }) {
      const k = scaleOf(displayAt([x, y], displays, physical));
      return { x: Math.round(x / k), y: Math.round(y / k) };
    },
  };
}

const convert = (pos, fn) => {
  if (!isPos(pos) || !fn) return pos;
  try {
    const out = fn({ x: pos[0], y: pos[1] });
    return out && Number.isFinite(out.x) && Number.isFinite(out.y) ? [Math.round(out.x), Math.round(out.y)] : pos;
  } catch (e) {
    return pos; // e.g. Electron on a platform without the conversion
  }
};

/** A buddy position in DIPs -> physical pixels (what the core compares with window rects). `conv` has Electron's
 *  `dipToScreenPoint`; anything that is not a finite [x, y] passes through. */
function toPhysical(pos, conv) {
  return convert(pos, conv && conv.dipToScreenPoint && conv.dipToScreenPoint.bind(conv));
}

/** The inverse of toPhysical, through `conv.screenToDipPoint`. */
function toDip(pos, conv) {
  return convert(pos, conv && conv.screenToDipPoint && conv.screenToDipPoint.bind(conv));
}

/** Keep the latest message of each replay type, in recency order, so a replay ends with the newest (state or
 *  expression wins by being last). */
function rememberForReplay(cache, msg) {
  if (!REPLAY_TYPES.includes(msg.type)) return;
  cache.delete(msg.type);
  cache.set(msg.type, msg);
}

/** On a core disconnect: drop what would replay as live but isn't (the core sends a fresh look and agents on hello). */
const FORGET_ON_DISCONNECT = ["look", "agents", "reminders", "actions", "learning", "confirm_request", "health"];
function forgetOnDisconnect(cache) {
  for (const type of FORGET_ON_DISCONNECT) cache.delete(type);
}

/** The one page "Get a key" opens, from the main process. The renderer asks by channel only, never with a URL. */
const KEY_URL = "https://opencode.ai/console";

/** Open the welcome once per session, and only for a core that says the user is not onboarded (an older core without
 *  the field never opens it). */
function shouldOpenWelcome(settings, alreadyShown) {
  return !alreadyShown && !!settings && settings.onboarded === false;
}

/** What to send when the welcome window closes: `onboarding_done`, unless Riya is quitting (before-quit destroyed the
 *  window, so a new user who quits mid-welcome stays not onboarded) or the user is onboarded already. Closing the
 *  welcome yourself counts as skipping. */
function welcomeClosedMessage({ quitting, onboarded } = {}) {
  return quitting === true || onboarded === true ? null : { type: "onboarding_done" };
}

/** The welcome window's own webPreferences, a fresh object each call: isolated, sandboxed, no node in the page. */
function welcomeWebPreferences(preload) {
  return {
    preload,
    additionalArguments: ["--riya-mode=welcome"],
    contextIsolation: true,
    nodeIntegration: false,
    sandbox: true,
  };
}

/** The events after which Riya is going away, each setting state.quitting: the app's "before-quit", and a window's
 *  "session-end" (a Windows shutdown or logoff really happening, where Electron emits no before-quit). So a welcome
 *  closed by the session ending sends no onboarding_done, as when quitting. Never "query-session-end": that logoff can
 *  still be cancelled, and quitting would then stay true and stop the reconnects and respawns for the session. */
const QUIT_EVENTS = { app: ["before-quit"], window: ["session-end"] };
function isQuitEvent(source, name) {
  return Object.prototype.hasOwnProperty.call(QUIT_EVENTS, source) && QUIT_EVENTS[source].includes(name);
}

const WELCOME_IPC = { "welcome:open": ["settings"], "welcome:key-page": ["welcome", "settings"] };

/** Which window kind may use a main-process channel of the welcome: open it only from Settings, the key page from
 *  either window. Any other channel or sender is refused. */
function ipcAllowed(channel, senderKind) {
  return Object.prototype.hasOwnProperty.call(WELCOME_IPC, channel) && WELCOME_IPC[channel].includes(senderKind);
}

// ---- Little Riya (v0.9): click versus drag, which agent a click brings forward, the focus helper, her menu ---------------

/** Body-to-core types only main may send: a renderer's `core:send` for one of these is dropped. */
const MAIN_ONLY = ["focus_agent"];

/** A renderer may ask for anything except a MAIN_ONLY type (and anything that is not a typed message). */
function rendererMayRequest(msg) {
  return Boolean(msg) && typeof msg === "object" && typeof msg.type === "string" && !MAIN_ONLY.includes(msg.type);
}

const CLICK_SLOP_DIP = 4; // a press that moved this far or less, for this long or less, is a click and not a drag
const CLICK_MS = 500;

function isClick({ dx, dy, ms }) {
  return Math.abs(dx) <= CLICK_SLOP_DIP && Math.abs(dy) <= CLICK_SLOP_DIP && ms <= CLICK_MS;
}

const RECENT_ACTIVITY_MS = 600000; // the latest activity names a click target for this long

/** The session a click brings forward: the first Claude session (in the core's order) that needs the user; else the
 *  session of the latest wave; else the session of the latest activity. The last two count while they are recent and the
 *  session is not listed as ended, and also while it is not listed at all: the activity reaches the body at once, the
 *  agents list only with the core's next tick, so a click on a fresh wave must not be lost. The core still checks the id
 *  (an unknown one answers hwnd null, which opens the agents list). */
function focusTarget(agents, lastActivity, nowMs, lastWave = null) {
  const items = Array.isArray(agents) ? agents.filter((a) => a !== null && typeof a === "object") : [];
  const waiting = items.find((a) => a.kind === "claude" && a.needs_attention === true);
  if (waiting) return waiting.id;
  for (const seen of [lastWave, lastActivity]) {
    if (!seen || typeof seen.session !== "string" || !(nowMs - seen.atMs <= RECENT_ACTIVITY_MS)) continue;
    const known = items.find((a) => a.id === seen.session);
    if (!known) return seen.session;
    if (known.status !== "ended") return known.id;
  }
  return null;
}

/** What a click does. With no target she is poked (nothing is sent). A target while disconnected opens the agents list:
 *  focus_agent is NOW_ONLY, so it would be dropped and never answered. */
function clickAction(target, connected) {
  if (target === null || target === undefined) return { poke: true };
  return connected ? { focus: target } : { list: true };
}

const FOCUS_WAIT_MS = 3000; // an agent_window counts as the answer to a click for this long

/** What an agent_window message does: only the answer to the pending click (same id, fresh) acts. A good hwnd is spawned
 *  for; no window (null, zero, negative, fractional, not a safe integer) opens the agents list. */
function decideFocus(pending, msg, nowMs) {
  if (!pending || !msg || msg.id !== pending.id || nowMs - pending.atMs > FOCUS_WAIT_MS) return null;
  return Number.isSafeInteger(msg.hwnd) && msg.hwnd > 0 ? { spawn: msg.hwnd } : { list: true };
}

/** The focus helper's command: the venv's python when it exists, else "python"; fixed arguments, run without a shell.
 *  Null unless hwnd is a positive safe integer. */
function focusCommand(repoRoot, hwnd, existsSync) {
  if (!Number.isSafeInteger(hwnd) || hwnd <= 0) return null;
  const venv = path.join(repoRoot, ".venv", "Scripts", "python.exe");
  return { file: existsSync(venv) ? venv : "python", args: ["-m", "riya.agents.peer", "focus", String(hwnd)] };
}

/** Her right-click menu as plain items (main adds the clicks): the display mode, don't look, reduced motion (the effective
 *  value), sleep, settings and quit. */
function petMenuItems({ mode, dontLook, reducedMotion }) {
  const radio = (id, label) => ({ id: `mode:${id}`, label, type: "radio", checked: mode === id });
  return [
    radio("buddy", "Buddy"),
    radio("panel", "Panel"),
    radio("window", "Window"),
    { id: "sep", label: "", type: "separator" },
    { id: "dont_look", label: "Don't look", type: "checkbox", checked: dontLook === true },
    { id: "reduced_motion", label: "Keep her still", type: "checkbox", checked: reducedMotion === true },
    { id: "sleep", label: "Go to sleep", type: "normal" },
    { id: "settings", label: "Settings…", type: "normal" },
    { id: "quit", label: "Quit", type: "normal" },
  ];
}

/** The setting the reduced-motion checkbox writes: `checked` is its new value, read after Electron toggled it. */
function reducedMotionPatch(checked) {
  return { reduced_motion: checked ? "on" : "off" };
}

/** What a window show or hide does to her loop: "pause" when the window hides while she runs, "resume" when it shows while
 *  she is paused, else null. A paused pet has no timers and saves nothing. */
function petVisibilityAction(paused, visible) {
  if (visible && paused) return "resume";
  if (!visible && !paused) return "pause";
  return null;
}

/** A wave that is still current is worth showing when her window shows again: it is as old as her wave pose lasts. */
const HELD_WAVE_MS = 6000;

/** What is kept of a reaction that arrives while her window is hidden: only a wave (the badge carries the rest). Anything
 *  else is dropped, so nothing stale replays when she shows again. Returns the new held wave. */
function holdWhileHidden(held, kind, label, nowMs) {
  return kind === "wave" ? { label, atMs: nowMs } : held;
}

/** On show: the held wave as a `{ kind, label }` reaction while it is still current, else null. */
function heldReactionOnShow(held, nowMs) {
  if (!held || !(nowMs - held.atMs <= HELD_WAVE_MS)) return null;
  return { kind: "wave", label: held.label };
}

/** Start the focus helper through the injected `spawn` (execFile in main) with fixed options. A failure (a non-zero exit, a
 *  spawn error or a throw) calls onFailed, but only while isCurrent() says this is still the current click's helper, so a
 *  stale one opens nothing. Returns false, spawning nothing, without a command. */
function runFocusHelper({ cmd, spawn, cwd, isCurrent, onFailed }) {
  if (!cmd) return false;
  const failed = () => { if (isCurrent()) onFailed(); };
  try {
    spawn(cmd.file, cmd.args, { cwd, windowsHide: true, timeout: 5000 }, (err) => { if (err) failed(); });
  } catch (e) {
    failed();
  }
  return true;
}

/** A walking or falling frame as a setBounds rect: integer position, her fixed size. */
function moveRect(frame, size) {
  return { x: Math.round(frame.x), y: Math.round(frame.y), width: size.width, height: size.height };
}

module.exports = {
  MAIN_ONLY, rendererMayRequest, CLICK_SLOP_DIP, CLICK_MS, isClick, RECENT_ACTIVITY_MS, focusTarget, clickAction,
  FOCUS_WAIT_MS, decideFocus, focusCommand, petMenuItems, reducedMotionPatch, moveRect, petVisibilityAction, runFocusHelper,
  HELD_WAVE_MS, holdWhileHidden, heldReactionOnShow,
  MODES, REPLAY_TYPES, NOW_ONLY, FORGET_ON_DISCONNECT, DONT_LOOK_HOTKEYS, QUIT_FLUSH_MS, pickHotkey, modeToApply, repositionTarget, shouldSendToCore,
  quitDelay, rememberForReplay, forgetOnDisconnect, toPhysical, toDip, displayConverter, KEY_URL, shouldOpenWelcome,
  welcomeClosedMessage, ipcAllowed, welcomeWebPreferences, QUIT_EVENTS, isQuitEvent,
};
