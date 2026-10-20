const test = require("node:test");
const assert = require("node:assert/strict");
const { pickHotkey, modeToApply, repositionTarget, shouldSendToCore } = require("../src/main-logic");

const WA = { x: 0, y: 0, width: 1920, height: 1040 };

test("pickHotkey takes the first accelerator that registers", () => {
  const tried = [];
  const register = (a) => { tried.push(a); return a === "B"; };
  assert.equal(pickHotkey(["A", "B", "C"], register), "B");
  assert.deepEqual(tried, ["A", "B"]);
});

test("pickHotkey uses the primary when it registers", () => {
  assert.equal(pickHotkey(["A", "B"], () => true), "A");
});

test("pickHotkey treats a throw as a failure and moves on", () => {
  const register = (a) => { if (a === "A") throw new Error("bad accelerator"); return true; };
  assert.equal(pickHotkey(["A", "B"], register), "B");
});

test("pickHotkey returns null when nothing registers", () => {
  assert.equal(pickHotkey(["A", "B"], () => false), null);
  assert.equal(pickHotkey([], () => true), null);
});

test("modeToApply keeps a valid requested mode", () => {
  assert.equal(modeToApply("panel", "buddy"), "panel");
  assert.equal(modeToApply("window", null), "window");
});

test("modeToApply falls back to the current mode, else buddy, for an invalid or missing mode", () => {
  for (const bad of [undefined, null, "blob", 3, ""]) {
    assert.equal(modeToApply(bad, null), "buddy", String(bad));
    assert.equal(modeToApply(bad, "panel"), "panel", String(bad));
  }
});

const base = { mode: "buddy", dragging: false, userDragged: false, current: [1596, 496], savedPos: [100, 200], workArea: WA };

test("repositionTarget moves a shown buddy to the late-arriving saved position", () => {
  assert.deepEqual(repositionTarget(base), { x: 100, y: 200, width: 300, height: 520 });
});

test("repositionTarget clamps an off-screen saved position", () => {
  assert.deepEqual(repositionTarget({ ...base, savedPos: [5000, -300] }), { x: 1620, y: 0, width: 300, height: 520 });
});

test("repositionTarget does nothing while dragging or after the user dragged her this session", () => {
  assert.equal(repositionTarget({ ...base, dragging: true }), null);
  assert.equal(repositionTarget({ ...base, userDragged: true }), null);
});

test("repositionTarget does nothing without a saved position, outside buddy mode, or when already there", () => {
  assert.equal(repositionTarget({ ...base, savedPos: null }), null);
  assert.equal(repositionTarget({ ...base, savedPos: [1, "x"] }), null);
  assert.equal(repositionTarget({ ...base, mode: "panel" }), null);
  assert.equal(repositionTarget({ ...base, mode: "window" }), null);
  assert.equal(repositionTarget({ ...base, current: [100, 200] }), null);
  assert.equal(repositionTarget({ ...base, savedPos: [5000, -300], current: [1620, 0] }), null);
});

test("shouldSendToCore drops talk_hotkey while disconnected but queues everything else", () => {
  assert.equal(shouldSendToCore({ type: "talk_hotkey" }, false), false);
  assert.equal(shouldSendToCore({ type: "talk_hotkey" }, true), true);
  assert.equal(shouldSendToCore({ type: "set_settings", patch: {} }, false), true);
  assert.equal(shouldSendToCore({ type: "user_text", text: "hi" }, false), true);
});

const { rememberForReplay } = require("../src/main-logic");

test("rememberForReplay keeps the latest of each replay type in recency order", () => {
  const cache = new Map();
  for (const msg of [
    { type: "hello" }, { type: "settings", n: 1 }, { type: "state", n: 1 }, { type: "expression", n: 1 },
    { type: "say", text: "x" }, { type: "state", n: 2 }, { type: "settings", n: 2 },
  ]) rememberForReplay(cache, msg);
  assert.deepEqual([...cache.values()], [
    { type: "hello" }, { type: "expression", n: 1 }, { type: "state", n: 2 }, { type: "settings", n: 2 },
  ], "the newer state replays after the older expression, so it wins");
  rememberForReplay(cache, { type: "expression", n: 2 });
  assert.deepEqual([...cache.values()].map((m) => m.type), ["hello", "state", "settings", "expression"]);
});

const {
  DONT_LOOK_HOTKEYS, REPLAY_TYPES, toPhysical, toDip, displayConverter,
} = require("../src/main-logic");

test("the don't-look hotkey falls back from Control+Alt+H to Control+Alt+F10", () => {
  assert.deepEqual(DONT_LOOK_HOTKEYS, ["Control+Alt+H", "Control+Alt+F10"]);
  assert.equal(pickHotkey(DONT_LOOK_HOTKEYS, () => true), "Control+Alt+H");
  assert.equal(pickHotkey(DONT_LOOK_HOTKEYS, (a) => a === "Control+Alt+F10"), "Control+Alt+F10");
  assert.equal(pickHotkey(DONT_LOOK_HOTKEYS, () => false), null);
});

test("a don't-look press is not queued while the core is away", () => {
  assert.equal(shouldSendToCore({ type: "dont_look", on: true }, false), false);
  assert.equal(shouldSendToCore({ type: "dont_look", on: true }, true), true);
  assert.equal(shouldSendToCore({ type: "glance_delete", id: 1 }, false), true);
});

test("look and eyes are replayed to a new window, newest last", () => {
  assert.ok(REPLAY_TYPES.includes("look") && REPLAY_TYPES.includes("eyes"));
  const cache = new Map();
  rememberForReplay(cache, { type: "eyes", dont_look: false });
  rememberForReplay(cache, { type: "look", side: "left", watching: true });
  rememberForReplay(cache, { type: "eyes", dont_look: true });
  assert.deepEqual([...cache.values()].map((m) => m.type), ["look", "eyes"]);
  assert.equal(cache.get("eyes").dont_look, true);
});

const DISPLAYS = [
  { bounds: { x: 0, y: 0, width: 1536, height: 864 }, scaleFactor: 1.25 }, // a 1920x1080 panel at 125%
  { bounds: { x: 1536, y: 0, width: 1536, height: 864 }, scaleFactor: 1.25 }, // a second panel to its right
];
const MIXED = [DISPLAYS[0], { bounds: { x: 1536, y: 0, width: 1920, height: 1080 }, scaleFactor: 1 }];

test("toPhysical multiplies a DIP position by its display's scale factor", () => {
  assert.deepEqual(toPhysical([1000, 400], displayConverter(DISPLAYS)), [1250, 500]);
  assert.deepEqual(toPhysical([1600, 100], displayConverter(DISPLAYS)), [2000, 125]);
  assert.deepEqual(toPhysical([1600, 100], displayConverter(MIXED)), [1600, 100]); // the display it is on decides the factor
  assert.deepEqual(toPhysical([1001, 401], displayConverter(DISPLAYS)), [1251, 501]); // rounded to whole pixels
});

test("toDip divides a physical position by the scale factor of the display it falls on", () => {
  assert.deepEqual(toDip([1250, 500], displayConverter(DISPLAYS)), [1000, 400]);
  assert.deepEqual(toDip([2000, 125], displayConverter(DISPLAYS)), [1600, 100]);
});

test("toPhysical and toDip round-trip on a scaled display", () => {
  for (const pos of [[0, 0], [10, 20], [1180, 464], [1600, 100]]) assert.deepEqual(toDip(toPhysical(pos, displayConverter(DISPLAYS)), displayConverter(DISPLAYS)), pos);
  const one = [{ bounds: { x: 0, y: 0, width: 2560, height: 1440 }, scaleFactor: 1 }];
  assert.deepEqual(toPhysical([123, 456], displayConverter(one)), [123, 456]);
});

test("a position off every display uses the nearest display's scale; junk passes through", () => {
  assert.deepEqual(toPhysical([-100, 50], displayConverter(DISPLAYS)), [-125, 63]);
  assert.deepEqual(toPhysical([5000, 50], displayConverter(DISPLAYS)), [6250, 63]);
  assert.deepEqual(toDip([-125, 63], displayConverter(DISPLAYS)), [-100, 50]);
  assert.equal(toPhysical(null, displayConverter(DISPLAYS)), null);
  assert.equal(toDip(undefined, displayConverter(DISPLAYS)), undefined);
  assert.deepEqual(toPhysical([1, "x"], displayConverter(DISPLAYS)), [1, "x"]);
  assert.deepEqual(toPhysical([10, 20], displayConverter([])), [10, 20]);
  assert.deepEqual(toDip([10, 20], displayConverter([{ bounds: DISPLAYS[0].bounds, scaleFactor: 0 }])), [10, 20]);
});

/** A fake Electron `screen` for a 1.25x primary (1536x864 DIP, 1920x1080 physical) with a 1.0x secondary to its right
 *  (DIP x 1536.., physical x 1920..), converting the way Windows does: per display, from its own origin. */
function fakeMixedScreen() {
  const shown = [
    { dip: { x: 0, y: 0, width: 1536, height: 864 }, phys: { x: 0, y: 0 }, k: 1.25 },
    { dip: { x: 1536, y: 0, width: 1920, height: 1080 }, phys: { x: 1920, y: 0 }, k: 1 },
  ];
  const within = (v, lo, size) => v >= lo && v < lo + size;
  return {
    dipToScreenPoint({ x, y }) {
      const d = shown.find((m) => within(x, m.dip.x, m.dip.width) && within(y, m.dip.y, m.dip.height));
      return { x: d.phys.x + (x - d.dip.x) * d.k, y: d.phys.y + (y - d.dip.y) * d.k };
    },
    screenToDipPoint({ x, y }) {
      const d = shown.find((m) => within(x, m.phys.x, m.dip.width * m.k) && within(y, m.phys.y, m.dip.height * m.k));
      return { x: d.dip.x + (x - d.phys.x) / d.k, y: d.dip.y + (y - d.phys.y) / d.k };
    },
  };
}

test("mixed DPI: a position saved from the 1.0x monitor comes back on that monitor", () => {
  const conv = fakeMixedScreen();
  assert.deepEqual(toPhysical([1600, 100], conv), [1984, 100]);
  assert.deepEqual(toDip([1984, 100], conv), [1600, 100]);
  for (const pos of [[0, 0], [1180, 464], [1535, 863], [1536, 0], [1600, 100], [3000, 900]]) {
    assert.deepEqual(toDip(toPhysical(pos, conv), conv), pos, String(pos));
  }
});

test("a conversion that throws or answers junk leaves the position as it was", () => {
  const broken = { dipToScreenPoint() { throw new Error("not implemented on this platform"); }, screenToDipPoint: () => ({ x: NaN, y: 1 }) };
  assert.deepEqual(toPhysical([5, 6], broken), [5, 6]);
  assert.deepEqual(toDip([5, 6], broken), [5, 6]);
  assert.deepEqual(toPhysical([5, 6], null), [5, 6]);
});

const { quitDelay, QUIT_FLUSH_MS } = require("../src/main-logic");

test("quit_core is only meaningful now: never queued while disconnected", () => {
  assert.equal(shouldSendToCore({ type: "quit_core" }, true), true);
  assert.equal(shouldSendToCore({ type: "quit_core" }, false), false);
});

test("quitting waits briefly for quit_core to reach a connected core, and not at all otherwise", () => {
  assert.ok(QUIT_FLUSH_MS > 0 && QUIT_FLUSH_MS <= 1000);
  assert.equal(quitDelay(true), QUIT_FLUSH_MS);
  assert.equal(quitDelay(false), 0);
});

test("agents is replayed to a new window (the settings page reads hooks_seen_at from it)", () => {
  assert.ok(REPLAY_TYPES.includes("agents"));
  const cache = new Map();
  rememberForReplay(cache, { type: "agents", items: [], hooks_seen_at: null });
  rememberForReplay(cache, { type: "agents", items: [{ id: "a" }], hooks_seen_at: 5 });
  assert.equal(cache.size, 1);
  assert.equal(cache.get("agents").hooks_seen_at, 5);
});

const { forgetOnDisconnect, FORGET_ON_DISCONNECT } = require("../src/main-logic");

test("a disconnect forgets the cached look and agents, so a reload never replays a stale agents list", () => {
  const cache = new Map();
  for (const msg of [{ type: "hello" }, { type: "settings" }, { type: "look", hwnd: 1 },
    { type: "agents", items: [{ id: "a", status: "busy" }] }, { type: "state" }]) rememberForReplay(cache, msg);
  forgetOnDisconnect(cache);
  assert.deepEqual([...cache.keys()], ["hello", "settings", "state"]);
});

const { NOW_ONLY } = require("../src/main-logic");

test("reminders and actions are replayed to a new window; a confirm answer is never queued while the core is away", () => {
  assert.ok(REPLAY_TYPES.includes("reminders") && REPLAY_TYPES.includes("actions"));
  assert.ok(!REPLAY_TYPES.includes("confirm_request"), "a confirmation expires: it is not replayed");
  assert.ok(NOW_ONLY.includes("confirm"));
  assert.equal(shouldSendToCore({ type: "confirm", id: "c", yes: true }, false), false);
  assert.equal(shouldSendToCore({ type: "confirm", id: "c", yes: true }, true), true);
  assert.equal(shouldSendToCore({ type: "reminder_cancel", id: 1 }, false), false);
  const cache = new Map();
  rememberForReplay(cache, { type: "reminders", items: [] });
  rememberForReplay(cache, { type: "reminders", items: [{ id: 1 }] });
  rememberForReplay(cache, { type: "actions", items: [] });
  assert.equal(cache.size, 2);
  assert.equal(cache.get("reminders").items.length, 1);
});

test("a disconnect forgets reminders, actions and a confirm request, so a window opened while the core is away gets none", () => {
  const cache = new Map();
  for (const msg of [{ type: "hello" }, { type: "reminders", items: [{ id: 1 }] }, { type: "actions", items: [{ id: 1 }] },
    { type: "confirm_request", id: "c" }, { type: "settings" }]) rememberForReplay(cache, msg);
  assert.deepEqual([...cache.keys()], ["hello", "reminders", "actions", "settings"], "confirm_request is never cached");
  forgetOnDisconnect(cache);
  assert.deepEqual([...cache.keys()], ["hello", "settings"]);
  assert.ok(FORGET_ON_DISCONNECT.includes("confirm_request"), "even if one were cached");
});

test("confirm, reminder_cancel, note_delete and actions_clear sent while the core is away are dropped, never replayed", () => {
  for (const type of ["confirm", "reminder_cancel", "note_delete", "actions_clear"]) {
    assert.ok(NOW_ONLY.includes(type), type);
    assert.equal(shouldSendToCore({ type, id: 1 }, false), false, type);
    assert.equal(shouldSendToCore({ type, id: 1 }, true), true, type);
  }
  assert.equal(shouldSendToCore({ type: "set_settings", patch: {} }, false), true, "settings still queue");
});

test("learning is replayed to a new window but forgotten on disconnect, and its three actions are never queued", () => {
  assert.ok(REPLAY_TYPES.includes("learning"));
  assert.ok(FORGET_ON_DISCONNECT.includes("learning"), "stale dials would pass for live");
  const cache = new Map();
  for (const msg of [{ type: "hello" }, { type: "learning", dials: {}, history: [], notes: [], twin: [], enabled: true }]) rememberForReplay(cache, msg);
  assert.deepEqual([...cache.keys()], ["hello", "learning"]);
  forgetOnDisconnect(cache);
  assert.deepEqual([...cache.keys()], ["hello"]);
  for (const type of ["note_delete_style", "twin_reset", "learning_reset"]) {
    assert.ok(NOW_ONLY.includes(type), type);
    assert.equal(shouldSendToCore({ type }, false), false, type);
    assert.equal(shouldSendToCore({ type }, true), true, type);
  }
});

test("health is never replayed and is forgotten on disconnect; setup messages are never queued", () => {
  for (const t of ["health", "voices", "setup_result", "setup_confirm"]) assert.ok(!REPLAY_TYPES.includes(t), t);
  assert.ok(FORGET_ON_DISCONNECT.includes("health"));
  const cache = new Map();
  for (const msg of [{ type: "settings" }, { type: "health", items: [] }]) rememberForReplay(cache, msg);
  assert.ok(cache.has("settings") && !cache.has("health"), "health is never cached for replay");
  cache.set("health", { type: "health", items: [] }); // even if one got in, a disconnect drops it
  forgetOnDisconnect(cache);
  assert.deepEqual([...cache.keys()], ["settings"]);
  for (const type of ["setup_request", "setup_action", "onboarding_done"]) {
    assert.ok(NOW_ONLY.includes(type), type);
    assert.equal(shouldSendToCore({ type }, false), false, type);
    assert.equal(shouldSendToCore({ type }, true), true, type);
  }
});

test("agent_activity, presence and agent_window are never replayed; focus_agent is never queued", () => {
  for (const t of ["agent_activity", "presence", "agent_window"]) assert.ok(!REPLAY_TYPES.includes(t), t);
  assert.ok(NOW_ONLY.includes("focus_agent"));
  assert.equal(shouldSendToCore({ type: "focus_agent", id: "a" }, false), false);
  assert.equal(shouldSendToCore({ type: "focus_agent", id: "a" }, true), true);
});

const M = require("../src/main-logic");
const readBody = (f) => require("node:fs").readFileSync(require("node:path").join(__dirname, "..", f), "utf8");

test("shouldOpenWelcome only for onboarded false, once", () => {
  assert.equal(M.shouldOpenWelcome({ onboarded: false }, false), true);
  assert.equal(M.shouldOpenWelcome({ onboarded: false }, true), false);
  for (const s of [{ onboarded: true }, {}, null, undefined, { onboarded: "false" }, { onboarded: 0 }]) {
    assert.equal(M.shouldOpenWelcome(s, false), false, JSON.stringify(s));
  }
});

test("KEY_URL is the one fixed page the main process opens", () => {
  assert.equal(M.KEY_URL, "https://aistudio.google.com/apikey");
});

test("welcomeClosedMessage: closing counts as skipping, quitting sends nothing, onboarded sends nothing", () => {
  const Ajv = require("ajv");
  const schema = JSON.parse(readBody("../docs/protocol.schema.json"));
  const validate = new Ajv({ strict: false }).compile(schema.body_to_core);
  for (const arg of [{ quitting: false, onboarded: false }, { quitting: false }, { onboarded: false }, {}]) {
    const m = M.welcomeClosedMessage(arg);
    assert.deepEqual(m, { type: "onboarding_done" }, JSON.stringify(arg));
    assert.ok(validate(m));
  }
  assert.equal(M.welcomeClosedMessage({ quitting: true, onboarded: false }), null);
  assert.equal(M.welcomeClosedMessage({ quitting: true }), null);
  assert.equal(M.welcomeClosedMessage({ quitting: false, onboarded: true }), null);
});

test("ipcAllowed holds the welcome's IPC rules", () => {
  assert.equal(M.ipcAllowed("welcome:open", "settings"), true);
  assert.equal(M.ipcAllowed("welcome:open", "welcome"), false);
  assert.equal(M.ipcAllowed("welcome:open", "other"), false);
  assert.equal(M.ipcAllowed("welcome:key-page", "welcome"), true);
  assert.equal(M.ipcAllowed("welcome:key-page", "settings"), true);
  assert.equal(M.ipcAllowed("welcome:key-page", "other"), false);
  assert.equal(M.ipcAllowed("core:send", "settings"), false);
  assert.equal(M.ipcAllowed("welcome:open", undefined), false);
  assert.equal(M.ipcAllowed("toString", "settings"), false);
});

test("the quit-mid-welcome and close-once rules, as the closed handler applies them", () => {
  // The handler is `const m = welcomeClosedMessage({quitting, onboarded}); if (m) sendToCore(m);` on each close.
  const sent = [];
  const closed = (state) => {
    const m = M.welcomeClosedMessage({ quitting: state.quitting, onboarded: state.settings.onboarded });
    if (m) sent.push(m);
  };
  closed({ quitting: true, settings: { onboarded: false } }); // Riya quitting mid-welcome: before-quit destroyed the window
  assert.deepEqual(sent, []);
  const state = { quitting: false, settings: { onboarded: false } };
  closed(state); // Finish, Skip or the window's own X
  assert.deepEqual(sent, [{ type: "onboarding_done" }]);
  state.settings.onboarded = true; // the core's settings message answers
  closed(state); // reopened from Settings and closed again: nothing
  assert.equal(sent.length, 1);
});

test("main.js wires the welcome through the pure rules, and the renderer can never pass a URL", () => {
  const main = readBody("main.js");
  const preload = readBody("preload.js");
  assert.match(main, /welcomeClosedMessage\(\{ quitting: state\.quitting, onboarded: state\.settings\.onboarded \}\)/);
  assert.match(main, /ipcAllowed\("welcome:open", senderKind\(event\.sender\)\)/);
  assert.match(main, /ipcAllowed\("welcome:key-page", senderKind\(event\.sender\)\)/);
  assert.match(main, /shouldOpenWelcome\(msg\.settings, state\.welcomeShown\)/);
  assert.equal((main.match(/openExternal\(/g) || []).length, 1, "the one fixed link is the only openExternal");
  assert.match(main, /shell\.openExternal\(KEY_URL\)/);
  assert.doesNotMatch(main, /onboarding_done/, "main.js reaches onboarding_done only through welcomeClosedMessage");
  assert.deepEqual(M.welcomeWebPreferences("p").additionalArguments, ["--riya-mode=welcome"]);
  assert.match(preload, /openWelcome: \(\) => ipcRenderer\.send\("welcome:open"\)/);
  assert.match(preload, /openKeyPage: \(\) => ipcRenderer\.send\("welcome:key-page"\)/);
});

/** main.js from `function name(` up to the next top-level function: assertions about one window stay inside it. */
const fnSlice = (src, name) => {
  const start = src.indexOf(`function ${name}(`);
  assert.ok(start >= 0, `main.js has ${name}`);
  const next = src.indexOf("\nfunction ", start + 1);
  return src.slice(start, next === -1 ? undefined : next);
};

test("the welcome window's own webPreferences are isolated, sandboxed and without node", () => {
  const prefs = M.welcomeWebPreferences("C:/x/preload.js");
  assert.deepEqual(prefs, {
    preload: "C:/x/preload.js",
    additionalArguments: ["--riya-mode=welcome"],
    contextIsolation: true,
    nodeIntegration: false,
    sandbox: true,
  });
  assert.notEqual(M.welcomeWebPreferences("a"), M.welcomeWebPreferences("a"), "a fresh object each time");
  const welcome = fnSlice(readBody("main.js"), "openWelcome");
  assert.match(welcome, /webPreferences: welcomeWebPreferences\(path\.join\(__dirname, "preload\.js"\)\)/);
  assert.doesNotMatch(welcome, /sandbox: false|contextIsolation: false|nodeIntegration: true/);
});

test("a Windows shutdown or logoff counts as quitting, so a mid-welcome user is not marked onboarded", () => {
  assert.deepEqual(M.QUIT_EVENTS, { app: ["before-quit"], window: ["session-end"] });
  for (const [source, name] of [["app", "before-quit"], ["window", "session-end"]]) {
    assert.equal(M.isQuitEvent(source, name), true, `${source} ${name}`);
  }
  for (const [source, name] of [["window", "query-session-end"], ["window", "close"], ["window", "closed"],
    ["app", "session-end"], ["x", "before-quit"]]) {
    assert.equal(M.isQuitEvent(source, name), false, `${source} ${name}`);
  }
  // a welcome closed by the session ending sends nothing, like quitting
  assert.equal(M.welcomeClosedMessage({ quitting: true, onboarded: false }), null);
  const main = readBody("main.js");
  assert.doesNotMatch(main, /query-session-end/, "only the real session end counts");
  const welcome = fnSlice(main, "openWelcome");
  assert.match(welcome, /for \(const event of QUIT_EVENTS\.window\) win\.on\(event, \(\) => \{ state\.quitting = true; \}\);/);
  assert.ok(welcome.indexOf("QUIT_EVENTS.window") < welcome.indexOf('win.on("closed"'), "wired before the closed handler");
  assert.match(main, /for \(const event of QUIT_EVENTS\.app\) app\.on\(event, \(\) => \{ state\.quitting = true; \}\);/);
});

test("a logoff query that is then cancelled leaves quitting false: Riya keeps reconnecting and respawning", () => {
  const state = { quitting: false };
  const fire = (source, name) => { if (M.isQuitEvent(source, name)) state.quitting = true; };
  fire("window", "query-session-end"); // Windows asks; another app blocks it, or the user cancels
  assert.equal(state.quitting, false);
  assert.notEqual(M.welcomeClosedMessage({ quitting: state.quitting, onboarded: false }), null, "closing the welcome still counts as skipping");
  fire("window", "session-end"); // the session really ends
  assert.equal(state.quitting, true);
  assert.equal(M.welcomeClosedMessage({ quitting: state.quitting, onboarded: false }), null);
});

// ---- v0.9 Little Riya: main-process wiring -------------------------------------------------------------------------

test("isClick: a short press that barely moved", () => {
  assert.equal(M.CLICK_SLOP_DIP, 4);
  assert.equal(M.CLICK_MS, 500);
  assert.equal(M.isClick({ dx: 3, dy: -4, ms: 499 }), true);
  assert.equal(M.isClick({ dx: 4, dy: 4, ms: 500 }), true);
  assert.equal(M.isClick({ dx: 5, dy: 0, ms: 100 }), false);
  assert.equal(M.isClick({ dx: 0, dy: 0, ms: 501 }), false);
  assert.equal(M.isClick({ dx: 0, dy: -5, ms: 100 }), false);
});

test("focusTarget prefers the session that needs you", () => {
  const agents = [
    { id: "a", kind: "claude", needs_attention: false, status: "busy" },
    { id: "b", kind: "claude", needs_attention: true, status: "waiting" },
  ];
  assert.equal(M.RECENT_ACTIVITY_MS, 600000);
  assert.equal(M.focusTarget(agents, null, 0), "b");
  assert.equal(M.focusTarget(agents, { session: "a", atMs: 0 }, 1000), "b", "attention beats recent activity");
  const colab = [{ id: "c", kind: "colab", needs_attention: true, status: "busy" }, agents[0]];
  assert.equal(M.focusTarget(colab, null, 0), null, "a colab item needing attention is skipped");
  assert.equal(M.focusTarget(agents.slice(0, 1), { session: "a", atMs: 0 }, 600000), "a");
  assert.equal(M.focusTarget(agents.slice(0, 1), { session: "a", atMs: 0 }, 601000), null, "601 s old");
  assert.equal(M.focusTarget([{ ...agents[0], status: "ended" }], { session: "a", atMs: 0 }, 1000), null);
  assert.equal(M.focusTarget(agents.slice(0, 1), { session: "zz", atMs: 0 }, 1000), null, "not in the list");
  assert.equal(M.focusTarget([], { session: "a", atMs: 0 }, 1000), null);
  assert.equal(M.focusTarget(undefined, undefined, 0), null);
  assert.equal(M.focusTarget([null, 3, { id: "a", kind: "claude", needs_attention: true }], null, 0), "a");
});

test("a focus only follows its own fresh click", () => {
  const msg = (id, hwnd) => ({ type: "agent_window", id, hwnd });
  assert.equal(M.FOCUS_WAIT_MS, 3000);
  assert.equal(M.decideFocus(null, msg("a", 77), 0), null);
  const pending = { id: "a", atMs: 1000 };
  assert.deepEqual(M.decideFocus(pending, msg("a", 77), 3000), { spawn: 77 });
  assert.deepEqual(M.decideFocus(pending, msg("a", 77), 4000), { spawn: 77 });
  assert.equal(M.decideFocus(pending, msg("a", 77), 4001), null);
  assert.equal(M.decideFocus(pending, msg("b", 77), 3000), null);
  for (const hwnd of [null, 0, -5, 1.5, "77", undefined, NaN, 2 ** 60]) {
    assert.deepEqual(M.decideFocus(pending, msg("a", hwnd), 3000), { list: true }, String(hwnd));
  }
});

test("a click with no answer opens the agents list", () => {
  assert.deepEqual(M.clickAction(null, true), { poke: true });
  assert.deepEqual(M.clickAction(null, false), { poke: true });
  assert.deepEqual(M.clickAction("a", false), { list: true });
  assert.deepEqual(M.clickAction("a", true), { focus: "a" });
  const main = readBody("main.js");
  assert.match(fnSlice(main, "petClick"), /setTimeout\(focusTimedOut, FOCUS_WAIT_MS\)/);
  assert.match(fnSlice(main, "focusTimedOut"), /openAgentsList\(/);
  const branch = main.slice(main.indexOf('msg.type === "agent_window"'));
  assert.match(branch.slice(0, 900), /decideFocus\([\s\S]*clearTimeout\(/);
});

test("renderers can never send focus_agent", () => {
  assert.deepEqual(M.MAIN_ONLY, ["focus_agent"]);
  assert.equal(M.rendererMayRequest({ type: "focus_agent", id: "a" }), false);
  assert.equal(M.rendererMayRequest({ type: "user_text", text: "hi" }), true);
  assert.equal(M.rendererMayRequest(null), false);
  assert.equal(M.rendererMayRequest({}), false);
});

test("the focus helper command is fixed", () => {
  const path = require("node:path");
  const venv = path.join("E:\\riya", ".venv", "Scripts", "python.exe");
  assert.deepEqual(M.focusCommand("E:\\riya", 77, () => true), {
    file: venv, args: ["-m", "riya.agents.peer", "focus", "77"],
  });
  assert.deepEqual(M.focusCommand("E:\\riya", 77, (p) => p !== venv), {
    file: "python", args: ["-m", "riya.agents.peer", "focus", "77"],
  });
  for (const bad of ["77; calc", 0, -1, NaN, 1.5, null, undefined, Infinity]) {
    assert.equal(M.focusCommand("E:\\riya", bad, () => true), null, String(bad));
  }
});

test("petMenuItems and the reduced-motion toggle", () => {
  const items = M.petMenuItems({ mode: "buddy", dontLook: false, reducedMotion: false });
  assert.deepEqual(items.map((i) => i.id),
    ["mode:buddy", "mode:panel", "mode:window", "sep", "dont_look", "reduced_motion", "sleep", "settings", "quit"]);
  assert.deepEqual(items.slice(0, 3).map((i) => [i.type, i.checked]), [["radio", true], ["radio", false], ["radio", false]]);
  assert.equal(items.find((i) => i.id === "sep").type, "separator");
  assert.equal(items.find((i) => i.id === "dont_look").type, "checkbox");
  assert.equal(items.find((i) => i.id === "reduced_motion").type, "checkbox");
  assert.equal(items.find((i) => i.id === "sleep").label, "Go to sleep");
  assert.equal(items.find((i) => i.id === "settings").label, "Settings\u2026");
  const panel = M.petMenuItems({ mode: "panel", dontLook: true, reducedMotion: false });
  assert.equal(panel.filter((i) => i.type === "radio").map((i) => i.checked).join(), "false,true,false");
  assert.equal(panel.find((i) => i.id === "dont_look").checked, true);
  assert.equal(M.petMenuItems({ mode: "buddy", dontLook: false, reducedMotion: true }).find((i) => i.id === "reduced_motion").checked, true);
  assert.deepEqual(M.reducedMotionPatch(true), { reduced_motion: "on" });
  assert.deepEqual(M.reducedMotionPatch(false), { reduced_motion: "off" });
  assert.match(readBody("main.js"), /reducedMotionPatch\(\s*menuItem\.checked\s*\)/);
});

test("moveRect is integer with a fixed size", () => {
  assert.deepEqual(M.moveRect({ x: 10.6, y: 695.5 }, { width: 240, height: 344 }), { x: 11, y: 696, width: 240, height: 344 });
});

test("repositionTarget takes the pet size", () => {
  assert.deepEqual(M.repositionTarget({ ...base, size: { buddy: [240, 344] } }), { x: 100, y: 200, width: 240, height: 344 });
  assert.deepEqual(M.repositionTarget({ ...base, savedPos: [5000, 5000], size: { buddy: [240, 344] } }),
    { x: 1680, y: 696, width: 240, height: 344 });
});

test("DEFAULT_SIZE is exported from geometry", () => {
  assert.deepEqual(require("../src/geometry").DEFAULT_SIZE, { buddy: [300, 520] });
});

test("presence comes from both state and presence messages and is forgotten on disconnect", () => {
  const main = readBody("main.js");
  assert.match(main, /msg\.type === "presence"[\s\S]{0,200}state\.presence = /);
  assert.match(main, /msg\.type === "state"[\s\S]{0,300}state\.presence/);
  assert.match(fnSlice(main, "onCoreStatus"), /if \(!connected\)[\s\S]{0,300}state\.presence = null/);
});

test("the pet steps at once on every feed and reads idle fresh", () => {
  const main = readBody("main.js");
  assert.match(main, /function stepPet\(\)[\s\S]{0,400}getSystemIdleTime\(\)/);
  assert.match(main, /setTimeout\(stepPet, Math\.max\(0, /);
  const step = fnSlice(main, "stepPet");
  const outside = main.replace(step, "");
  assert.equal((outside.match(/schedulePet\(/g) || []).length, 1, "only its own definition remains outside stepPet");
  assert.match(step, /schedulePet\(\)/);
  for (const call of ["pet.react(", "pet.setContext(", "pet.setArea(", "pet.setSize(", "pet.poke(", "pet.sleepNow(", "pet.grab(", "pet.drop("]) {
    let from = 0;
    let seen = 0;
    for (;;) {
      const at = outside.indexOf(call, from);
      if (at < 0) break;
      seen += 1;
      assert.match(outside.slice(at, at + 400), /stepPet\(\)/, `${call} at ${at} is followed by stepPet()`);
      from = at + 1;
    }
    assert.ok(seen > 0, `${call} is used`);
  }
});

test("a new window gets the current view, and stopping sends the whole OFF_VIEW", () => {
  const main = readBody("main.js");
  assert.match(fnSlice(main, "createStageWindow"), /did-finish-load[\s\S]{0,500}"pet:view", state\.lastPetView/);
  const stop = fnSlice(main, "stopPet");
  assert.match(stop, /state\.lastPetView = OFF_VIEW/);
  assert.match(stop, /"pet:view", OFF_VIEW/);
  assert.doesNotMatch(main, /[{,]\s*on: false\b/, "never a partial view");
});

test("main spawns the helper only through decideFocus", () => {
  const main = readBody("main.js");
  assert.equal((main.match(/\bexecFile\b/g) || []).length, 2, "the require and the one use");
  assert.equal((main.match(/runFocusHelper\(/g) || []).length, 1, "the helper starts in one place");
  const branch = main.slice(main.indexOf('msg.type === "agent_window"'));
  assert.ok(branch.indexOf("decideFocus(") >= 0 && branch.indexOf("decideFocus(") < branch.indexOf("runFocusHelper("));
  assert.match(branch, /runFocusHelper\(\{[^}]*spawn: execFile/);
  assert.match(main, /focusCommand\(REPO_ROOT, /);
  assert.equal((main.match(/"focus_agent"/g) || []).length, 1, "focus_agent is named once");
  assert.ok(fnSlice(main, "petClick").includes('"focus_agent"'), "and only in petClick");
  assert.match(main, /ipcMain\.on\("core:send"[\s\S]{0,400}rendererMayRequest\(msg\)/);
  assert.match(main, /getDisplayMatching\(/);
  assert.equal((main.match(/setInterval\(stepPet/g) || []).length, 1);
  assert.ok(fnSlice(main, "schedulePet").includes("setInterval(stepPet"));
  const end = main.slice(main.indexOf('ipcMain.on("drag-end"'));
  const handler = end.slice(0, end.indexOf("\n});"));
  const area = handler.indexOf("pet.setArea(");
  const drop = handler.indexOf("pet.drop(");
  assert.ok(area >= 0 && drop >= 0 && area < drop, "the monitor is re-read before the first drop");
  assert.ok(handler.indexOf("getDisplayMatching(") >= 0 && handler.indexOf("getDisplayMatching(") < drop);
  const click = handler.slice(handler.indexOf("isClick("));
  assert.ok(click.indexOf("pet.drop(") >= 0 && click.indexOf("pet.drop(") < click.indexOf("petClick("));
  assert.match(click.slice(click.indexOf("pet.drop("), click.indexOf("petClick(")), /save: false/);
});

test("the windows stay sandboxed, and the preload gains the pet's narrow surface", () => {
  const main = readBody("main.js");
  assert.ok((main.match(/sandbox: true/g) || []).length >= 2);
  assert.doesNotMatch(main, /sandbox: false|contextIsolation: false|nodeIntegration: true/);
  const preload = readBody("preload.js");
  assert.match(preload, /onPetView: \(cb\) => ipcRenderer\.on\("pet:view"/);
  assert.match(preload, /reportReducedMotion: \(on\) => ipcRenderer\.send\("pet:reduced-motion", Boolean\(on\)\)/);
  assert.match(preload, /petMenu: \(\) => ipcRenderer\.send\("pet:menu"\)/);
  assert.match(preload, /onShowSection: \(cb\) => ipcRenderer\.on\("settings:show"/);
});

// ---- round 1 fixes -------------------------------------------------------------------------------------------------

test("the pet pauses while its window is hidden and resumes when it shows", () => {
  assert.equal(M.petVisibilityAction(false, false), "pause");
  assert.equal(M.petVisibilityAction(true, false), null, "already paused");
  assert.equal(M.petVisibilityAction(true, true), "resume");
  assert.equal(M.petVisibilityAction(false, true), null, "already running");
  const main = readBody("main.js");
  assert.match(main, /function stepPet\(\)[\s\S]{0,120}state\.petPaused/);
  assert.match(fnSlice(main, "setPetVisible"), /petVisibilityAction\(/);
  assert.match(fnSlice(main, "createStageWindow"), /win\.on\("hide", \(\) => setPetVisible\(win, false\)\)/);
  assert.match(fnSlice(main, "createStageWindow"), /win\.on\("show", \(\) => setPetVisible\(win, true\)\)/);
});

test("a late settings message leaves her alone once she has moved this session", () => {
  const moved = { ...base, moved: true };
  assert.equal(M.repositionTarget(moved), null);
  assert.deepEqual(M.repositionTarget({ ...base, moved: false }), { x: 100, y: 200, width: 300, height: 520 });
  assert.deepEqual(M.repositionTarget(base), { x: 100, y: 200, width: 300, height: 520 }, "the flag is optional");
  const main = readBody("main.js");
  assert.match(fnSlice(main, "applyLateSavedPosition"), /moved: state\.petMoved/);
  assert.match(fnSlice(main, "stepPet"), /f\.moved[\s\S]{0,200}state\.petMoved = true/);
  const end = main.slice(main.indexOf('ipcMain.on("drag-end"'));
  assert.match(end.slice(0, end.indexOf("\n});")), /state\.petMoved = true/, "a drop counts too");
});

test("a failed focus helper opens the agents list, only for the current click", () => {
  const cmd = { file: "python", args: ["-m", "riya.agents.peer", "focus", "77"] };
  const run = (outcome, current = () => true) => {
    const calls = [];
    const opened = [];
    const spawn = (file, args, opts, cb) => {
      calls.push({ file, args, opts });
      if (outcome === "throw") throw new Error("EINVAL");
      cb(outcome);
    };
    const started = M.runFocusHelper({ cmd, spawn, cwd: "E:\riya", isCurrent: current, onFailed: () => opened.push(1) });
    return { calls, opened, started };
  };
  const ok = run(null);
  assert.equal(ok.started, true);
  assert.deepEqual(ok.calls, [{ file: "python", args: cmd.args, opts: { cwd: "E:\riya", windowsHide: true, timeout: 5000 } }]);
  assert.equal(ok.opened.length, 0, "success: nothing more");
  assert.equal(run(Object.assign(new Error("exit 1"), { code: 1 })).opened.length, 1, "non-zero exit");
  assert.equal(run(Object.assign(new Error("spawn python ENOENT"), { code: "ENOENT" })).opened.length, 1, "spawn error");
  assert.equal(run("throw").opened.length, 1, "a synchronous throw");
  assert.equal(run(new Error("exit 1"), () => false).opened.length, 0, "a stale click opens nothing");
  assert.equal(run("throw", () => false).opened.length, 0);
  const none = M.runFocusHelper({ cmd: null, spawn: () => assert.fail("never spawns"), cwd: "x", isCurrent: () => true, onFailed() {} });
  assert.equal(none, false);
  const main = readBody("main.js");
  assert.match(fnSlice(main, "petClick"), /state\.focusToken \+= 1/);
  assert.match(fnSlice(main, "stopPet"), /state\.focusToken \+= 1/);
  assert.match(main, /isCurrent: \(\) => token === state\.focusToken/);
});
