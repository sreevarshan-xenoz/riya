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
