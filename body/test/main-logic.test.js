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
  DONT_LOOK_HOTKEYS, REPLAY_TYPES, toPhysical, toDip,
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
  assert.deepEqual(toPhysical([1000, 400], DISPLAYS), [1250, 500]);
  assert.deepEqual(toPhysical([1600, 100], DISPLAYS), [2000, 125]);
  assert.deepEqual(toPhysical([1600, 100], MIXED), [1600, 100]); // the display it is on decides the factor
  assert.deepEqual(toPhysical([1001, 401], DISPLAYS), [1251, 501]); // rounded to whole pixels
});

test("toDip divides a physical position by the scale factor of the display it falls on", () => {
  assert.deepEqual(toDip([1250, 500], DISPLAYS), [1000, 400]);
  assert.deepEqual(toDip([2000, 125], DISPLAYS), [1600, 100]);
});

test("toPhysical and toDip round-trip on a scaled display", () => {
  for (const pos of [[0, 0], [10, 20], [1180, 464], [1600, 100]]) assert.deepEqual(toDip(toPhysical(pos, DISPLAYS), DISPLAYS), pos);
  const one = [{ bounds: { x: 0, y: 0, width: 2560, height: 1440 }, scaleFactor: 1 }];
  assert.deepEqual(toPhysical([123, 456], one), [123, 456]);
});

test("a position off every display uses the nearest display's scale; junk passes through", () => {
  assert.deepEqual(toPhysical([-100, 50], DISPLAYS), [-125, 63]);
  assert.deepEqual(toPhysical([5000, 50], DISPLAYS), [6250, 63]);
  assert.deepEqual(toDip([-125, 63], DISPLAYS), [-100, 50]);
  assert.equal(toPhysical(null, DISPLAYS), null);
  assert.equal(toDip(undefined, DISPLAYS), undefined);
  assert.deepEqual(toPhysical([1, "x"], DISPLAYS), [1, "x"]);
  assert.deepEqual(toPhysical([10, 20], []), [10, 20]);
  assert.deepEqual(toDip([10, 20], [{ bounds: DISPLAYS[0].bounds, scaleFactor: 0 }]), [10, 20]);
});
