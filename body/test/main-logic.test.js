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
