const test = require("node:test");
const assert = require("node:assert/strict");
const { boundsFor, clampToArea } = require("../src/geometry");
const { isOpaqueAt } = require("../src/hit-test");

const WA = { x: 0, y: 0, width: 1920, height: 1040 };

test("buddy default is bottom-right with margin", () => {
  assert.deepEqual(boundsFor("buddy", WA, null), { x: 1596, y: 496, width: 300, height: 520 });
});

test("buddy saved position off-screen is clamped", () => {
  assert.deepEqual(boundsFor("buddy", WA, [5000, -300]), { x: 1620, y: 0, width: 300, height: 520 });
});

test("buddy saved position inside the area is kept", () => {
  assert.deepEqual(boundsFor("buddy", WA, [100, 200]), { x: 100, y: 200, width: 300, height: 520 });
});

test("buddy ignores a malformed saved position", () => {
  const dflt = { x: 1596, y: 496, width: 300, height: 520 };
  for (const bad of [undefined, [1], ["a", 2], [NaN, 3], "x", [1, 2, 3]]) assert.deepEqual(boundsFor("buddy", WA, bad), dflt);
});

test("buddy respects a work area offset (taskbar on top or a second monitor)", () => {
  const wa = { x: 1920, y: 40, width: 1280, height: 1000 };
  assert.deepEqual(boundsFor("buddy", wa, null), { x: 2876, y: 496, width: 300, height: 520 });
  assert.deepEqual(boundsFor("buddy", wa, [0, 0]), { x: 1920, y: 40, width: 300, height: 520 });
});

test("buddy honours a custom size", () => {
  assert.deepEqual(boundsFor("buddy", WA, null, { buddy: [200, 400] }), { x: 1696, y: 616, width: 200, height: 400 });
});

test("panel docks right", () => {
  assert.deepEqual(boundsFor("panel", WA, null), { x: 1600, y: 0, width: 320, height: 1040 });
});

test("window is centered", () => {
  assert.deepEqual(boundsFor("window", WA, null), { x: 480, y: 200, width: 960, height: 640 });
});

test("window never exceeds a small work area", () => {
  const wa = { x: 0, y: 0, width: 800, height: 600 };
  assert.deepEqual(boundsFor("window", wa, null), { x: 0, y: 0, width: 800, height: 600 });
});

test("unknown mode throws", () => {
  assert.throws(() => boundsFor("blob", WA, null), /unknown mode/);
});

test("clampToArea pins a rect inside and keeps its size", () => {
  assert.deepEqual(clampToArea({ x: -50, y: 2000, width: 300, height: 520 }, WA), { x: 0, y: 520, width: 300, height: 520 });
});

test("clampToArea pins a rect larger than the area to the origin", () => {
  assert.deepEqual(clampToArea({ x: 50, y: 50, width: 3000, height: 2000 }, WA), { x: 0, y: 0, width: 3000, height: 2000 });
});

test("alpha hit test", () => {
  const px = new Uint8ClampedArray([0, 0, 0, 0, 255, 255, 255, 255]);
  assert.equal(isOpaqueAt(px, 2, 0, 0), false);
  assert.equal(isOpaqueAt(px, 2, 1, 0), true);
});

test("alpha hit test threshold and bounds", () => {
  const px = new Uint8ClampedArray([0, 0, 0, 15, 0, 0, 0, 16, 0, 0, 0, 255, 0, 0, 0, 0]);
  assert.equal(isOpaqueAt(px, 2, 0, 0), false);
  assert.equal(isOpaqueAt(px, 2, 1, 0), true);
  assert.equal(isOpaqueAt(px, 2, 0, 0, 10), true);
  assert.equal(isOpaqueAt(px, 2, 0, 1), true);
  assert.equal(isOpaqueAt(px, 2, 1, 1), false);
  assert.equal(isOpaqueAt(px, 2, -1, 0), false);
  assert.equal(isOpaqueAt(px, 2, 2, 0), false);
  assert.equal(isOpaqueAt(px, 2, 0, 2), false);
  assert.equal(isOpaqueAt(px, 2, 0.9, 1.9), true);
  assert.equal(isOpaqueAt(px, 2, NaN, 0), false);
});
