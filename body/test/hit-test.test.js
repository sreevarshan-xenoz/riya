"use strict";
const test = require("node:test");
const assert = require("node:assert/strict");
const { buddyClickable, isOpaqueAt, toImageCoords } = require("../src/hit-test");

const none = { onBubble: false, onConfirm: false, confirmShown: false, overSprite: false };

test("buddyClickable: her pixels and her bubble take the mouse, the empty window does not", () => {
  assert.equal(buddyClickable(none), false);
  assert.equal(buddyClickable({ ...none, overSprite: true }), true);
  assert.equal(buddyClickable({ ...none, onBubble: true }), true);
});

test("buddyClickable: the confirm buttons take the mouse only while the confirm is shown", () => {
  assert.equal(buddyClickable({ ...none, onConfirm: true, confirmShown: true }), true);
  assert.equal(buddyClickable({ ...none, onConfirm: true, confirmShown: false }), false);
  assert.equal(buddyClickable({ ...none, confirmShown: true }), false, "a shown confirm alone doesn't make the window solid");
  assert.equal(buddyClickable({ ...none, onConfirm: true, confirmShown: false, overSprite: true }), true);
});

test("buddyClickable returns a real boolean and survives missing fields", () => {
  assert.equal(buddyClickable({}), false);
  assert.equal(buddyClickable(), false);
  assert.strictEqual(buddyClickable({ onConfirm: 1, confirmShown: 1 }), true);
});

test("isOpaqueAt and toImageCoords still work", () => {
  const rgba = new Uint8Array(4 * 4);
  rgba[3 * 4 + 3] = 255;
  assert.equal(isOpaqueAt(rgba, 4, 3, 0), true);
  assert.equal(isOpaqueAt(rgba, 4, 0, 0), false);
  assert.deepEqual(toImageCoords({ left: 10, top: 10, width: 100, height: 50 }, 10, 5, 60, 35), { x: 5, y: 2.5 });
  assert.equal(toImageCoords({ left: 0, top: 0, width: 0, height: 5 }, 1, 1, 0, 0), null);
});
