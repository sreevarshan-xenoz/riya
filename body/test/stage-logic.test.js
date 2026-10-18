"use strict";
const test = require("node:test");
const assert = require("node:assert/strict");
const { ExpressionModel, MouthTracker, StatusLine, cropBox, CURIOUS_BID } = require("../src/stage-logic");
const { toImageCoords } = require("../src/hit-test");

test("expression comes from expression and state messages", () => {
  const m = new ExpressionModel();
  assert.equal(m.current(false), "neutral");
  m.onState({ expression: "happy" });
  assert.equal(m.current(false), "happy");
  m.onExpression("pouty");
  assert.equal(m.current(false), "pouty");
});

test("a disconnected body shows neutral and recovers", () => {
  const m = new ExpressionModel();
  m.onExpression("happy");
  m.setConnected(false);
  assert.equal(m.current(false), "neutral");
  m.setConnected(true);
  assert.equal(m.current(false), "happy");
});

test("a visible bid shows curious unless the core sends an expression meanwhile", () => {
  const m = new ExpressionModel();
  m.onState({ expression: "sleepy" });
  m.onBid();
  assert.equal(CURIOUS_BID, "curious");
  assert.equal(m.current(true), "curious");
  m.onState({ expression: "sleepy" }); // a state push does not cancel the override
  assert.equal(m.current(true), "curious");
  assert.equal(m.current(false), "sleepy", "once the bubble is gone she goes back");
  assert.equal(m.current(true), "sleepy", "and the override is spent");

  m.onBid();
  m.onExpression("worried");
  assert.equal(m.current(true), "worried");
});

test("mouth opens and closes with hysteresis and closes when messages stop", () => {
  const t = new MouthTracker();
  assert.equal(t.isOpen(0), false);
  t.onLevel(0.5, 100);
  assert.equal(t.isOpen(110), true);
  t.onLevel(0.2, 133); // between the thresholds: stays open
  assert.equal(t.isOpen(140), true);
  t.onLevel(0.05, 166);
  assert.equal(t.isOpen(170), false);
  t.onLevel(0.9, 200);
  assert.equal(t.isOpen(200 + 249), true);
  assert.equal(t.isOpen(200 + 250), false, "no mouth message for 250 ms means she stopped");
  t.onLevel(0.2, 1000); // gate was reset by the silence: 0.2 is below the open threshold
  assert.equal(t.isOpen(1000), false);
});

test("status line shows an error briefly", () => {
  const s = new StatusLine({ ttlMs: 5000 });
  assert.equal(s.text(0), "");
  s.show("model is down", 1000);
  assert.equal(s.text(5999), "model is down");
  assert.equal(s.text(6000), "");
  s.show("again", 7000);
  s.show("newer", 8000);
  assert.equal(s.text(8001), "newer");
  assert.equal(s.text(8000 + 5000), "");
});

test("cropBox scales the upper crop to the box width", () => {
  const box = cropBox(320, { width: 384, height: 672 }, [0, 0, 384, 350]);
  assert.deepEqual(box, { width: 320, height: 350 * (320 / 384), imgWidth: 320, imgHeight: 672 * (320 / 384), left: 0, top: 0 });
  const off = cropBox(100, { width: 400, height: 800 }, [100, 50, 200, 100]);
  assert.deepEqual(off, { width: 100, height: 50, imgWidth: 200, imgHeight: 400, left: -50, top: -25 });
});

test("toImageCoords maps client coordinates into image pixels", () => {
  const rect = { left: 10, top: 20, width: 150, height: 300 };
  assert.deepEqual(toImageCoords(rect, 300, 600, 85, 170), { x: 150, y: 300 });
  assert.equal(toImageCoords({ left: 0, top: 0, width: 0, height: 0 }, 300, 600, 1, 1), null);
});
