"use strict";
const test = require("node:test");
const assert = require("node:assert/strict");
const {
  ExpressionModel, MouthTracker, StatusLine, cropBox, CURIOUS_BID, idleAnimates, Motion, transformCss, nextDeadline,
  isSettingsError, buddySpriteBox,
} = require("../src/stage-logic");
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
  m.onSay({ kind: "bid", fresh: true });
  assert.equal(CURIOUS_BID, "curious");
  assert.equal(m.current(true), "curious");
  m.onState({ expression: "sleepy" }); // a state push does not cancel the override
  m.onSay({ kind: "bid", fresh: false }); // a later sentence of the same bid keeps it
  assert.equal(m.current(true), "curious");
  assert.equal(m.current(false), "sleepy", "once the bubble is gone she goes back");
  assert.equal(m.current(true), "sleepy", "and the override is spent");

  m.onSay({ kind: "bid", fresh: true });
  m.onExpression("worried");
  assert.equal(m.current(true), "worried");
});

test("a voice reply clears the bid's curious face", () => {
  const m = new ExpressionModel();
  m.onState({ expression: "happy" });
  m.onSay({ kind: "bid", fresh: true });
  assert.equal(m.current(true), "curious");
  m.onSay({ kind: "reply", fresh: true });
  assert.equal(m.current(true), "happy");
  m.onSay({ kind: "bid", fresh: true });
  m.onSay({ fresh: false }); // no kind counts as a reply
  assert.equal(m.current(true), "happy");
});

test("idle animation (breathing, blink, lean) runs only while visible and not gaming", () => {
  assert.equal(idleAnimates({ presence: "idle", hidden: false }), true);
  assert.equal(idleAnimates({ presence: null, hidden: false }), true);
  assert.equal(idleAnimates({ presence: "focused", hidden: false }), true);
  assert.equal(idleAnimates({ presence: "gaming", hidden: false }), false);
  assert.equal(idleAnimates({ presence: "idle", hidden: true }), false);
});

test("transformCss rounds to 0.2 px and 0.2 degree", () => {
  assert.equal(transformCss({ dx: -11.96, dy: -0.04, rotateDeg: -5.97, scale: 1.000004 }),
    "translate(-12px, 0px) rotate(-6deg) scale(1)");
  assert.equal(transformCss({ dx: 0.04, dy: -1.26, rotateDeg: 0.09, scale: 1.0051 }),
    "translate(0px, -1.2px) rotate(0deg) scale(1.0051)");
  assert.equal(transformCss({ dx: 0.09, dy: -1.31, rotateDeg: 0.11, scale: 1.00512 }),
    "translate(0px, -1.4px) rotate(0.2deg) scale(1.0051)");
  assert.equal(transformCss({ dx: 0.01, dy: -1.21, rotateDeg: 0.04, scale: 1.00512 }),
    transformCss({ dx: 0.04, dy: -1.26, rotateDeg: 0.01, scale: 1.0051 }), "sub-threshold changes give the same text");
});

test("breathing changes the transform text on only about half the 15 Hz ticks (the rest skip the DOM write)", () => {
  const m = new Motion();
  let writes = 0;
  let prev = "";
  for (let t = 0; t < 8000; t += 1000 / 15) {
    const css = m.step(t, { rotateDeg: 0, dx: 0 });
    if (css !== prev) writes += 1;
    prev = css;
  }
  assert.ok(writes < 120 * 0.6, `writes: ${writes}`);
});

test("Motion.resume clamps the first step after a pause to 2 frames, so the lean eases instead of snapping", () => {
  const target = { rotateDeg: -6, dx: -12 };
  const m = new Motion();
  m.step(0, target);
  m.step(66, target);
  const before = m.dx;
  m.resume();
  m.step(60000, target); // a minute later
  assert.ok(Math.abs(m.dx - (before + (-12 - before) * (1 - Math.pow(0.92, 2)))) < 1e-9, `eased by 2 frames: ${m.dx}`);
  const after = m.dx;
  m.step(60000 + 4 * (1000 / 60), target); // next normal tick eases by its real 4 frames
  const k4 = 1 - Math.pow(0.92, 4);
  assert.ok(Math.abs(m.dx - (after + (-12 - after) * k4)) < 1e-6, "clamp applies to one step only");
});

test("Motion without resume() still eases by the real elapsed time", () => {
  const m = new Motion();
  m.step(0, { rotateDeg: 0, dx: 0 });
  m.step(60000, { rotateDeg: -6, dx: -12 });
  assert.ok(Math.abs(m.dx - -12) < 1e-6);
});

test("Motion.resume before the very first step does not move the lean", () => {
  const m = new Motion();
  m.resume();
  m.step(5000, { rotateDeg: -6, dx: -12 });
  assert.equal(m.dx, 0);
});

test("Motion eases the lean by elapsed time, so a 15 Hz timer matches 60 fps", () => {
  const target = { rotateDeg: -6, dx: -12 };
  const fast = new Motion();
  const slow = new Motion();
  for (let i = 0; i <= 60; i += 1) fast.step((i * 1000) / 60, target);
  for (let i = 0; i <= 15; i += 1) slow.step((i * 1000) / 15, target);
  assert.ok(Math.abs(fast.dx - slow.dx) < 1e-9 && Math.abs(fast.rotateDeg - slow.rotateDeg) < 1e-9);
  assert.ok(fast.dx < -12 * 0.9, `eased most of the way in a second: ${fast.dx}`);
});

test("Motion gives the same transform text once the lean has settled and breathing repeats", () => {
  const m = new Motion();
  const target = { rotateDeg: -6, dx: -12 };
  for (let t = 0; t <= 10000; t += 67) m.step(t, target);
  assert.equal(m.step(12000, target), m.step(16000, target));
  assert.match(m.step(20000, target), /^translate\(-12px, 0px\) rotate\(-6deg\) scale\(1\)$/);
});

test("nextDeadline is the earliest future time, or null", () => {
  assert.equal(nextDeadline(100, [null, 300, 250, undefined]), 250);
  assert.equal(nextDeadline(100, [100, 50, null]), null);
  assert.equal(nextDeadline(100, []), null);
});

test("MouthTracker.closesAt is when silence will close an open mouth", () => {
  const t = new MouthTracker({ silenceMs: 250 });
  assert.equal(t.closesAt(), null);
  t.onLevel(0.5, 100);
  assert.equal(t.closesAt(), 350);
  t.onLevel(0.05, 133);
  assert.equal(t.closesAt(), null, "already closed");
});

test("StatusLine.clearsAt is when the error line disappears", () => {
  const s = new StatusLine({ ttlMs: 5000 });
  assert.equal(s.clearsAt(0), null);
  s.show("model is down", 1000);
  assert.equal(s.clearsAt(2000), 6000);
  assert.equal(s.clearsAt(6000), null);
});

test("isSettingsError spots errors that name a settings key", () => {
  assert.equal(isSettingsError("invalid value for setting 'voice': ''"), true);
  assert.equal(isSettingsError("unknown setting 'colour'"), true);
  assert.equal(isSettingsError("couldn't change autostart: access denied"), true);
  assert.equal(isSettingsError("the model is not responding"), false);
  assert.equal(isSettingsError("fact 3 is superseded; correct its current version instead"), false);
  assert.equal(isSettingsError(undefined), false);
});

test("buddySpriteBox keeps her face below a 40vh bubble anchored above her head", () => {
  const win = { winWidth: 300, winHeight: 520, bubbleBottom: 216, gap: 12 };
  // placeholder pack: the upper crop starts at the canvas top, so the whole box must clear the bubble
  const ph = buddySpriteBox({ ...win, canvas: { width: 384, height: 672 }, crops: { upper: [0, 0, 384, 350] } });
  assert.equal(ph.height, 520 - 216 - 12);
  assert.ok(Math.abs(ph.width - ph.height * (384 / 672)) < 1e-9);
  // a real pack: the crop (face box) starts 20.6% down the canvas, so she can be taller
  const real = buddySpriteBox({ ...win, canvas: { width: 768, height: 1344 }, crops: { upper: [216, 277, 295, 870] } });
  const faceTop = 520 - real.height + real.height * (277 / 1344);
  assert.ok(Math.abs(faceTop - (216 + 12)) < 1e-9, `face top ${faceTop}`);
  assert.ok(real.height > ph.height && real.height <= 520 && real.width <= 300);
  // never taller than the window or wider than it
  const low = buddySpriteBox({ ...win, canvas: { width: 100, height: 100 }, crops: { upper: [0, 90, 100, 10] } });
  assert.deepEqual(low, { width: 300, height: 300 });
  const tall = buddySpriteBox({ ...win, canvas: { width: 10, height: 100 }, crops: { upper: [0, 95, 10, 5] } });
  assert.equal(tall.height, 520);
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
