"use strict";
const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const {
  ACTIVITY_KINDS, PET_WALKS_WHEN, REDUCED_MOTION, SCALE_RANGE, BUBBLE_MAX, PET, STATES, REACTIONS, POKE, BIG_EFFECTS,
  BIG_MOTIONS, OFF_VIEW, Pet, makeRng, clampScale, petSize, floorY, walkBounds, inQuietHours, reducedMotionOn, gates,
  bubbleText, petTransformCss, walkFrame, mirrorForFacing,
} = require("../src/pet-logic");

const AREA = { x: 0, y: 0, width: 1920, height: 1040 };
const CANVAS = { width: 384, height: 672 };
const SIZE = petSize({ canvas: CANVAS, scale: 0.6 });
const HI = () => 0.99; // never a 1-in-50 surprise, always the late end of every range
const BASE = new Date(2026, 0, 5, 10, 0).getTime(); // a local 10:00, for tests that read quiet hours
const FOCUSED = { presence: "focused", idleS: 60 };

function mk(over = {}) {
  const { ctx, ...rest } = over;
  return new Pet({ rng: makeRng(1), nowMs: 0, area: AREA, size: SIZE, ctx: { ...FOCUSED, ...ctx }, ...rest });
}

/** Step every `stepMs` from `from` (exclusive) to `to` (inclusive); returns the last frame and calls `each(frame, t)`. */
function run(pet, from, to, stepMs = 50, each = null) {
  let frame = null;
  for (let t = from + stepMs; t <= to; t += stepMs) {
    frame = pet.step(t);
    if (each) each(frame, t);
  }
  return frame;
}

/** Step until `pred(frame)`; returns { frame, t }. */
function until(pet, from, pred, limit = 4 * 3600 * 1000, stepMs = 50) {
  for (let t = from; t <= from + limit; t += stepMs) {
    const frame = pet.step(t);
    if (pred(frame)) return { frame, t };
  }
  assert.fail("the condition never came");
}

test("petSize follows pet_scale", () => {
  assert.deepEqual(petSize({ canvas: CANVAS, scale: 0.6 }), { spriteH: 240, spriteW: 137, width: 240, height: 344 });
  assert.deepEqual(petSize({ canvas: CANVAS, scale: 1.0 }), { spriteH: 400, spriteW: 229, width: 240, height: 504 });
  assert.deepEqual(petSize({ canvas: CANVAS, scale: 0.4 }), { spriteH: 160, spriteW: 91, width: 240, height: 264 });
  assert.deepEqual(petSize({ canvas: CANVAS, scale: 5 }), petSize({ canvas: CANVAS, scale: 1.0 }));
  assert.deepEqual(petSize({ canvas: CANVAS, scale: NaN }), petSize({ canvas: CANVAS, scale: 0.6 }));
  const wide = petSize({ canvas: { width: 1000, height: 500 }, scale: 1.0 });
  assert.equal(wide.width, 800);
  assert.equal(wide.height, 504);
  assert.deepEqual(petSize({ canvas: null, scale: 0.6 }), petSize({ canvas: CANVAS, scale: 0.6 }));
  assert.deepEqual(petSize({ canvas: { width: 0, height: 0 }, scale: 0.6 }), petSize({ canvas: CANVAS, scale: 0.6 }));
  assert.equal(clampScale(0.1), 0.4);
  assert.equal(clampScale(Infinity), 0.6);
  assert.deepEqual(SCALE_RANGE, [0.4, 1.0]);
});

test("geometry", () => {
  assert.equal(floorY(AREA, 344), 696);
  assert.deepEqual(walkBounds(AREA, 240), { minX: 0, maxX: 1680 });
  assert.deepEqual(walkBounds({ x: 10, y: 0, width: 100, height: 100 }, 240), { minX: 10, maxX: 10 }, "a narrow area");
});

test("inQuietHours and reducedMotionOn", () => {
  const at = (h, m) => new Date(2026, 0, 5, h, m);
  assert.equal(inQuietHours("23:30-08:00", at(23, 45)), true);
  assert.equal(inQuietHours("23:30-08:00", at(7, 59)), true);
  assert.equal(inQuietHours("23:30-08:00", at(12, 0)), false);
  assert.equal(inQuietHours("09:00-17:00", at(9, 0)), true);
  assert.equal(inQuietHours("09:00-17:00", at(17, 0)), false);
  assert.equal(inQuietHours(null, at(12, 0)), false);
  assert.equal(inQuietHours("08:00-08:00", at(8, 0)), false);
  assert.equal(inQuietHours("bad", at(12, 0)), false);
  assert.equal(inQuietHours(42, at(12, 0)), false);
  assert.equal(inQuietHours("23:30-08:00", new Date(NaN)), false);
  assert.equal(reducedMotionOn("on", false), true);
  assert.equal(reducedMotionOn("off", true), false);
  assert.equal(reducedMotionOn("system", true), true);
  assert.equal(reducedMotionOn("system", false), false);
  assert.equal(reducedMotionOn("system", undefined), false);
  assert.equal(reducedMotionOn("junk", true), true);
});

test("gates follow the table", () => {
  const base = { enabled: true, presence: "focused", idleS: 0, reducedMotion: false, walksWhen: "idle", forceWalk: false };
  const g = (ctx, extra = {}) => gates({ ...base, ...ctx }, { quiet: false, sinceAgentS: 0, ...extra });
  const all = (v) => ({ walk: v, big: v, show: v, sleep: v });
  assert.deepEqual(g({ enabled: false, idleS: 999, forceWalk: true }), all(false), "disabled");
  assert.deepEqual([g({ presence: "gaming", idleS: 999 }).show, g({ presence: "gaming", idleS: 999 }).walk], [false, false]);
  assert.equal(g({ presence: "gaming", idleS: 999 }).big, false);
  const away = g({ presence: "away" });
  assert.deepEqual([away.sleep, away.show, away.walk], [true, false, false]);
  const unknown = g({ presence: null, idleS: 60 });
  assert.deepEqual([unknown.walk, unknown.big, unknown.show], [false, false, true]);
  const quiet = g({ idleS: 60 }, { quiet: true });
  assert.deepEqual([quiet.walk, quiet.big, quiet.sleep], [false, false, true]);
  const reduced = g({ idleS: 60, reducedMotion: true });
  assert.deepEqual([reduced.walk, reduced.big, reduced.show], [false, false, true]);
  const near = g({ idleS: 10 });
  assert.deepEqual([near.walk, near.big], [false, false]);
  const far = g({ idleS: 30 });
  assert.deepEqual([far.walk, far.big], [true, true]);
  const eager = g({ idleS: 10, walksWhen: "always_but_focus" });
  assert.deepEqual([eager.walk, eager.big], [true, false]);
  assert.equal(g({ idleS: 4, walksWhen: "always_but_focus" }).walk, false);
  const asleep = g({ idleS: 600 }, { sinceAgentS: 600 });
  assert.deepEqual([asleep.sleep, asleep.walk], [true, false]);
  assert.equal(g({ idleS: 600 }, { sinceAgentS: 599 }).sleep, false, "a recent agent event keeps her up");
  assert.equal(g({ presence: "break", idleS: 30 }).walk, true);
  assert.equal(g({ presence: "talking", idleS: 30 }).walk, true);
  assert.equal(g({ forceWalk: true, presence: null, idleS: 0 }).walk, true);
  assert.equal(g({ forceWalk: true, presence: "focused", idleS: 0 }).walk, true);
  assert.equal(g({ forceWalk: true, reducedMotion: true }).walk, false);
  assert.equal(g({ forceWalk: true, presence: "gaming" }).walk, false);
  assert.equal(g({ forceWalk: true, presence: "away" }).walk, false);
  assert.equal(g({ forceWalk: true }, { quiet: true }).walk, false);
});

test("walks come in 3-10 s bursts every 40-180 s at 60-90 DIP/s", () => {
  const pet = mk();
  const durations = [];
  const gaps = [];
  let inWalk = false;
  let startedAt = 0;
  let lastEnd = 0;
  let lastX = pet.step(0).x;
  const TWO_HOURS = 2 * 3600 * 1000;
  run(pet, 0, TWO_HOURS, 50, (frame, t) => {
    const walking = frame.view.state === "walk";
    if (walking && !inWalk) { gaps.push(t - lastEnd); startedAt = t; }
    if (!walking && inWalk) { durations.push(t - startedAt); lastEnd = t; }
    inWalk = walking;
    assert.ok(Math.abs(frame.x - lastX) <= 90 * 0.05 + 1, `a step moved ${frame.x - lastX}`);
    lastX = frame.x;
  });
  assert.ok(durations.length >= 20, `only ${durations.length} walks`);
  for (const d of durations) assert.ok(d >= 3000 && d <= 10000 + PET.MOVE_MS, `walk of ${d} ms`);
  for (const gap of gaps) assert.ok(gap >= 40000 && gap <= 180000 + PET.MOVE_MS, `gap of ${gap} ms`);
});

test("a due walk never starts while focused, gaming, quiet, reduced or unknown", () => {
  const contexts = [
    { presence: "focused", idleS: 10 },
    { presence: "gaming", idleS: 999 },
    { presence: "focused", idleS: 60, quietHours: "09:00-18:00" },
    { presence: "focused", idleS: 60, reducedMotion: true },
    { presence: null, idleS: 60 },
  ];
  for (const ctx of contexts) {
    const pet = mk({ nowMs: BASE, ctx });
    run(pet, BASE, BASE + 3600 * 1000, 250, (frame) => {
      assert.notEqual(frame.view.state, "walk", JSON.stringify(ctx));
      assert.equal(pet.ticking(), false, JSON.stringify(ctx));
    });
  }
});

test("a walk never leaves the work area or covers the taskbar, on any monitor", () => {
  const areas = [
    { x: -1920, y: 0, width: 1920, height: 1040 },
    { x: 1920, y: -200, width: 2560, height: 1400 },
    { x: 0, y: 0, width: 300, height: 800 },
  ];
  for (const area of areas) {
    const pet = mk({ area, ctx: { presence: null, idleS: 0, forceWalk: true } });
    const { minX, maxX } = walkBounds(area, SIZE.width);
    const floor = floorY(area, SIZE.height);
    let atMin = false;
    let atMax = false;
    let walked = false;
    run(pet, 0, 30 * 60 * 1000, 50, (frame) => {
      assert.ok(Number.isInteger(frame.x) && Number.isInteger(frame.y));
      assert.ok(frame.x >= minX && frame.x <= maxX, `x ${frame.x} outside ${minX}..${maxX}`);
      assert.ok(frame.y + SIZE.height <= area.y + area.height);
      if (frame.view.state === "walk") {
        walked = true;
        assert.equal(frame.y, floor);
        if (frame.x === minX) atMin = true;
        if (frame.x === maxX) atMax = true;
      }
    });
    assert.ok(walked && atMin && atMax, `turned at both edges on ${JSON.stringify(area)}`);
  }
});

test("a shrunk work area re-clamps her at once", () => {
  const pet = mk({ pos: { x: 1600, y: 696 }, ctx: { presence: null, idleS: 0, forceWalk: true } });
  const { t } = until(pet, 0, (f) => f.view.state === "walk");
  pet.setArea({ x: 0, y: 0, width: 1280, height: 960 }, t);
  const frame = pet.step(t + 50);
  assert.ok(frame.x <= 1040, `x ${frame.x}`);
  assert.equal(frame.y, 616);
  assert.equal(frame.view.state, "walk", "she keeps walking");
  assert.equal(frame.moved, true);
});

test("falls land on the floor with a squash, then stand and save", () => {
  const pet = mk();
  pet.step(0);
  pet.grab(1000);
  assert.equal(pet.drop({ x: 100, y: 696 - 300 }, 1000), "fall");
  let sawSurprise = false;
  let landedAt = null;
  let saves = 0;
  let idleAfterLanding = false;
  run(pet, 1000, 3000, 50, (frame, t) => {
    if (frame.view.state === "fall" && frame.view.expression === "surprised") sawSurprise = true;
    if (landedAt === null && frame.view.motion === "squash") {
      landedAt = t;
      assert.equal(frame.y, 696);
    }
    if (frame.save) saves += 1;
    if (landedAt !== null && frame.view.state === "idle") { idleAfterLanding = true; assert.equal(frame.y, 696); }
  });
  assert.ok(sawSurprise);
  assert.ok(landedAt !== null && landedAt - 1000 <= 550, `landed after ${landedAt - 1000} ms`);
  assert.ok(idleAfterLanding);
  assert.equal(saves, 1);
  assert.equal(pet.ticking(), false);

  const calm = mk({ ctx: { reducedMotion: true } });
  calm.grab(0);
  assert.equal(calm.drop({ x: 100, y: 396 }, 0), "idle");
  const frame = calm.step(50);
  assert.equal(frame.y, 396);
  assert.equal(frame.save, true);
  assert.equal(frame.view.state, "idle");

  const flat = mk();
  flat.grab(0);
  assert.equal(flat.drop({ x: 100, y: 696 }, 0), "idle");
  assert.equal(flat.step(50).save, true);
});

test("a fall ticks until it lands and a taskbar that moves away makes her fall", () => {
  const pet = mk();
  pet.step(0);
  pet.grab(0);
  pet.drop({ x: 100, y: 300 }, 0);
  assert.equal(pet.ticking(), true);
  run(pet, 0, 1000);
  assert.equal(pet.ticking(), false);
  pet.setArea({ x: 0, y: 0, width: 1920, height: 1080 }, 1100); // taskbar auto-hid: the floor dropped by 40
  const frame = pet.step(1150);
  assert.equal(frame.view.state, "fall");
});

test("a click releases her without saving", () => {
  const pet = mk();
  const start = pet.step(0);
  pet.grab(1000);
  assert.equal(pet.step(1050).view.state, "drag");
  assert.equal(pet.drop({ x: start.x, y: start.y }, 1100, { save: false }), "idle");
  const frame = pet.step(1150);
  assert.notEqual(frame.view.state, "drag");
  assert.equal(frame.save, false);
  pet.poke(1200);
  assert.equal(pet.step(1250).view.expression, POKE.expression);
});

test("a drop on another monitor stays there", () => {
  const other = { x: 1920, y: 0, width: 2560, height: 1400 };
  const pet = mk();
  pet.step(0);
  pet.grab(100);
  pet.setArea(other, 100);
  assert.equal(pet.drop({ x: 3000, y: floorY(other, 344) }, 100), "idle");
  const frame = pet.step(150);
  assert.equal(frame.x, 3000);
  assert.equal(frame.y, 1056);

  const stale = mk();
  stale.step(0);
  stale.grab(100);
  assert.equal(stale.drop({ x: 3000, y: 1056 }, 100), "idle");
  assert.equal(stale.step(150).x, 1680, "without the new area she is clamped back");
});

test("drag dangles flustered or excited and never ticks", () => {
  const pet = mk();
  pet.step(0);
  pet.grab(100);
  const frame = pet.step(150);
  assert.equal(frame.view.state, "drag");
  assert.equal(frame.view.motion, "dangle");
  assert.ok(["flustered", "excited"].includes(frame.view.expression));
  assert.equal(pet.ticking(), false);
  pet.setArea({ x: 0, y: 0, width: 1280, height: 960 }, 200);
  pet.setSize({ width: 300, height: 400 }, 200);
  const during = pet.step(250);
  assert.equal(during.moved, false);
  assert.equal(during.save, false);
  assert.equal(during.view.state, "drag");
});

test("OFF_VIEW is a full neutral view", () => {
  assert.deepEqual(Object.keys(OFF_VIEW), Object.keys(mk().step(0).view));
  assert.equal(OFF_VIEW.on, false);
  assert.ok(Object.isFrozen(OFF_VIEW));
  assert.deepEqual({ ...OFF_VIEW }, {
    on: false, state: "idle", facing: 1, expression: null, motion: null, effect: null, effectSeq: 0, bubble: null,
    reduced: false, motionAt: 0,
  });
  const off = mk({ ctx: { enabled: false } });
  assert.deepEqual(off.step(0).view, OFF_VIEW);
  assert.equal(off.nextWakeMs(0), null);
  assert.equal(mk().step(0).view.on, true);
});

test("the constants are the brief's", () => {
  assert.equal(ACTIVITY_KINDS.length, 12);
  assert.deepEqual(Object.keys(REACTIONS), ACTIVITY_KINDS);
  assert.deepEqual(STATES, ["idle", "walk", "sit", "sleep", "drag", "fall", "react"]);
  assert.deepEqual(PET_WALKS_WHEN, ["idle", "always_but_focus"]);
  assert.deepEqual(REDUCED_MOTION, ["system", "on", "off"]);
  assert.equal(BUBBLE_MAX, 40);
  assert.deepEqual(BIG_EFFECTS, ["confetti", "heart"]);
  assert.deepEqual(BIG_MOTIONS, ["spin"]);
  assert.ok(Object.isFrozen(PET) && Object.isFrozen(REACTIONS) && Object.isFrozen(POKE));
  assert.equal(PET.MAX_DT_MS, 250);
  assert.deepEqual(PET.WALK_SPEED, [60, 90]);
  assert.deepEqual(REACTIONS.wave, { expression: "excited", motion: "wave", effect: null, ms: 6000, bubbleOnly: false });
  assert.deepEqual(REACTIONS.edit, { expression: null, motion: null, effect: null, ms: 0, bubbleOnly: true });
});

/** A pet that is walking right now (rng HI keeps every roll deterministic and unsurprising). */
function walking(ctx = {}) {
  const pet = mk({ rng: HI, ctx });
  const { t } = until(pet, 0, (f) => f.view.state === "walk", 400000);
  return { pet, t };
}

test("reactions follow the table", () => {
  for (const kind of ACTIVITY_KINDS) {
    const row = REACTIONS[kind];
    if (row.bubbleOnly) {
      const { pet, t } = walking();
      pet.react(kind, "editing engine.py", t);
      const frame = pet.step(t + 50);
      assert.equal(frame.view.state, "walk", `${kind} keeps her walking`);
      assert.equal(frame.view.bubble.text, "editing engine.py");
      assert.equal(frame.view.bubble.untilMs, t + PET.BUBBLE_MS);
      assert.equal(pet.ticking(), true);
    } else {
      const pet = mk({ rng: HI });
      pet.step(0);
      pet.react(kind, "x", 100);
      const frame = pet.step(150);
      assert.equal(frame.view.state, "react", kind);
      assert.equal(frame.view.expression, row.expression, kind);
      assert.equal(frame.view.motion, row.motion, kind);
      assert.equal(frame.view.effect, row.effect, kind);
      assert.equal(frame.view.effectSeq, row.effect ? 1 : 0, kind);
      assert.equal(frame.view.bubble.untilMs, 150 + Math.max(row.ms, PET.BUBBLE_MS), kind);
      const end = pet.step(150 + row.ms + 100);
      assert.equal(end.view.state, "sit", `${kind} ends in a sit`);
      assert.equal(end.view.effect, null);
    }
  }
});

test("wave stops a walk and shows the bubble", () => {
  const { pet, t } = walking();
  pet.react("wave", "needs you! running tests", t);
  const frame = pet.step(t + 50);
  assert.equal(frame.view.state, "react");
  assert.equal(frame.view.expression, "excited");
  assert.equal(frame.view.motion, "wave");
  assert.equal(frame.view.bubble.text, "needs you! running tests");
  assert.equal(pet.ticking(), false);
  assert.equal(frame.save, true, "the walk's end saves where she stopped");
});

test("three pose kinds in a row queue and run in order; the oldest of a fourth drops", () => {
  const order = (kinds) => {
    const pet = mk({ rng: HI });
    pet.step(0);
    for (const k of kinds) pet.react(k, k, 100);
    const seen = [];
    run(pet, 100, 100 + 20000, 50, (f) => {
      const e = f.view.expression;
      if (f.view.state === "react" && seen[seen.length - 1] !== e) seen.push(e);
    });
    return seen;
  };
  assert.deepEqual(order(["think", "done", "fail"]), ["thinking", "happy", "sad"]);
  assert.deepEqual(order(["think", "done", "fail", "oops"]), ["happy", "sad", "worried"], "think was the oldest");
  assert.deepEqual(order(["done", "done"]), ["happy"], "the same kind replaces itself");
  const pet = mk({ rng: HI });
  pet.step(0);
  pet.react("done", "first", 100);
  pet.react("done", "second", 100);
  assert.equal(pet.step(150).view.bubble.text, "second");
});

test("a wave is the highest priority: it jumps the queue and cuts a running pose short", () => {
  const pet = mk({ rng: HI });
  pet.step(0);
  pet.react("think", "t", 100);
  pet.step(150);
  pet.react("done", "d", 200);
  pet.react("wave", "needs you!", 250);
  const frame = pet.step(300);
  assert.equal(frame.view.motion, "wave", "it does not wait for the think pose or the queued finish");
  assert.equal(frame.view.bubble.text, "needs you!");
  const seen = [];
  run(pet, 300, 300 + 20000, 50, (f) => {
    if (f.view.state === "react" && seen[seen.length - 1] !== f.view.expression) seen.push(f.view.expression);
  });
  assert.deepEqual(seen, ["excited", "happy"], "the finish still runs after it");
  // a wave never cuts another wave
  const two = mk({ rng: HI });
  two.step(0);
  two.react("wave", "first", 100);
  two.step(150);
  two.react("wave", "second", 200);
  const second = two.step(250).view;
  assert.equal(second.motion, "wave");
  assert.equal(second.bubble.text, "second", "the latest bubble wins");
  assert.equal(second.effectSeq, 0);
});

test("during a drag the pose waits until the drop, but a wave's bubble shows at once", () => {
  const pet = mk({ rng: HI });
  pet.step(0);
  pet.grab(100);
  pet.react("done", "all done!", 150);
  const during = pet.step(200);
  assert.equal(during.view.state, "drag");
  assert.equal(during.view.bubble, null);
  pet.drop({ x: 100, y: 696 }, 250);
  assert.equal(pet.step(300).view.state, "react");

  const w = mk({ rng: HI });
  w.step(0);
  w.grab(100);
  w.react("wave", "needs you!", 150);
  const frame = w.step(200);
  assert.equal(frame.view.state, "drag");
  assert.equal(frame.view.bubble.text, "needs you!");
  w.drop({ x: 100, y: 696 }, 250);
  assert.equal(w.step(300).view.motion, "wave");
});

test("while gaming a reaction shows nothing; while focused a finish is pose and bubble only", () => {
  const gaming = mk({ rng: HI, ctx: { presence: "gaming", idleS: 999 } });
  const before = gaming.step(0).view;
  gaming.react("done", "all done!", 100);
  gaming.react("wave", "needs you!", 100);
  const after = gaming.step(150).view;
  assert.deepEqual(after, { ...before });

  const focused = mk({ rng: HI, ctx: { presence: "focused", idleS: 5 } });
  focused.step(0);
  focused.react("done", "all done!", 100);
  const f = focused.step(150).view;
  assert.equal(f.expression, "happy");
  assert.equal(f.effect, null);
  assert.equal(f.bubble.text, "all done!");

  const free = mk({ rng: HI, ctx: { presence: "focused", idleS: 60 } });
  free.step(0);
  free.react("done", "all done!", 100);
  assert.equal(free.step(150).view.effect, "confetti");

  const calm = mk({ rng: HI, ctx: { presence: "focused", idleS: 60, reducedMotion: true } });
  calm.step(0);
  calm.react("done", "all done!", 100);
  const c = calm.step(150).view;
  assert.deepEqual([c.expression, c.motion, c.effect, c.bubble.text], ["happy", null, null, "all done!"]);
  calm.react("oops", "oops", 5000);
  const o = calm.step(5050).view;
  assert.deepEqual([o.expression, o.effect], ["worried", null], "the sweat drop is an effect too");

  const away = mk({ rng: HI, ctx: { presence: "away", idleS: 999 } });
  assert.equal(away.step(0).view.state, "sleep");
  away.react("done", "all done!", 100);
  away.react("edit", "editing a.py", 100);
  const a = away.step(150).view;
  assert.equal(a.state, "sleep");
  assert.equal(a.bubble, null);

  const unknown = mk({ rng: HI, ctx: { presence: null, idleS: 60 } });
  unknown.step(0);
  unknown.react("done", "all done!", 100);
  assert.equal(unknown.step(150).view.effect, null, "unknown presence is not a not-focused user");
  unknown.react("oops", "oops", 5000);
  assert.equal(unknown.step(5050).view.effect, "sweat", "sweat is never big");
});

test("gaming or away ends a running reaction at once", () => {
  const pet = mk({ rng: HI });
  pet.step(0);
  pet.react("done", "all done!", 100);
  const on = pet.step(150).view;
  assert.deepEqual([on.state, on.effect, on.bubble !== null], ["react", "confetti", true]);
  pet.setContext({ presence: "gaming" }, 200);
  const gaming = pet.step(200).view;
  assert.notEqual(gaming.state, "react");
  assert.equal(gaming.effect, null);
  assert.equal(gaming.bubble, null);

  const away = mk({ rng: HI });
  away.step(0);
  away.react("wave", "needs you!", 100);
  assert.equal(away.step(150).view.state, "react");
  away.setContext({ presence: "away" }, 200);
  const asleep = away.step(200).view;
  assert.equal(asleep.state, "sleep");
  assert.equal(asleep.bubble, null);
  assert.equal(asleep.effect, null);
  away.setContext({ presence: "focused" }, 300);
  assert.equal(away.step(300).view.state, "sit", "and she is not left holding a stale reaction or queue");
});

test("a bubble-only label does not replace a running wave's bubble", () => {
  const pet = mk({ rng: HI });
  pet.step(0);
  pet.react("wave", "needs you! running tests", 100);
  pet.step(150);
  pet.react("edit", "editing a.py", 200);
  assert.equal(pet.step(250).view.bubble.text, "needs you! running tests");
  pet.react("wave", "needs you! again", 300); // a new wave still follows the normal rules
  assert.equal(pet.step(350).view.bubble.text, "needs you! again");
  const after = run(pet, 350, 350 + 15000); // the first wave, then the queued second one
  assert.equal(after.view.state, "sit");
  pet.react("edit", "editing a.py", 20000);
  assert.equal(pet.step(20050).view.bubble.text, "editing a.py", "once the wave is over a label shows again");
  // a pose that is not a wave is not shielded
  const t = mk({ rng: HI });
  t.step(0);
  t.react("think", "thinking", 100);
  t.step(150);
  t.react("edit", "editing a.py", 200);
  assert.equal(t.step(250).view.bubble.text, "editing a.py");
});

test("reduced motion mid-fall snaps her to the floor without a squash", () => {
  const pet = mk();
  pet.step(0);
  pet.grab(0);
  assert.equal(pet.drop({ x: 100, y: 300 }, 0), "fall");
  pet.step(50);
  pet.setContext({ reducedMotion: true }, 100);
  const frame = pet.step(100);
  assert.equal(frame.y, 696);
  assert.equal(frame.view.state, "idle");
  assert.equal(frame.view.motion, null);
  assert.equal(frame.save, true);
  assert.equal(pet.ticking(), false);
  let squashed = false;
  run(pet, 100, 2000, 50, (f) => { if (f.view.motion === "squash") squashed = true; assert.equal(f.save, false); });
  assert.equal(squashed, false);
});

test("a finish surprises about 1 in 50, and never while focused", () => {
  const count = (idleS, seed) => {
    const pet = mk({ rng: makeRng(seed), ctx: { presence: "focused", idleS } });
    pet.step(0);
    let surprises = 0;
    for (let i = 1; i <= 5000; i += 1) {
      const t = i * 10000;
      pet.react("done", "all done!", t);
      const v = pet.step(t).view;
      if (v.effect === "heart" || v.motion === "spin") surprises += 1;
      pet.step(t + 4500);
    }
    return surprises;
  };
  const n = count(60, 7);
  assert.ok(n >= 60 && n <= 140, `${n} surprises in 5000`);
  assert.equal(count(5, 7), 0);
});

test("sleep and wake", () => {
  const away = mk({ ctx: { presence: "away", idleS: 0 } });
  assert.equal(away.step(0).view.state, "sleep");
  assert.equal(away.step(0).view.expression, "sleepy");
  assert.equal(away.step(0).view.motion, "sleep");
  away.setContext({ presence: "focused" }, 100);
  assert.equal(away.step(100).view.state, "sit");

  const quiet = mk({ rng: HI, nowMs: BASE, ctx: { quietHours: "09:00-18:00" } });
  assert.equal(quiet.step(BASE).view.state, "sleep");
  quiet.react("wave", "needs you!", BASE + 100);
  assert.equal(quiet.step(BASE + 150).view.state, "react");
  let slept = false;
  run(quiet, BASE + 150, BASE + 20000, 50, (f) => { if (f.view.state === "sleep") slept = true; });
  assert.ok(slept, "she sleeps again after");
  quiet.react("edit", "editing a.py", BASE + 21000);
  const bubble = quiet.step(BASE + 21050).view;
  assert.equal(bubble.state, "sit", "a bubble wakes her");
  assert.equal(quiet.step(BASE + 22000).view.state, "sit", "and keeps her up while it shows");
  assert.equal(quiet.step(BASE + 21050 + PET.BUBBLE_MS + 100).view.state, "sleep");

  const idle = mk({ ctx: { presence: "focused", idleS: 0 } });
  idle.step(0);
  idle.react("edit", "editing a.py", 0);
  idle.setContext({ idleS: 600 }, 0);
  assert.notEqual(idle.step(599000).view.state, "sleep");
  assert.equal(idle.step(600000).view.state, "sleep");
  idle.react("edit", "editing a.py", 601000);
  assert.equal(idle.step(601050).view.state, "sit", "an edit bubble wakes her to sit");
  assert.equal(idle.step(602000).view.state, "sit");

  const manual = mk({ rng: HI });
  manual.step(0);
  manual.sleepNow(100);
  assert.equal(manual.step(150).view.state, "sleep");
  manual.react("edit", "editing a.py", 200);
  assert.equal(manual.step(250).view.state, "sleep", "sleepNow ignores a bubble");
  manual.react("wave", "needs you!", 300);
  assert.equal(manual.step(350).view.state, "react", "but wakes on a wave");
  const again = mk({ rng: HI });
  again.step(0);
  again.sleepNow(100);
  again.poke(200);
  assert.equal(again.step(250).view.state, "react");
  const end = run(again, 250, 5000);
  assert.equal(end.view.state, "sit", "and is not forced back to sleep");
  const grabbed = mk();
  grabbed.step(0);
  grabbed.sleepNow(100);
  grabbed.grab(200);
  grabbed.drop({ x: 100, y: 696 }, 250);
  assert.notEqual(grabbed.step(300).view.state, "sleep");

  const walkSleeper = walking();
  walkSleeper.pet.sleepNow(walkSleeper.t + 10);
  const f = walkSleeper.pet.step(walkSleeper.t + 50);
  assert.equal(f.view.state, "sleep");
  assert.equal(f.save, true);
  assert.equal(walkSleeper.pet.ticking(), false);
});

test("ticks only while walking or falling, and idle wakes are at least 1 s apart", () => {
  const rng = makeRng(99);
  const pet = mk({ rng: makeRng(5), nowMs: BASE, ctx: { presence: "focused", idleS: 60 } });
  const presences = ["focused", "break", "talking", "gaming", "away", null];
  const kinds = ACTIVITY_KINDS;
  let t = BASE;
  let nextEvent = BASE + 20000;
  let walks = 0;
  let falls = 0;
  const END = BASE + 3600 * 1000;
  pet.step(t);
  while (t < END) {
    const wake = pet.nextWakeMs(t);
    assert.ok(wake !== null && wake > t - 1, "a wake is never in the past");
    t = Math.min(wake, nextEvent);
    if (t >= nextEvent) {
      const r = rng();
      if (r < 0.4) pet.setContext({ presence: presences[Math.floor(rng() * presences.length)], idleS: Math.floor(rng() * 700) }, t);
      else if (r < 0.7) pet.react(kinds[Math.floor(rng() * kinds.length)], "label", t);
      else if (r < 0.8) { pet.grab(t); pet.drop({ x: Math.floor(rng() * 1600), y: Math.floor(rng() * 696) }, t); }
      else pet.setContext({ presence: "focused", idleS: 60 + Math.floor(rng() * 100) }, t);
      nextEvent = t + 5000 + Math.floor(rng() * 90000);
    }
    const frame = pet.step(t);
    const state = frame.view.state;
    if (state === "walk") walks += 1;
    if (state === "fall") falls += 1;
    const ticking = pet.ticking();
    if (ticking) {
      assert.ok(state === "walk" || state === "fall", `ticking in ${state}`);
      assert.equal(pet.nextWakeMs(t), t + PET.MOVE_MS);
    } else {
      assert.notEqual(state, "walk");
    }
    const inProgress = state === "react" || state === "fall" || state === "drag" || frame.view.motion === "fidget"
      || frame.view.bubble !== null || pet.awakeUntil > t || pet.queue.length > 0;
    if (!ticking) {
      const gap = pet.nextWakeMs(t) - t;
      assert.ok(gap >= 1000 || inProgress, `an idle wake ${gap} ms away in ${state}`);
    }
  }
  assert.ok(walks > 20 && falls > 3, `the mix really walked and fell (${walks} walk steps, ${falls} fall steps)`);
  assert.equal(mk({ ctx: { enabled: false } }).nextWakeMs(0), null);
});

test("a 10 s walk is 200 steps at MOVE_MS", () => {
  const pet = mk({ rng: () => 0.9999999, ctx: { presence: null, idleS: 0, forceWalk: true } });
  let t = 0;
  let steps = 0;
  let started = false;
  for (let i = 0; i < 1000; i += 1) {
    const frame = pet.step(t);
    if (frame.view.state === "walk") { started = true; steps += 1; } else if (started) break;
    t = pet.nextWakeMs(t);
  }
  assert.equal(steps, 200);
});

test("a long pause is one clamped step, never a leap", () => {
  const { pet, t } = walking();
  const before = pet.step(t + 50).x;
  const after = pet.step(t + 50 + 60000).x; // the laptop slept for a minute
  assert.ok(Math.abs(after - before) <= 90 * (PET.MAX_DT_MS / 1000) + 1, `leapt ${after - before}`);
});

test("it never lets a NaN, an infinity or a tiny work area escape", () => {
  const areas = [
    { x: 0, y: 0, width: 0, height: 0 },
    { x: 5, y: 7, width: 100, height: 50 },
    { x: NaN, y: Infinity, width: -4, height: NaN },
    null,
  ];
  for (const area of areas) {
    const pet = mk({ area, ctx: { presence: null, idleS: 0, forceWalk: true } });
    let t = 0;
    const check = (frame) => {
      assert.ok(Number.isInteger(frame.x) && Number.isInteger(frame.y), `${JSON.stringify(area)} gave ${frame.x},${frame.y}`);
      assert.ok(Number.isFinite(frame.view.motionAt));
    };
    for (let i = 0; i < 2000; i += 1) { t += 50; check(pet.step(t)); }
    check(pet.step(NaN));
    check(pet.step(-1000));
    pet.setPos({ x: NaN, y: Infinity });
    pet.grab(t);
    pet.drop({ x: NaN, y: -Infinity }, t);
    pet.setSize({ width: NaN, height: -1 }, t);
    pet.setArea({ x: 0, y: 0, width: 0, height: 0 }, t);
    pet.setContext({ idleS: NaN, presence: "focused" }, t);
    pet.react("done", 12345, t);
    pet.react("edit", null, t);
    for (let i = 0; i < 400; i += 1) { t += 50; check(pet.step(t)); }
    assert.ok(Number.isFinite(pet.nextWakeMs(t)));
  }
});

test("bubbleText clips to 40", () => {
  assert.equal(bubbleText("a".repeat(40)), "a".repeat(40));
  const clipped = bubbleText("a".repeat(41));
  assert.equal(clipped.length, 40);
  assert.ok(clipped.endsWith("…"));
  assert.equal(clipped, "a".repeat(39) + "…");
  const emoji = "\u{1F600}"; // one code point, two UTF-16 units
  const long = "a".repeat(38) + emoji + "bc";
  assert.equal(Array.from(long).length, 41);
  assert.equal(bubbleText(long), "a".repeat(38) + emoji + "…", "the pair is kept whole at the cut");
  const edge = "a".repeat(39) + emoji + "b";
  assert.equal(bubbleText(edge), "a".repeat(39) + "…", "and dropped whole when it is the cut");
  assert.equal(bubbleText(emoji.repeat(40)), emoji.repeat(40), "40 code points fit");
  assert.equal(bubbleText(emoji.repeat(41)), emoji.repeat(39) + "…");
  assert.doesNotMatch(bubbleText(edge), /[\ud800-\udfff]/);
  assert.equal(bubbleText(null), "");
  assert.equal(bubbleText(12), "");
  assert.equal(bubbleText({}), "");
});

test("a bubble lasts BUBBLE_MS and then clears, and the latest one wins", () => {
  const pet = mk();
  pet.step(0);
  pet.react("edit", "editing a.py", 100);
  pet.react("run", "running tests", 200);
  assert.equal(pet.step(250).view.bubble.text, "running tests");
  const gone = pet.step(200 + PET.BUBBLE_MS);
  assert.equal(gone.view.bubble, null);
  assert.equal(gone.viewChanged, true);
  assert.equal(pet.step(200 + PET.BUBBLE_MS + 100).viewChanged, false, "an unchanged view is not a change");
});

test("renderer helpers", () => {
  const walk = { on: true, state: "walk", facing: -1, expression: null, motion: "walk", effect: null, effectSeq: 0,
    bubble: null, reduced: false, motionAt: 1000 };
  assert.ok(petTransformCss(walk, 1100).includes("scale(-1"));
  assert.notEqual(petTransformCss(walk, 1100), petTransformCss(walk, 1230));
  assert.match(petTransformCss({ ...walk, facing: 1 }, 1100), /^translate\(-?[\d.]+px, -?[\d.]+px\) rotate\(-?[\d.]+deg\) scale\(1, 1\)$/);
  const calm = { ...walk, reduced: true, motion: "wave" };
  assert.equal(petTransformCss(calm, 1000), petTransformCss(calm, 1777));
  assert.ok(petTransformCss(calm, 0).includes("scale(-1"), "facing still applies");
  const sit = { ...walk, facing: 1, motion: "sit" };
  assert.equal(petTransformCss(sit, 1000), "translate(0px, 4px) rotate(0deg) scale(1.02, 0.95)");
  assert.equal(petTransformCss({ ...sit, motion: "sleep" }, 1000), "translate(0px, 6px) rotate(0deg) scale(1, 0.97)");
  assert.equal(petTransformCss({ ...sit, motion: "squash" }, 1000), "translate(0px, 0px) rotate(0deg) scale(1.06, 0.9)");
  assert.equal(petTransformCss({ ...sit, reduced: true }, 1000), "translate(0px, 4px) rotate(0deg) scale(1.02, 0.95)");
  assert.equal(petTransformCss({ ...sit, motion: "wave" }, 1150), "translate(0px, 0px) rotate(5deg) scale(1, 1)");
  assert.equal(petTransformCss({ ...sit, motion: "spin" }, 1000 + 400), "translate(0px, 0px) rotate(0deg) scale(-1, 1)");
  assert.equal(petTransformCss({ ...sit, motion: "spin" }, 1000 + 5000), "translate(0px, 0px) rotate(0deg) scale(1, 1)");
  assert.equal(petTransformCss(OFF_VIEW, 5), "translate(0px, 0px) rotate(0deg) scale(1, 1)");

  assert.equal(walkFrame(["a", "b"], 0), "a");
  assert.equal(walkFrame(["a", "b"], 125), "b");
  assert.equal(walkFrame(["a", "b"], 250), "a");
  assert.equal(walkFrame([], 0), null);
  assert.equal(walkFrame(null, 0), null);
  assert.deepEqual(mirrorForFacing({ x: 10, y: 5 }, 100, -1), { x: 90, y: 5 });
  assert.deepEqual(mirrorForFacing({ x: 10, y: 5 }, 100, 1), { x: 10, y: 5 });
});

test("makeRng is a seeded mulberry32 in [0, 1)", () => {
  const a = makeRng(42);
  const b = makeRng(42);
  const seq = Array.from({ length: 1000 }, () => a());
  assert.deepEqual(seq, Array.from({ length: 1000 }, () => b()));
  assert.ok(seq.every((v) => v >= 0 && v < 1));
  assert.notDeepEqual(seq.slice(0, 5), Array.from({ length: 5 }, makeRng(43)));
  assert.equal(makeRng(-1)(), makeRng(4294967295)(), "the seed is taken as a uint32");
});

test("the module loads as a plain script", () => {
  const g = {};
  const source = fs.readFileSync(path.join(__dirname, "..", "src", "pet-logic.js"), "utf8");
  vm.runInNewContext(source, { globalThis: g });
  assert.equal(typeof g.RiyaPet.Pet, "function");
  assert.equal(typeof g.RiyaPet.petTransformCss, "function");
});

test("the file needs no DOM, no Electron and no real clock or randomness", () => {
  const source = fs.readFileSync(path.join(__dirname, "..", "src", "pet-logic.js"), "utf8");
  const code = source.replace(/\/\*[\s\S]*?\*\//g, "").replace(/^\s*\/\/.*$/gm, "").replace(/\s\/\/ .*$/gm, "");
  assert.ok(!/Math\.random|Date\.now|performance\.now|setTimeout|setInterval/.test(code));
  assert.ok(!/\bdocument\b|\bwindow\b|\brequire\s*\(|\belectron\b/i.test(code), "no DOM, no Electron, no require");
  assert.ok(/typeof module !== "undefined"/.test(code) && /globalThis\.RiyaPet/.test(code), "only the dual-load guard");
});
