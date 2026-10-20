"use strict";
/** Little Riya's tiny effects: the particles of a confetti burst, a sparkle, a sweat drop and a heart, as pure seeded
 *  kinematics. No DOM and no clock: the renderer asks where each particle is at `tMs` and draws it on one small canvas.
 *  Loads in Node (tests) and as a plain script in the renderer (after pet-logic.js). */

// renamed on import: pet-logic.js is a classic script too, and its top-level names are taken in the renderer
const { makeRng: seededRng, PET: PET_ROOMS } = typeof require === "function" ? require("./pet-logic") : globalThis.RiyaPet;

const EFFECT_HZ = 30;
const EFFECTS = Object.freeze({
  confetti: Object.freeze({ ms: 1500, count: 24 }),
  sparkle: Object.freeze({ ms: 900, count: 8 }),
  sweat: Object.freeze({ ms: 1200, count: 1 }),
  heart: Object.freeze({ ms: 1200, count: 1 }),
});
const COLORS = Object.freeze(["#ff8fb1", "#ffd66b", "#8fd3ff", "#b9f18f", "#c9a7ff"]);

const CONFETTI_GRAVITY = 360; // px/s^2
const CONFETTI_DRAG = 2; // 1/s, horizontal air drag: a burst spreads about 70 px and flutters down
const HEAD_ROOM = PET_ROOMS.EFFECTS_ROOM + PET_ROOMS.BUBBLE_ROOM; // the sprite's top edge in the window
const HEAD_AT = 0.12; // her face centre, as a share of the sprite's height below its top edge

const num = (v, fallback) => (Number.isFinite(v) ? v : fallback);
const bound = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
const between = (rng, lo, hi) => lo + (hi - lo) * rng();
const pick = (rng) => COLORS[Math.floor(rng() * COLORS.length) % COLORS.length];

/** A particle: where it starts, how it moves, and how big and what colour it is. Positions are clamped to the window. */
function particle(base) {
  return { vx: 0, vy: 0, g: 0, drag: 0, ...base };
}

/** The particles of `kind` for a window of `width` x `height` DIP, from `seed`. An unknown kind has none. */
function particles(kind, seed, width, height) {
  const spec = Object.hasOwn(EFFECTS, kind) ? EFFECTS[kind] : null;
  if (!spec) return [];
  const w = Math.max(0, num(width, 0));
  const h = Math.max(0, num(height, 0));
  const rng = seededRng(num(seed, 0));
  const spriteH = Math.max(0, h - HEAD_ROOM);
  const cx = w / 2;
  const headY = HEAD_ROOM + HEAD_AT * spriteH;
  const out = [];
  for (let i = 0; i < spec.count; i++) {
    const common = { w, h, ms: spec.ms };
    if (kind === "confetti") {
      const angle = -Math.PI * between(rng, 25 / 180, 155 / 180); // up and out
      const speed = between(rng, 90, 200);
      out.push(particle({
        ...common, x0: cx, y0: HEAD_ROOM, vx: Math.cos(angle) * speed, vy: Math.sin(angle) * speed,
        g: CONFETTI_GRAVITY, drag: CONFETTI_DRAG, r: between(rng, 2.5, 4.5), color: pick(rng),
      }));
    } else if (kind === "sparkle") {
      const angle = (i / spec.count) * 2 * Math.PI + between(rng, -0.2, 0.2);
      const ring = between(rng, 30, 50);
      out.push(particle({
        ...common, x0: cx + Math.cos(angle) * ring, y0: headY + Math.sin(angle) * ring * 0.8,
        vx: Math.cos(angle) * 12, vy: Math.sin(angle) * 12 - 10, r: between(rng, 2, 4), color: pick(rng),
      }));
    } else if (kind === "sweat") {
      out.push(particle({
        ...common, x0: cx + HEAD_AT * spriteH + between(rng, -3, 3), y0: headY, vy: 25, g: 40, r: 4, color: COLORS[2],
      }));
    } else {
      out.push(particle({
        ...common, x0: cx + between(rng, -4, 4), y0: headY - 0.1 * spriteH, vy: -40, r: 7, color: COLORS[0],
      }));
    }
  }
  return out;
}

/** Where each particle is at `tMs` since the effect began. Alpha holds for half its life, then fades to 0 at `ms`
 *  (and is 0 from then on, and before the start). */
function particlesAt(list, tMs) {
  if (!Array.isArray(list)) return [];
  const t = num(tMs, 0);
  const s = Math.max(0, t) / 1000;
  return list.map((p) => {
    const travel = p.drag > 0 ? (1 - Math.exp(-p.drag * s)) / p.drag : s;
    const alive = t >= 0 && t < p.ms;
    return {
      x: bound(p.x0 + p.vx * travel, 0, p.w),
      y: bound(p.y0 + p.vy * s + 0.5 * p.g * s * s, 0, p.h),
      r: p.r,
      color: p.color,
      alpha: alive ? Math.min(1, (p.ms - t) / (p.ms / 2)) : 0,
    };
  });
}

/** True once the effect has run its time, and for a kind that has none. */
function effectDone(kind, tMs) {
  if (!Object.hasOwn(EFFECTS, kind)) return true;
  return !(num(tMs, 0) < EFFECTS[kind].ms);
}

const effectsExports = { EFFECT_HZ, EFFECTS, COLORS, particles, particlesAt, effectDone };
if (typeof module !== "undefined" && module.exports) module.exports = effectsExports;
else globalThis.RiyaEffects = effectsExports;
