"use strict";
/** Little Riya's pure state machine: walks, sits, sleeps, drag and fall, and reactions to agent activity. No DOM and no
 *  Electron: the clock and the random source are injected, so it is deterministic. Loads in Node (tests) and as a plain
 *  script in the renderer. A Python test (tests/core/test_pet_body_sync.py) keeps the lists below equal to the core's. */

const ACTIVITY_KINDS = ["edit", "run", "read", "search", "web", "delegate", "think", "wave", "done", "fail", "oops", "sparkle"];
const PET_WALKS_WHEN = ["idle", "always_but_focus"];
const REDUCED_MOTION = ["system", "on", "off"];
const SCALE_RANGE = [0.4, 1.0];
const BUBBLE_MAX = 40;

const PET = Object.freeze({
  WALK_BURST_MS: [3000, 10000], WALK_GAP_MS: [40000, 180000], WALK_SPEED: [60, 90], MOVE_MS: 50,
  WALK_IDLE_S: 30, FOCUS_IDLE_S: 5, SLEEP_IDLE_S: 600, MAX_DT_MS: 250, GRAVITY: 2400, LAND_MS: 180,
  SIT_GAP_MS: [30000, 120000], FIDGET_GAP_MS: [15000, 45000], FIDGET_MS: 800, SIT_AFTER_WALK: 0.5,
  SURPRISE_ONE_IN: 50, BUBBLE_MS: 4000, QUEUE_MAX: 3, SCALE_DEFAULT: 0.6, SPRITE_REF_H: 400,
  BUBBLE_W: 240, BUBBLE_ROOM: 80, EFFECTS_ROOM: 24,
});

const STATES = ["idle", "walk", "sit", "sleep", "drag", "fall", "react"];

const REACTIONS = {
  edit: { expression: null, motion: null, effect: null, ms: 0, bubbleOnly: true },
  run: { expression: null, motion: null, effect: null, ms: 0, bubbleOnly: true },
  read: { expression: null, motion: null, effect: null, ms: 0, bubbleOnly: true },
  search: { expression: null, motion: null, effect: null, ms: 0, bubbleOnly: true },
  web: { expression: null, motion: null, effect: null, ms: 0, bubbleOnly: true },
  delegate: { expression: null, motion: null, effect: null, ms: 0, bubbleOnly: true },
  think: { expression: "thinking", motion: null, effect: null, ms: 3000, bubbleOnly: false },
  wave: { expression: "excited", motion: "wave", effect: null, ms: 6000, bubbleOnly: false },
  done: { expression: "happy", motion: null, effect: "confetti", ms: 3000, bubbleOnly: false },
  fail: { expression: "sad", motion: null, effect: null, ms: 4000, bubbleOnly: false },
  oops: { expression: "worried", motion: null, effect: "sweat", ms: 2500, bubbleOnly: false },
  sparkle: { expression: "happy", motion: null, effect: "sparkle", ms: 1500, bubbleOnly: false },
};
for (const row of Object.values(REACTIONS)) Object.freeze(row);
Object.freeze(REACTIONS);

const POKE = Object.freeze({ expression: "happy", motion: null, effect: null, ms: 1200 });
const BIG_EFFECTS = ["confetti", "heart"];
const BIG_MOTIONS = ["spin"];

/** The neutral view main sends when the pet stops, so a renderer never holds undefined fields. */
const OFF_VIEW = Object.freeze({
  on: false, state: "idle", facing: 1, expression: null, motion: null, effect: null, effectSeq: 0, bubble: null,
  reduced: false, motionAt: 0,
});

const DEFAULT_CTX = Object.freeze({
  enabled: true, presence: null, idleS: 0, quietHours: null, reducedMotion: false, walksWhen: "idle", forceWalk: false,
});
const DEFAULT_ASPECT = 384 / 672;
const START_INSET = 24; // a fresh pet sits this far in from the right edge
const IDLE_WAKE_MS = 1000; // an unhurried deadline waits at most this long for the wake it shares
const POLL_MS = 30000; // the longest she sleeps between looks, so a quiet-hours edge or idle time is never missed
const WALKERS = ["focused", "break", "talking"];
const QUIET_RE = /^([01]\d|2[0-3]):([0-5]\d)-([01]\d|2[0-3]):([0-5]\d)$/;

const finite = (v, fallback) => (Number.isFinite(v) ? v : fallback);
const unit = (r) => (Number.isFinite(r) ? Math.min(1, Math.max(0, r)) : 0.5);
const lerp = (range, r) => range[0] + (range[1] - range[0]) * unit(r);
const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));

/** mulberry32 over `seed >>> 0`: values in [0, 1). */
function makeRng(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function clampScale(v) {
  if (!Number.isFinite(v)) return PET.SCALE_DEFAULT;
  return clamp(v, SCALE_RANGE[0], SCALE_RANGE[1]);
}

/** The window's size: a fixed width and height, from the sprite's aspect and pet_scale. */
function petSize({ canvas, scale }) {
  const ok = canvas && Number.isFinite(canvas.width) && Number.isFinite(canvas.height) && canvas.width > 0 && canvas.height > 0;
  const aspect = ok ? canvas.width / canvas.height : DEFAULT_ASPECT;
  const spriteH = Math.round(PET.SPRITE_REF_H * clampScale(scale));
  const spriteW = Math.round(spriteH * aspect);
  return { width: Math.max(spriteW, PET.BUBBLE_W), height: spriteH + PET.BUBBLE_ROOM + PET.EFFECTS_ROOM, spriteW, spriteH };
}

const floorY = (area, height) => area.y + area.height - height;

function walkBounds(area, width) {
  return { minX: area.x, maxX: Math.max(area.x, area.x + area.width - width) };
}

/** Mirrors settings.in_quiet_hours on local hours and minutes. Equal ends and a bad spec are false. */
function inQuietHours(spec, date) {
  const m = typeof spec === "string" ? QUIET_RE.exec(spec) : null;
  if (!m || !(date instanceof Date)) return false;
  const [h1, m1, h2, m2] = m.slice(1).map(Number);
  const start = h1 * 60 + m1;
  const end = h2 * 60 + m2;
  const now = date.getHours() * 60 + date.getMinutes();
  if (start === end || Number.isNaN(now)) return false;
  return start < end ? now >= start && now < end : now >= start || now < end;
}

function reducedMotionOn(setting, osReduced) {
  if (setting === "on") return true;
  if (setting === "off") return false;
  return osReduced === true;
}

/** What she may do right now. `forceWalk` skips only the idle threshold and the unknown-presence check. */
function gates(ctx, { quiet, sinceAgentS }) {
  const enabled = ctx.enabled === true;
  const presence = ctx.presence === undefined ? null : ctx.presence;
  const idleS = finite(ctx.idleS, 0);
  const reduced = ctx.reducedMotion === true;
  const show = enabled && presence !== "gaming" && presence !== "away";
  const sleep = enabled && (presence === "away" || quiet === true
    || (idleS >= PET.SLEEP_IDLE_S && sinceAgentS >= PET.SLEEP_IDLE_S));
  const big = show && presence !== null && !quiet && !reduced && idleS >= PET.WALK_IDLE_S;
  const threshold = ctx.walksWhen === "always_but_focus" ? PET.FOCUS_IDLE_S : PET.WALK_IDLE_S;
  const walk = enabled && !reduced && !sleep && presence !== "gaming"
    && (ctx.forceWalk === true || (WALKERS.includes(presence) && idleS >= threshold));
  return { walk, big, show, sleep };
}

function bubbleText(label) {
  if (typeof label !== "string") return "";
  const chars = Array.from(label); // code points, so a pair of surrogates is never split
  return chars.length > BUBBLE_MAX ? chars.slice(0, BUBBLE_MAX - 1).join("") + "…" : label;
}

function cleanArea(area) {
  const a = area && typeof area === "object" ? area : {};
  return {
    x: finite(a.x, 0), y: finite(a.y, 0),
    width: Math.max(0, finite(a.width, 0)), height: Math.max(0, finite(a.height, 0)),
  };
}

function cleanSize(size) {
  const s = size && typeof size === "object" ? size : {};
  return { width: Math.max(0, finite(s.width, 0)), height: Math.max(0, finite(s.height, 0)) };
}

function viewsEqual(a, b) {
  for (const key of Object.keys(OFF_VIEW)) {
    if (key === "bubble") continue;
    if (a[key] !== b[key]) return false;
  }
  if (a.bubble === null || b.bubble === null) return a.bubble === b.bubble;
  return a.bubble.text === b.bubble.text && a.bubble.untilMs === b.bubble.untilMs;
}

class Pet {
  constructor({ rng, nowMs, area, size, pos = null, ctx = {} }) {
    this.rng = typeof rng === "function" ? rng : makeRng(1);
    this.area = cleanArea(area);
    this.size = cleanSize(size);
    this.ctx = { ...DEFAULT_CTX, ...ctx };
    const now = finite(nowMs, 0);
    this.last = now;
    const { minX, maxX } = walkBounds(this.area, this.size.width);
    const floor = floorY(this.area, this.size.height);
    this.x = clamp(pos ? finite(pos.x, maxX - START_INSET) : maxX - START_INSET, minX, maxX);
    this.y = Math.min(pos ? finite(pos.y, floor) : floor, floor);
    this.lx = Math.round(this.x); // the position the last frame reported
    this.ly = Math.round(this.y);
    this.facing = 1;
    this.state = "idle";
    this.expression = null;
    this.motion = null;
    this.motionAt = now;
    this.effect = null;
    this.effectSeq = 0;
    this.bubble = null;
    this.queue = [];
    this.reactKind = null;
    this.reactUntil = null;
    this.agentAt = -Infinity; // no agent event yet: "no event for 600 s" is already true
    this.awakeUntil = 0; // a reaction keeps her up (out of a quiet-hours sleep) until it has shown
    this.sleepForced = false;
    this.saveNext = false;
    this.vy = 0;
    this.landUntil = null;
    this.walkDir = 1;
    this.walkSpeed = 0;
    this.walkEndAt = null;
    this.fidgetUntil = null;
    this.disabled = false;
    this.lastView = OFF_VIEW;
    this._enterIdle(now);
    this._drawWalk(now);
  }

  // ---- inputs ----

  setContext(patch, nowMs) {
    this.ctx = { ...this.ctx, ...(patch && typeof patch === "object" ? patch : {}) };
  }

  setArea(area, nowMs) {
    this.area = cleanArea(area);
    if (this.state !== "drag") this._clampPos();
  }

  setSize(size, nowMs) {
    this.size = cleanSize(size);
    if (this.state !== "drag") this._clampPos();
  }

  setPos({ x, y } = {}) {
    this.x = finite(x, this.x);
    this.y = finite(y, this.y);
    if (this.state !== "drag") this._clampPos();
  }

  grab(nowMs) {
    const now = this._now(nowMs);
    this.last = now;
    this.sleepForced = false;
    this.awakeUntil = 0;
    this._cancelMotion();
    this._set("drag", this.rng() < 0.5 ? "flustered" : "excited", "dangle", now);
  }

  /** Returns "fall" when she is let go above the floor, else "idle" (she stands), with `save` on the next frame. */
  drop(pos, nowMs, { save = true } = {}) {
    const now = this._now(nowMs);
    this.last = now;
    const p = pos && typeof pos === "object" ? pos : {};
    const { minX, maxX } = walkBounds(this.area, this.size.width);
    const floor = floorY(this.area, this.size.height);
    this.x = clamp(finite(p.x, this.x), minX, maxX);
    this.y = Math.min(floor, Math.max(this.area.y, finite(p.y, this.y)));
    this._cancelMotion();
    if (this.y < floor - 1 && this.ctx.reducedMotion !== true) {
      this._startFall(now);
      return "fall";
    }
    this._enterIdle(now);
    this._drawWalk(now);
    this.saveNext = save === true;
    return "idle";
  }

  react(kind, label, nowMs) {
    if (!ACTIVITY_KINDS.includes(kind)) return;
    const now = this._now(nowMs);
    if (!this._gates(now).show) return;
    this.agentAt = now;
    const row = REACTIONS[kind];
    if (row.bubbleOnly) {
      if (this.state === "react" && this.reactKind === "wave") return; // "needs you!" stays; this would be stale by then
      this._showBubble(label, now, PET.BUBBLE_MS);
      if (this.state === "sleep" && !this.sleepForced) {
        this.awakeUntil = Math.max(this.awakeUntil, now + PET.BUBBLE_MS);
        this._enterSit(now);
        this._drawWalk(now);
      }
      return;
    }
    this._enqueue(kind, label);
    if (kind === "wave") this._showBubble(label, now, Math.max(row.ms, PET.BUBBLE_MS)); // never held behind a drag
  }

  poke(nowMs) {
    if (!this.ctx.enabled) return;
    this._now(nowMs);
    this._enqueue("poke", "");
  }

  sleepNow(nowMs) {
    if (!this.ctx.enabled) return;
    const now = this._now(nowMs);
    this.sleepForced = true;
    this.queue = [];
    if (this.state === "drag" || this.state === "fall") return; // she sleeps once she is standing again
    if (this.state === "walk") this._endWalk(now);
    this.reactUntil = null;
    this.reactKind = null;
    this._enterSleep(now);
  }

  // ---- the clock ----

  step(nowMs) {
    const now = this._now(nowMs);
    const dt = Math.min(Math.max(0, now - this.last), PET.MAX_DT_MS) / 1000;
    this.last = now;
    if (!this.ctx.enabled) return this._disabledFrame(now);
    if (this.disabled) {
      this.disabled = false;
      this._enterIdle(now);
      this._drawWalk(now);
    }
    const g = this._gates(now);
    if (this.bubble && now >= this.bubble.untilMs) this.bubble = null;
    if (this.state === "drag") return this._frame(true);
    if (!g.show) this._hide(now);

    const floor = floorY(this.area, this.size.height);
    if (this.state !== "fall" && this.y < floor - 1 && this.ctx.reducedMotion !== true) this._startFall(now);
    if (this.state === "fall" && this.ctx.reducedMotion === true) this._snapToFloor(now, floor);
    if (this.state === "fall") this._stepFall(now, dt, floor);

    if (this.state === "react") {
      const waveWaits = this.reactKind !== "wave" && this.queue.some((e) => e.kind === "wave");
      if (now >= this.reactUntil || waveWaits) this._endReaction(now, g);
    } else if (this.queue.length && this.state !== "fall") {
      this._startReaction(now, g);
    }

    const wantSleep = this.sleepForced || (g.sleep && (this.ctx.presence === "away" || now >= this.awakeUntil));
    const gaming = this.ctx.presence === "gaming";
    if (this.state === "walk") this._stepWalk(now, dt, g, wantSleep, floor);
    else if (gaming && (this.state === "idle" || this.state === "sit" || this.state === "sleep")) this._freeze(now);
    else if (this.state === "sleep") {
      if (!wantSleep) {
        this._enterSit(now);
        this._drawWalk(now);
      }
    } else if (this.state === "idle" || this.state === "sit") {
      if (wantSleep) this._enterSleep(now);
      else if (g.show) this._stepResting(now, g);
    }
    return this._frame(false);
  }

  /** True only while the window must move at MOVE_MS: walking, or falling before she lands. */
  ticking() {
    return this.ctx.enabled === true && (this.state === "walk" || (this.state === "fall" && this.landUntil === null));
  }

  /** When main should step next: MOVE_MS while ticking, else the earliest pending deadline. Null when disabled.
   *  An unhurried deadline (a walk, a sit toggle, a fidget, the poll) is taken up to IDLE_WAKE_MS late, so resting wakes
   *  are at least that far apart; a reaction, a landing, a bubble or a fidget in progress is on time. */
  nextWakeMs(nowMs) {
    const now = finite(nowMs, this.last);
    if (!this.ctx.enabled) return null;
    if (this.ticking()) return now + PET.MOVE_MS;
    const exact = [];
    const lazy = [now + POLL_MS];
    const add = (list, t) => { if (Number.isFinite(t)) list.push(Math.max(t, now)); };
    if (this.bubble) add(exact, this.bubble.untilMs);
    if (this.awakeUntil > now) add(exact, this.awakeUntil);
    if ((this.state === "idle" || this.state === "sit" || this.state === "sleep") && this.ctx.presence === "gaming") {
      // frozen while gaming: no toggle, fidget or walk deadline, only the poll
    } else if (this.state === "idle" || this.state === "sit") {
      add(lazy, this.nextWalkAt);
      add(lazy, this.toggleAt);
      if (this.state === "idle" && this.ctx.reducedMotion !== true) {
        if (this.fidgetUntil === null) add(lazy, this.fidgetAt);
        else add(exact, this.fidgetUntil);
      }
    } else if (this.state === "react") add(exact, this.reactUntil);
    else if (this.state === "fall") add(exact, this.landUntil);
    const pending = this.queue.length > 0 && this.state !== "drag" && this.state !== "fall"
      && (this.state !== "react" || (this.reactKind !== "wave" && this.queue.some((e) => e.kind === "wave")));
    if (pending) add(exact, now);
    return Math.min(...exact, Math.max(Math.min(...lazy), now + IDLE_WAKE_MS));
  }

  // ---- internals ----

  _now(nowMs) {
    return finite(nowMs, this.last);
  }

  _gates(now) {
    const quiet = inQuietHours(this.ctx.quietHours, new Date(now));
    return gates(this.ctx, { quiet, sinceAgentS: (now - this.agentAt) / 1000 });
  }

  _clampPos() {
    const { minX, maxX } = walkBounds(this.area, this.size.width);
    this.x = clamp(finite(this.x, minX), minX, maxX);
    this.y = Math.min(finite(this.y, floorY(this.area, this.size.height)), floorY(this.area, this.size.height));
  }

  _set(state, expression, motion, now) {
    if (state !== this.state || motion !== this.motion) this.motionAt = now;
    this.state = state;
    this.expression = expression;
    this.motion = motion;
    if (state !== "react") this.effect = null;
    if (state !== "idle") this.fidgetUntil = null;
  }

  _drawWalk(now) {
    this.nextWalkAt = this.ctx.forceWalk === true ? now : now + lerp(PET.WALK_GAP_MS, this.rng());
  }

  _enterIdle(now) {
    this._set("idle", null, null, now);
    this.toggleAt = now + lerp(PET.SIT_GAP_MS, this.rng());
    this.fidgetAt = now + lerp(PET.FIDGET_GAP_MS, this.rng());
    this.fidgetUntil = null;
  }

  _enterSit(now) {
    this._set("sit", null, "sit", now);
    this.toggleAt = now + lerp(PET.SIT_GAP_MS, this.rng());
  }

  _enterSleep(now) {
    this._set("sleep", "sleepy", "sleep", now);
  }

  /** Walk, fall or reaction in progress is dropped (a grab, a drop). The queue stays. */
  _cancelMotion() {
    this.walkEndAt = null;
    this.reactUntil = null;
    this.reactKind = null;
    this.landUntil = null;
    this.vy = 0;
    this.effect = null;
  }

  _showBubble(label, now, ms) {
    const text = bubbleText(label);
    if (text) this.bubble = { text, untilMs: now + ms };
  }

  _enqueue(kind, label) {
    const entry = { kind, label };
    const same = this.queue.findIndex((e) => e.kind === kind);
    if (same >= 0) {
      this.queue[same] = entry;
      return;
    }
    if (this.queue.length >= PET.QUEUE_MAX) {
      const drop = this.queue.findIndex((e) => e.kind !== "wave");
      this.queue.splice(drop >= 0 ? drop : 0, 1); // the oldest, but a wave outlives a pose
    }
    this.queue.push(entry);
  }

  _startFall(now) {
    this._cancelMotion();
    this._set("fall", "surprised", null, now);
  }

  _stepFall(now, dt, floor) {
    if (this.landUntil === null) {
      this.vy += PET.GRAVITY * dt;
      this.y += this.vy * dt;
      if (this.y < floor) return;
      this.y = floor;
      this.vy = 0;
      this.landUntil = now + PET.LAND_MS;
      this._set("fall", null, "squash", now);
    } else if (now >= this.landUntil) {
      this.landUntil = null;
      this._enterIdle(now);
      this._drawWalk(now);
      this.saveNext = true;
    }
  }

  /** Gaming or away: nothing shows. A running reaction ends at once (away then sleeps in the same step), the queue and
   *  the bubble are dropped. */
  _hide(now) {
    this.queue = [];
    this.bubble = null;
    if (this.state !== "react") return;
    this.reactUntil = null;
    this.reactKind = null;
    this.effect = null;
    this._enterSit(now);
    this._drawWalk(now);
  }

  /** Reduced motion came on mid-fall: she is on the floor, landed and standing, with no squash. */
  _snapToFloor(now, floor) {
    this.y = floor;
    this.vy = 0;
    this.landUntil = null;
    this._enterIdle(now);
    this._drawWalk(now);
    this.saveNext = true;
  }

  _startReaction(now, g) {
    const at = this.queue.findIndex((e) => e.kind === "wave");
    const [entry] = this.queue.splice(at < 0 ? 0 : at, 1);
    if (this.state === "walk") this._endWalk(now);
    const row = entry.kind === "poke" ? POKE : REACTIONS[entry.kind];
    let { motion, effect } = row;
    if (BIG_EFFECTS.includes(effect) && !g.big) effect = null;
    if (entry.kind === "done" && g.big && this.rng() < 1 / PET.SURPRISE_ONE_IN) {
      if (this.rng() < 0.5) effect = "heart";
      else motion = BIG_MOTIONS[0];
    }
    if (this.ctx.reducedMotion === true) {
      motion = null;
      effect = null;
    }
    this.sleepForced = false;
    this._set("react", row.expression, motion, now);
    this.motionAt = now;
    this.effect = effect;
    if (effect) this.effectSeq += 1;
    this.reactKind = entry.kind;
    this.reactUntil = now + row.ms;
    let until = this.reactUntil;
    if (bubbleText(entry.label)) {
      this._showBubble(entry.label, now, Math.max(row.ms, PET.BUBBLE_MS));
      until = Math.max(until, this.bubble.untilMs);
    }
    this.awakeUntil = Math.max(this.awakeUntil, until);
  }

  _endReaction(now, g) {
    this.reactUntil = null;
    this.reactKind = null;
    this.effect = null;
    if (this.queue.length) {
      this._startReaction(now, g);
      return;
    }
    this._enterSit(now);
    this._drawWalk(now);
  }

  _endWalk(now) {
    this.walkEndAt = null;
    if (this.rng() < PET.SIT_AFTER_WALK) this._enterSit(now);
    else this._enterIdle(now);
    this._drawWalk(now);
    this.saveNext = true;
  }

  _stepWalk(now, dt, g, wantSleep, floor) {
    const { minX, maxX } = walkBounds(this.area, this.size.width);
    if (!g.walk || wantSleep || maxX <= minX || now >= this.walkEndAt) {
      this._endWalk(now);
      if (wantSleep) this._enterSleep(now);
      return;
    }
    this.x += this.walkDir * this.walkSpeed * dt;
    if (this.x >= maxX) {
      this.x = maxX;
      this.walkDir = -1;
    } else if (this.x <= minX) {
      this.x = minX;
      this.walkDir = 1;
    }
    this.facing = this.walkDir;
    this.y = floor;
  }

  _stepResting(now, g) {
    if (now >= this.toggleAt) {
      if (this.state === "idle") this._enterSit(now);
      else this._enterIdle(now);
    }
    if (this.state === "idle") {
      if (this.ctx.reducedMotion === true) {
        if (this.motion === "fidget") this._set("idle", null, null, now);
        this.fidgetUntil = null;
      } else if (this.fidgetUntil !== null) {
        if (now >= this.fidgetUntil) {
          this.fidgetUntil = null;
          this._set("idle", null, null, now);
          this.fidgetAt = now + lerp(PET.FIDGET_GAP_MS, this.rng());
        }
      } else if (now >= this.fidgetAt) {
        this._set("idle", null, "fidget", now);
        this.fidgetUntil = now + PET.FIDGET_MS;
      }
    }
    if (now >= this.nextWalkAt) this._tryWalk(now, g);
  }

  /** Gaming: she holds her pose (no sit or stand toggle, no fidget, no walk, no falling asleep or waking). A fidget under
   *  way ends at once, so no wobble is left frozen on screen. */
  _freeze(now) {
    if (this.state === "idle" && this.fidgetUntil !== null) {
      this.fidgetUntil = null;
      this._set("idle", null, null, now);
    }
  }

  _tryWalk(now, g) {
    const { minX, maxX } = walkBounds(this.area, this.size.width);
    if (!g.walk || maxX <= minX) {
      this._drawWalk(now);
      return;
    }
    this.walkEndAt = now + lerp(PET.WALK_BURST_MS, this.rng());
    this.walkSpeed = lerp(PET.WALK_SPEED, this.rng());
    if (this.x <= minX) this.walkDir = 1;
    else if (this.x >= maxX) this.walkDir = -1;
    else this.walkDir = this.rng() < 0.5 ? -1 : 1;
    this.facing = this.walkDir;
    this.y = floorY(this.area, this.size.height);
    this._set("walk", null, "walk", now);
  }

  _view() {
    return {
      on: true, state: this.state, facing: this.facing, expression: this.expression, motion: this.motion,
      effect: this.effect, effectSeq: this.effectSeq, bubble: this.bubble ? { ...this.bubble } : null,
      reduced: this.ctx.reducedMotion === true, motionAt: this.motionAt,
    };
  }

  _frame(drag) {
    const x = Math.round(finite(this.x, 0));
    const y = Math.round(finite(this.y, 0));
    const view = this._view();
    const frame = {
      x, y, moved: !drag && (x !== this.lx || y !== this.ly), save: !drag && this.saveNext,
      viewChanged: !viewsEqual(view, this.lastView), view,
    };
    if (!drag) {
      this.lx = x;
      this.ly = y;
      this.saveNext = false;
    }
    this.lastView = view;
    return frame;
  }

  /** Disabled: she is off. Everything is put back to neutral once, and the frame carries OFF_VIEW. */
  _disabledFrame(now) {
    if (!this.disabled) {
      this.disabled = true;
      this._cancelMotion();
      this.queue = [];
      this.bubble = null;
      this.sleepForced = false;
      this.facing = 1;
      this._enterIdle(now);
    }
    const frame = {
      x: Math.round(finite(this.x, 0)), y: Math.round(finite(this.y, 0)), moved: false, save: false,
      viewChanged: this.lastView !== OFF_VIEW, view: OFF_VIEW,
    };
    this.lastView = OFF_VIEW;
    return frame;
  }
}

// ---- renderer helpers ----

/** The sprite transform for a view at `tMs`, rounded (0.2 px, 0.2 degree, 1e-3) so a held pose gives the same text. */
function petTransformCss(view, tMs) {
  const v = view || OFF_VIEW;
  const t = Math.max(0, finite(tMs, 0) - finite(v.motionAt, 0));
  const sign = v.facing === -1 ? -1 : 1;
  let dx = 0;
  let dy = 0;
  let deg = 0;
  let sx = 1;
  let sy = 1;
  if (v.motion === "sit") {
    dy = 4;
    sx = 1.02;
    sy = 0.95;
  } else if (v.motion === "sleep") {
    dy = 6;
    sy = 0.97;
  } else if (v.reduced !== true) {
    if (v.motion === "walk") {
      dy = -3 * Math.sin(Math.PI * ((t % 500) / 500));
      deg = 2 * Math.sin((2 * Math.PI * t) / 1000);
    } else if (v.motion === "wave") deg = 5 * Math.sin((2 * Math.PI * t) / 600);
    else if (v.motion === "dangle") deg = 8 * Math.sin((2 * Math.PI * t) / 900);
    else if (v.motion === "fidget") deg = 3 * Math.sin((2 * Math.PI * t) / 400);
    else if (v.motion === "spin") sx = Math.cos((2 * Math.PI * Math.min(t, 800)) / 800);
    else if (v.motion === "squash") {
      sx = 1.06;
      sy = 0.9;
    }
  }
  const n = (value, step, digits) => Number((Math.round(value / step) * step).toFixed(digits)); // -0 prints as 0
  const px = (value) => n(value, 0.2, 1);
  return `translate(${px(dx)}px, ${px(dy)}px) rotate(${px(deg)}deg) scale(${n(sign * sx, 1e-3, 3)}, ${n(sy, 1e-3, 3)})`;
}

/** The art pack's `motions.walk` frame at `tMs` (looping at `fps`), or null with none. */
function walkFrame(frames, tMs, fps = 8) {
  if (!Array.isArray(frames) || frames.length === 0) return null;
  const index = Math.floor((Math.max(0, finite(tMs, 0)) * fps) / 1000);
  return frames[index % frames.length];
}

/** A point in the sprite's own box seen through the facing flip (-1 mirrors #pet and its hit mask). */
function mirrorForFacing({ x, y }, width, facing) {
  return facing === -1 ? { x: width - x, y } : { x, y };
}

const petExports = {
  ACTIVITY_KINDS, PET_WALKS_WHEN, REDUCED_MOTION, SCALE_RANGE, BUBBLE_MAX, PET, STATES, REACTIONS, POKE, BIG_EFFECTS,
  BIG_MOTIONS, OFF_VIEW, Pet, makeRng, clampScale, petSize, floorY, walkBounds, inQuietHours, reducedMotionOn, gates,
  bubbleText, petTransformCss, walkFrame, mirrorForFacing,
};
if (typeof module !== "undefined" && module.exports) module.exports = petExports;
else globalThis.RiyaPet = petExports;
