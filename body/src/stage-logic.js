"use strict";
/** Pure stage helpers: which expression to show, the mouth, the error line, idle motion and when it runs, and the
 *  panel and buddy layouts. Loads in Node (tests) and as a plain script in the renderer. */

const { MouthGate: SpriteMouthGate, breathing: spriteBreathing } =
  typeof module !== "undefined" && module.exports ? require("./sprite-logic") : globalThis.RiyaSprite;

const CURIOUS_BID = "curious";
const MOUTH_SILENCE_MS = 250;
const ERROR_MS = 5000;
const LEAN_EASE_PER_60HZ_FRAME = 0.08;

/** The core emits no expression for a silent bid, so a visible bid shows `curious` unless an `expression` arrives. */
class ExpressionModel {
  constructor() {
    this.base = "neutral";
    this.connected = true;
    this.bidOverride = false;
  }

  onState(state) {
    if (state && typeof state.expression === "string") this.base = state.expression;
  }

  onExpression(name) {
    if (typeof name !== "string") return;
    this.base = name;
    this.bidOverride = false;
  }

  /** A `say` message: a new bid (`fresh`) shows curious; any other say (her reply) clears it. */
  onSay({ kind, fresh }) {
    if (kind !== "bid") this.bidOverride = false;
    else if (fresh) this.bidOverride = true;
  }

  setConnected(connected) {
    this.connected = Boolean(connected);
  }

  current(bidVisible) {
    if (!this.connected) return "neutral";
    if (!bidVisible) this.bidOverride = false;
    return this.bidOverride ? CURIOUS_BID : this.base;
  }
}

/** Mouth open/closed from `mouth` levels (hysteresis), closing by itself once the messages stop. */
class MouthTracker {
  constructor({ openAt = 0.3, closeAt = 0.15, silenceMs = MOUTH_SILENCE_MS } = {}) {
    this.gate = new SpriteMouthGate({ openAt, closeAt });
    this.silenceMs = silenceMs;
    this.lastAt = -Infinity;
  }

  onLevel(level, nowMs) {
    if (nowMs - this.lastAt >= this.silenceMs) this.gate.open = false;
    this.lastAt = nowMs;
    this.gate.update(level);
  }

  isOpen(nowMs) {
    if (nowMs - this.lastAt >= this.silenceMs) this.gate.open = false;
    return this.gate.open;
  }

  /** When silence will close an open mouth (for a timer), else null. */
  closesAt() {
    return this.gate.open ? this.lastAt + this.silenceMs : null;
  }
}

/** A short-lived one-line status (core errors), kept out of her speech bubble. */
class StatusLine {
  constructor({ ttlMs = ERROR_MS } = {}) {
    this.ttlMs = ttlMs;
    this.message = "";
    this.until = 0;
  }

  show(message, nowMs) {
    this.message = String(message || "");
    this.until = nowMs + this.ttlMs;
  }

  text(nowMs) {
    return nowMs < this.until ? this.message : "";
  }

  clearsAt(nowMs) {
    return nowMs < this.until ? this.until : null;
  }
}

/** Settings errors belong to the settings window: they name a setting (or autostart). */
function isSettingsError(message) {
  const text = typeof message === "string" ? message : "";
  return /\bsetting '\w+'/.test(text) || /\bautostart\b/i.test(text);
}

/** Breathing, blink and lean run only while she is visible and the user isn't gaming; mouth frames always run. */
function idleAnimates({ presence, hidden }) {
  return !hidden && presence !== "gaming";
}

/** The sprite transform, rounded (0.1 px, 0.1 degree, 1e-4 scale) so an unchanged pose gives the same text. */
function transformCss({ dx, dy, rotateDeg, scale }) {
  const n = (value, step, digits) => Number((Math.round(value / step) * step).toFixed(digits)); // -0 prints as 0
  const px = (value) => n(value, 0.1, 1);
  return `translate(${px(dx)}px, ${px(dy)}px) rotate(${px(rotateDeg)}deg) scale(${n(scale, 1e-4, 4)})`;
}

/** Breathing plus a lean eased toward its target by elapsed time, so any tick rate moves her the same way. */
class Motion {
  constructor() {
    this.rotateDeg = 0;
    this.dx = 0;
    this.lastAt = null;
  }

  /** The transform text at `nowMs` with the lean eased toward `target` ({ rotateDeg, dx }). */
  step(nowMs, target) {
    const frames = this.lastAt === null ? 0 : Math.max(0, nowMs - this.lastAt) / (1000 / 60);
    this.lastAt = nowMs;
    const k = 1 - Math.pow(1 - LEAN_EASE_PER_60HZ_FRAME, frames);
    this.rotateDeg += (target.rotateDeg - this.rotateDeg) * k;
    this.dx += (target.dx - this.dx) * k;
    const b = spriteBreathing(nowMs);
    return transformCss({ dx: this.dx, dy: b.dy, rotateDeg: this.rotateDeg, scale: b.scale });
  }
}

/** The earliest of `times` after `nowMs`, or null. */
function nextDeadline(nowMs, times) {
  let best = null;
  for (const t of times) if (typeof t === "number" && t > nowMs && (best === null || t < best)) best = t;
  return best;
}

/** Geometry for showing `crop` ([x, y, w, h] in canvas pixels) of the full image inside a box `boxWidth` wide. */
function cropBox(boxWidth, canvas, crop) {
  const [x, y, w, h] = crop;
  const scale = boxWidth / w;
  return {
    width: boxWidth,
    height: h * scale,
    imgWidth: canvas.width * scale,
    imgHeight: canvas.height * scale,
    left: 0 - x * scale,
    top: 0 - y * scale,
  };
}

/** Buddy sprite box size: as tall as fits with the face (the top of `crops.upper`) at least `gap` below the bubble
 *  region's bottom, standing on the window's bottom edge, and never taller or wider than the window. */
function buddySpriteBox({ winWidth, winHeight, canvas, crops, bubbleBottom, gap }) {
  const faceTop = Math.min(0.99, Math.max(0, crops.upper[1] / canvas.height));
  const aspect = canvas.width / canvas.height;
  const height = Math.min((winHeight - bubbleBottom - gap) / (1 - faceTop), winHeight, winWidth / aspect);
  return { width: height * aspect, height };
}

const stageExports = {
  ExpressionModel, MouthTracker, StatusLine, Motion, cropBox, buddySpriteBox, transformCss, idleAnimates, nextDeadline,
  isSettingsError, CURIOUS_BID,
};
if (typeof module !== "undefined" && module.exports) module.exports = stageExports;
else globalThis.RiyaStage = stageExports;
