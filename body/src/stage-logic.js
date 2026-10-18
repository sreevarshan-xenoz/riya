"use strict";
/** Pure stage helpers: which expression to show, the mouth, the error line and the panel crop.
 *  Loads in Node (tests) and as a plain script in the renderer. */

const { MouthGate: SpriteMouthGate } = typeof module !== "undefined" && module.exports ? require("./sprite-logic") : globalThis.RiyaSprite;

const CURIOUS_BID = "curious";
const MOUTH_SILENCE_MS = 250;
const ERROR_MS = 5000;

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

  onBid() {
    this.bidOverride = true;
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

const stageExports = { ExpressionModel, MouthTracker, StatusLine, cropBox, CURIOUS_BID };
if (typeof module !== "undefined" && module.exports) module.exports = stageExports;
else globalThis.RiyaStage = stageExports;
