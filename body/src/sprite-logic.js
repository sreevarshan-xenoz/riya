'use strict';
/** Pure sprite logic: manifest checks, frame fallback, mouth gate, blink, breathing and lean. */

const VARIANTS = ['closed', 'open', 'blink'];

function validateManifest(manifest, expressions) {
  // sprite-manifest.schema.json is the authoritative contract; this is a runtime sanity check.
  const need = (ok, what) => { if (!ok) throw new Error(`sprite manifest: missing ${what}`); };
  need(manifest && typeof manifest === 'object', 'manifest object');
  need(typeof manifest.name === 'string' && manifest.name, 'name');
  need(Number.isInteger(manifest.version), 'version');
  need(manifest.canvas && manifest.canvas.width > 0 && manifest.canvas.height > 0, 'canvas');
  need(manifest.crops && Array.isArray(manifest.crops.upper) && manifest.crops.upper.length === 4, 'crops.upper');
  const frames = manifest.expressions || {};
  need(frames.neutral && frames.neutral.closed, 'expressions.neutral.closed');
  const warnings = [];
  for (const expression of expressions) {
    for (const variant of VARIANTS) {
      if (!(frames[expression] && frames[expression][variant])) warnings.push(`missing ${expression}.${variant}`);
    }
  }
  return warnings;
}

/** variant -> closed -> neutral.<variant> -> neutral.closed */
function resolveFrame(manifest, expression, variant) {
  const frames = manifest.expressions;
  const own = frames[expression] || {};
  const neutral = frames.neutral;
  return own[variant] || own.closed || neutral[variant] || neutral.closed;
}

class MouthGate {
  constructor({ openAt = 0.3, closeAt = 0.15 } = {}) {
    this.openAt = openAt;
    this.closeAt = closeAt;
    this.open = false;
  }

  update(level) {
    if (this.open ? level < this.closeAt : level >= this.openAt) this.open = !this.open;
    return this.open;
  }
}

class BlinkScheduler {
  constructor({ rng = Math.random, minMs = 3000, maxMs = 6000, closedMs = 120 } = {}) {
    this.rng = rng;
    this.minMs = minMs;
    this.maxMs = maxMs;
    this.closedMs = closedMs;
    this.nextStart = null;
  }

  gap() {
    return this.minMs + Math.floor(this.rng() * (this.maxMs - this.minMs + 1));
  }

  isClosed(nowMs) {
    if (this.nextStart === null) this.nextStart = nowMs + this.gap();
    while (nowMs >= this.nextStart + this.closedMs) this.nextStart += this.gap();
    return nowMs >= this.nextStart;
  }

  /** When isClosed next flips (the next blink start, or the end of the current one), for a timer instead of a loop. */
  nextChange(nowMs) {
    return this.isClosed(nowMs) ? this.nextStart + this.closedMs : this.nextStart;
  }
}

function breathing(tMs) {
  const phase = (1 - Math.cos((2 * Math.PI * tMs) / 4000)) / 2;
  return { scale: 1 + 0.012 * phase, dy: 0 - 3 * phase };
}

function leanFor({ presence, curiosity } = {}) {
  return presence === 'focused' && curiosity > 0.6 ? { rotateDeg: -6, dx: -12 } : { rotateDeg: 0, dx: 0 };
}

function frameFor({ manifest, expression, mouthOpen, blinking }) {
  return resolveFrame(manifest, expression, blinking ? 'blink' : mouthOpen ? 'open' : 'closed');
}

const spriteExports = { VARIANTS, validateManifest, resolveFrame, MouthGate, BlinkScheduler, breathing, leanFor, frameFor };
if (typeof module !== 'undefined' && module.exports) module.exports = spriteExports;
else globalThis.RiyaSprite = spriteExports;
