"use strict";
/** Pure alpha hit test for buddy-mode click-through. Loads in Node (tests) and as a plain script in the renderer. */

/** `rgba` is RGBA pixel data (e.g. ImageData.data) of a `width`-pixel-wide image; opaque means alpha >= threshold. */
function isOpaqueAt(rgba, width, x, y, threshold = 16) {
  if (!Number.isFinite(x) || !Number.isFinite(y)) return false;
  const px = Math.floor(x);
  const py = Math.floor(y);
  const height = Math.floor(rgba.length / 4 / width);
  if (px < 0 || py < 0 || px >= width || py >= height) return false;
  return rgba[(py * width + px) * 4 + 3] >= threshold;
}

if (typeof module !== "undefined" && module.exports) module.exports = { isOpaqueAt };
else globalThis.RiyaHitTest = { isOpaqueAt };
