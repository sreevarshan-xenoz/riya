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

/** Client (mouse) coordinates -> pixel coordinates of an image of imgW x imgH drawn to fill `rect`; null for an empty rect. */
function toImageCoords(rect, imgW, imgH, clientX, clientY) {
  if (!(rect.width > 0) || !(rect.height > 0)) return null;
  return { x: ((clientX - rect.left) / rect.width) * imgW, y: ((clientY - rect.top) / rect.height) * imgH };
}

/** Should the buddy window take the mouse? Over her opaque pixels or her bubble, and over the confirmation's buttons only
 *  while a confirmation is shown (otherwise the hidden buttons' spot stays click-through). */
function buddyClickable({ onBubble, onConfirm, confirmShown, overSprite } = {}) {
  return Boolean(onBubble || (confirmShown && onConfirm) || overSprite);
}

if (typeof module !== "undefined" && module.exports) module.exports = { isOpaqueAt, toImageCoords, buddyClickable };
else globalThis.RiyaHitTest = { isOpaqueAt, toImageCoords, buddyClickable };
