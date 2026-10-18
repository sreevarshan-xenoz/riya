"use strict";
/** Pure decisions used by main.js: hotkey fallback, mode fallback, late saved-position repositioning, send filtering. */
const { boundsFor } = require("./geometry");

const MODES = ["buddy", "panel", "window"];
const REPLAY_TYPES = ["hello", "settings", "state", "expression", "look", "eyes"];
const DONT_LOOK_HOTKEYS = ["Control+Alt+H", "Control+Alt+F10"]; // first one that registers wins

/** First accelerator for which `register` returns true (a throw counts as a failure), or null. */
function pickHotkey(accelerators, register) {
  for (const accelerator of accelerators) {
    try {
      if (register(accelerator)) return accelerator;
    } catch (e) { /* invalid or unavailable accelerator: try the next */ }
  }
  return null;
}

/** The mode to show: the requested one if valid, else the current one, else buddy, so a window always appears. */
function modeToApply(requested, current) {
  if (MODES.includes(requested)) return requested;
  return MODES.includes(current) ? current : "buddy";
}

/** New bounds for an already-shown buddy window when a saved position arrives late, or null to leave her alone. */
function repositionTarget({ mode, dragging, userDragged, current, savedPos, workArea }) {
  if (mode !== "buddy" || dragging || userDragged) return null;
  if (!Array.isArray(savedPos) || savedPos.length !== 2 || !savedPos.every(Number.isFinite)) return null;
  const target = boundsFor("buddy", workArea, savedPos);
  return target.x === current[0] && target.y === current[1] ? null : target;
}

/** A talk or don't-look press is only meaningful now: never queue it for replay after a reconnect. */
function shouldSendToCore(msg, connected) {
  return connected || (msg.type !== "talk_hotkey" && msg.type !== "dont_look");
}

const isPos = (pos) => Array.isArray(pos) && pos.length === 2 && pos.every(Number.isFinite);
const scaleOf = (display) => (Number.isFinite(display && display.scaleFactor) && display.scaleFactor > 0 ? display.scaleFactor : 1);

/** The display whose `rectOf` rect holds the point, else the one whose rect is nearest, else null. */
function displayAt(point, displays, rectOf) {
  let best = null;
  let bestDist = Infinity;
  for (const display of Array.isArray(displays) ? displays : []) {
    const { x, y, width, height } = rectOf(display);
    const dx = Math.max(x - point[0], 0, point[0] - (x + width - 1));
    const dy = Math.max(y - point[1], 0, point[1] - (y + height - 1));
    const dist = dx * dx + dy * dy;
    if (dist < bestDist) { best = display; bestDist = dist; }
  }
  return best;
}

/** A buddy position in DIPs -> physical pixels (what the core compares with window rects); `displays` are Electron's
 *  { bounds, scaleFactor }. Anything that is not a finite [x, y] passes through. */
function toPhysical(pos, displays) {
  if (!isPos(pos)) return pos;
  const k = scaleOf(displayAt(pos, displays, (d) => d.bounds));
  return [Math.round(pos[0] * k), Math.round(pos[1] * k)];
}

/** The inverse of toPhysical, using the display whose physical rect holds (or is nearest to) the position. */
function toDip(pos, displays) {
  if (!isPos(pos)) return pos;
  const physical = (d) => {
    const k = scaleOf(d);
    return { x: d.bounds.x * k, y: d.bounds.y * k, width: d.bounds.width * k, height: d.bounds.height * k };
  };
  const k = scaleOf(displayAt(pos, displays, physical));
  return [Math.round(pos[0] / k), Math.round(pos[1] / k)];
}

/** Keep the latest message of each replay type, in recency order, so a replay ends with the newest (state or
 *  expression wins by being last). */
function rememberForReplay(cache, msg) {
  if (!REPLAY_TYPES.includes(msg.type)) return;
  cache.delete(msg.type);
  cache.set(msg.type, msg);
}

module.exports = {
  MODES, REPLAY_TYPES, DONT_LOOK_HOTKEYS, pickHotkey, modeToApply, repositionTarget, shouldSendToCore, rememberForReplay,
  toPhysical, toDip,
};
