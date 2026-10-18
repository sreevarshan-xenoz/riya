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

/** A converter over Electron's `{ bounds, scaleFactor }` displays with the shape of `screen.dipToScreenPoint` and
 *  `screen.screenToDipPoint`: DIP x scale of the display the point is on. Exact only where all displays share a scale
 *  (the main process uses the real `screen` on Windows, which is exact with mixed DPI). */
function displayConverter(displays) {
  const physical = (d) => {
    const k = scaleOf(d);
    return { x: d.bounds.x * k, y: d.bounds.y * k, width: d.bounds.width * k, height: d.bounds.height * k };
  };
  return {
    dipToScreenPoint({ x, y }) {
      const k = scaleOf(displayAt([x, y], displays, (d) => d.bounds));
      return { x: Math.round(x * k), y: Math.round(y * k) };
    },
    screenToDipPoint({ x, y }) {
      const k = scaleOf(displayAt([x, y], displays, physical));
      return { x: Math.round(x / k), y: Math.round(y / k) };
    },
  };
}

const convert = (pos, fn) => {
  if (!isPos(pos) || !fn) return pos;
  try {
    const out = fn({ x: pos[0], y: pos[1] });
    return out && Number.isFinite(out.x) && Number.isFinite(out.y) ? [Math.round(out.x), Math.round(out.y)] : pos;
  } catch (e) {
    return pos; // e.g. Electron on a platform without the conversion
  }
};

/** A buddy position in DIPs -> physical pixels (what the core compares with window rects). `conv` has Electron's
 *  `dipToScreenPoint`; anything that is not a finite [x, y] passes through. */
function toPhysical(pos, conv) {
  return convert(pos, conv && conv.dipToScreenPoint && conv.dipToScreenPoint.bind(conv));
}

/** The inverse of toPhysical, through `conv.screenToDipPoint`. */
function toDip(pos, conv) {
  return convert(pos, conv && conv.screenToDipPoint && conv.screenToDipPoint.bind(conv));
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
  toPhysical, toDip, displayConverter,
};
