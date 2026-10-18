"use strict";
/** Pure window geometry for the three modes. All values are in DIPs (Electron's screen coordinates). */

const MARGIN = 24;
const PANEL_WIDTH = 320;
const WINDOW_SIZE = [960, 640];
const DEFAULT_SIZE = { buddy: [300, 520] };

/** Move `rect` so it lies inside `workArea`; a rect larger than the area is pinned to the area's origin. */
function clampToArea(rect, workArea) {
  const maxX = workArea.x + workArea.width - rect.width;
  const maxY = workArea.y + workArea.height - rect.height;
  return {
    ...rect,
    x: Math.round(Math.max(workArea.x, Math.min(rect.x, maxX))),
    y: Math.round(Math.max(workArea.y, Math.min(rect.y, maxY))),
  };
}

function validPos(pos) {
  return Array.isArray(pos) && pos.length === 2 && pos.every((n) => typeof n === "number" && Number.isFinite(n));
}

function boundsFor(mode, workArea, savedPos, size = DEFAULT_SIZE) {
  const wa = workArea;
  if (mode === "buddy") {
    const [width, height] = size.buddy || DEFAULT_SIZE.buddy;
    const rect = validPos(savedPos)
      ? { x: savedPos[0], y: savedPos[1], width, height }
      : { x: wa.x + wa.width - width - MARGIN, y: wa.y + wa.height - height - MARGIN, width, height };
    return clampToArea(rect, wa);
  }
  if (mode === "panel") {
    return { x: wa.x + wa.width - PANEL_WIDTH, y: wa.y, width: PANEL_WIDTH, height: wa.height };
  }
  if (mode === "window") {
    const width = Math.min(WINDOW_SIZE[0], wa.width);
    const height = Math.min(WINDOW_SIZE[1], wa.height);
    return {
      x: wa.x + Math.round((wa.width - width) / 2),
      y: wa.y + Math.round((wa.height - height) / 2),
      width,
      height,
    };
  }
  throw new Error(`unknown mode: ${mode}`);
}

module.exports = { boundsFor, clampToArea, MARGIN, PANEL_WIDTH, WINDOW_SIZE };
