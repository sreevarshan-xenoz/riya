'use strict';
/** Generates body/placeholder-pack/: 14 expressions x (closed, open, blink) SVG frames plus manifest.json. Deterministic. */
const fs = require('node:fs');
const path = require('node:path');

const W = 384;
const H = 672;
const OUT = path.join(__dirname, '..', 'placeholder-pack');
const EXPRESSIONS = JSON.parse(
  fs.readFileSync(path.join(__dirname, '..', '..', 'docs', 'protocol.schema.json'), 'utf8'),
).expressions;

const SKIN = '#ffe4d6';
const SKIN_SHADE = '#f2c4b0';
const HAIR = '#cfd5e2';
const HAIR_SHADE = '#aab3c7';
const IRIS = '#4c8fe0';
const INK = '#3a3f55';
const RIBBON = '#d8243c';
const TEAR = '#9fd2ff';
const EX = [150, 234];
const EY = 205;
const MX = 192;
const MY = 258;

const arc = (cx, cy, w, up) => `<path d="M ${cx - w} ${cy} Q ${cx} ${cy + (up ? -w : w)} ${cx + w} ${cy}" fill="none" stroke="${INK}" stroke-width="5" stroke-linecap="round"/>`;
const line = (d) => `<path d="${d}" fill="none" stroke="${INK}" stroke-width="5" stroke-linecap="round" stroke-linejoin="round"/>`;
const hole = (d) => `<path d="${d}" fill="#c0475f" stroke="${INK}" stroke-width="4" stroke-linejoin="round"/>`;
const closedEye = (cx) => arc(cx, EY + 4, 18, false);
const brow = (cx, tilt) => `<path d="M ${cx - 18} ${EY - 38 + tilt} L ${cx + 18} ${EY - 38 - tilt}" stroke="${INK}" stroke-width="4" stroke-linecap="round"/>`;
const blush = (opacity) => EX.map((cx) => `<ellipse cx="${cx + (cx < 192 ? -14 : 14)}" cy="238" rx="18" ry="9" fill="#ff8aa0" opacity="${opacity}"/>`).join('');
const sparkle = (x, y) => `<path d="M ${x} ${y - 14} L ${x + 4} ${y - 4} L ${x + 14} ${y} L ${x + 4} ${y + 4} L ${x} ${y + 14} L ${x - 4} ${y + 4} L ${x - 14} ${y} L ${x - 4} ${y - 4} Z" fill="#ffd54a"/>`;
const drop = (x, y) => `<path d="M ${x} ${y} Q ${x + 9} ${y + 14} ${x} ${y + 22} Q ${x - 9} ${y + 14} ${x} ${y} Z" fill="${TEAR}"/>`;
const glyph = (x, y, size, ch) => `<text x="${x}" y="${y}" font-family="sans-serif" font-size="${size}" font-weight="bold" fill="${INK}">${ch}</text>`;

/** Open eye: white, blue iris, pupil, highlight. `lid` (0..1) hides that fraction of the eye with skin. */
function eye(cx, { rx = 20, ry = 24, px = 0, py = 0, lid = 0, glossy = false } = {}) {
  let s = `<ellipse cx="${cx}" cy="${EY}" rx="${rx}" ry="${ry}" fill="#fff" stroke="${INK}" stroke-width="3"/>`;
  s += `<ellipse cx="${cx + px}" cy="${EY + py}" rx="${Math.round(rx * 0.72)}" ry="${Math.round(ry * 0.8)}" fill="${IRIS}"/>`;
  s += `<ellipse cx="${cx + px}" cy="${EY + py}" rx="${Math.round(rx * 0.36)}" ry="${Math.round(ry * 0.42)}" fill="${INK}"/>`;
  s += `<circle cx="${cx + px - 6}" cy="${EY + py - 8}" r="${glossy ? 7 : 5}" fill="#fff"/>`;
  if (glossy) s += `<circle cx="${cx + px + 7}" cy="${EY + py + 8}" r="3" fill="#fff"/>`;
  if (lid > 0) {
    const top = EY - ry - 4;
    const bottom = Math.round(EY - ry + 2 * ry * lid);
    s += `<rect x="${cx - rx - 3}" y="${top}" width="${2 * rx + 6}" height="${bottom - top}" fill="${SKIN}"/>`;
    s += `<path d="M ${cx - rx} ${bottom} L ${cx + rx} ${bottom}" stroke="${INK}" stroke-width="5" stroke-linecap="round"/>`;
  }
  return s;
}

function hands() {
  return EX.map((cx) => `<circle cx="${cx}" cy="${EY}" r="34" fill="${SKIN}" stroke="${SKIN_SHADE}" stroke-width="5"/>`
    + `<path d="M ${cx - 18} ${EY - 8} L ${cx - 18} ${EY + 14} M ${cx} ${EY - 12} L ${cx} ${EY + 18} M ${cx + 18} ${EY - 8} L ${cx + 18} ${EY + 14}" stroke="${SKIN_SHADE}" stroke-width="3" stroke-linecap="round"/>`).join('');
}

const smile = (w) => `M ${MX - w} ${MY} Q ${MX} ${MY + w} ${MX + w} ${MY}`;
const grin = `M ${MX - 22} ${MY - 4} Q ${MX} ${MY + 34} ${MX + 22} ${MY - 4} Z`;
const smallO = `M ${MX} ${MY - 8} a 9 12 0 1 0 0.01 0 Z`;
const bigO = `M ${MX} ${MY - 12} a 13 18 0 1 0 0.01 0 Z`;
const wave = `M ${MX - 18} ${MY} q 6 -8 12 0 t 12 0 t 12 0`;
const frown = `M ${MX - 14} ${MY + 6} Q ${MX} ${MY - 8} ${MX + 14} ${MY + 6}`;
const flat = `M ${MX - 12} ${MY} L ${MX + 12} ${MY}`;
const smirk = `M ${MX - 14} ${MY + 2} Q ${MX + 4} ${MY + 14} ${MX + 18} ${MY - 6}`;
const pout = `M ${MX - 10} ${MY + 4} Q ${MX} ${MY - 4} ${MX + 10} ${MY + 4}`;
const halfOpen = (d) => `M ${MX - 12} ${MY} Q ${MX} ${MY + d} ${MX + 12} ${MY} Z`;
const pair = (opts) => () => [eye(EX[0], opts), eye(EX[1], opts)];

/** eyes(blinking) -> svg[]; mouth(open) -> svg; extra -> svg drawn on the face. */
const SPECS = {
  neutral: { eyes: pair({}), mouth: (o) => (o ? hole(halfOpen(14)) : line(smile(10))), extra: '' },
  happy: { eyes: pair({ glossy: true }), mouth: (o) => (o ? hole(grin) : line(smile(16))), extra: blush(0.4) },
  excited: { eyes: pair({ rx: 23, ry: 27, glossy: true }), mouth: (o) => (o ? hole(grin) : line(smile(16))), extra: blush(0.5) + sparkle(84, 120) + sparkle(304, 110) },
  missed_you: { eyes: pair({ glossy: true, py: 2 }), mouth: (o) => (o ? hole(halfOpen(18)) : line(smile(14))), extra: blush(0.55) + drop(EX[1] + 22, 220) },
  pouty: { eyes: pair({ lid: 0.35 }), mouth: (o) => (o ? hole(smallO) : line(pout)), extra: brow(EX[0], -6) + brow(EX[1], 6) + blush(0.55) },
  flustered: { eyes: pair({ px: 4, py: 4 }), mouth: (o) => (o ? hole(smallO) : line(wave)), extra: blush(0.85) + drop(300, 150) },
  teasing: {
    eyes: (blinking) => [eye(EX[0], { glossy: true }), blinking ? closedEye(EX[1]) : arc(EX[1], EY + 4, 18, true)],
    mouth: (o) => (o ? hole(halfOpen(18)) : line(smirk)),
    extra: blush(0.35),
  },
  worried: { eyes: pair({ py: 2 }), mouth: (o) => (o ? hole(smallO) : line(wave)), extra: brow(EX[0], 6) + brow(EX[1], -6) + drop(96, 150) },
  sad: {
    eyes: pair({ lid: 0.25, py: 4 }),
    mouth: (o) => (o ? hole(`M ${MX - 12} ${MY + 8} Q ${MX} ${MY - 10} ${MX + 12} ${MY + 8} Z`) : line(frown)),
    extra: brow(EX[0], 6) + brow(EX[1], -6) + drop(EX[0] - 6, 232),
  },
  sleepy: { eyes: pair({ lid: 0.6 }), mouth: (o) => (o ? hole(smallO) : line(flat)), extra: glyph(296, 130, 26, 'z') + glyph(316, 104, 20, 'z') },
  curious: { eyes: pair({ rx: 21, ry: 26, px: 5, glossy: true }), mouth: (o) => (o ? hole(smallO) : line(smile(8))), extra: brow(EX[1], -4) + glyph(300, 130, 30, '?') },
  thinking: {
    eyes: pair({ px: 8, py: -8 }),
    mouth: (o) => (o ? hole(smallO) : line(flat)),
    extra: [0, 1, 2].map((i) => `<circle cx="${290 + i * 16}" cy="${130 - i * 18}" r="${4 + i * 3}" fill="#fff" stroke="${INK}" stroke-width="2"/>`).join(''),
  },
  surprised: {
    eyes: pair({ rx: 25, ry: 30 }),
    mouth: (o) => (o ? hole(bigO) : line(smile(6))),
    extra: `<path d="M 60 140 L 80 150 M 56 170 L 80 170 M 324 140 L 304 150 M 328 170 L 304 170" stroke="${INK}" stroke-width="4" stroke-linecap="round"/>`,
  },
  covering_eyes: { eyes: () => [hands()], mouth: (o) => (o ? hole(smallO) : line(pout)), extra: blush(0.7) },
};

function frame(expression, variant) {
  const spec = SPECS[expression];
  const blinking = variant === 'blink';
  let eyes = spec.eyes(blinking);
  if (blinking && expression !== 'teasing' && expression !== 'covering_eyes') eyes = [closedEye(EX[0]), closedEye(EX[1])];
  return `<svg xmlns="http://www.w3.org/2000/svg" width="${W}" height="${H}" viewBox="0 0 ${W} ${H}">
<title>${expression} ${variant}</title>
<path d="M 64 180 C 44 400 70 560 112 596 L 272 596 C 314 560 340 400 320 180 Z" fill="${HAIR_SHADE}"/>
<rect x="148" y="480" width="26" height="104" rx="12" fill="${SKIN}"/><rect x="210" y="480" width="26" height="104" rx="12" fill="${SKIN}"/>
<ellipse cx="161" cy="590" rx="24" ry="12" fill="${INK}"/><ellipse cx="223" cy="590" rx="24" ry="12" fill="${INK}"/>
<path d="M 126 322 Q 192 296 258 322 L 276 500 Q 192 520 108 500 Z" fill="#2f3b66"/>
<path d="M 168 304 L 192 336 L 216 304 Z" fill="#fff"/>
<path d="M 192 330 l -16 -10 v 22 z M 192 330 l 16 -10 v 22 z" fill="${RIBBON}"/>
<rect x="178" y="286" width="28" height="30" fill="${SKIN}"/>
<ellipse cx="192" cy="185" rx="118" ry="112" fill="${SKIN}"/>
<path d="M 74 176 C 66 70 318 70 310 176 C 284 132 252 140 232 168 C 212 128 172 128 152 168 C 132 140 100 132 74 176 Z" fill="${HAIR}" stroke="${HAIR_SHADE}" stroke-width="3"/>
<path d="M 74 176 C 60 220 66 260 82 300 C 70 250 82 210 100 190 Z M 310 176 C 324 220 318 260 302 300 C 314 250 302 210 284 190 Z" fill="${HAIR}"/>
<path d="M 276 74 l 34 -22 v 44 z M 276 74 l -34 -22 v 44 z" fill="${RIBBON}"/><circle cx="276" cy="74" r="9" fill="#a8182c"/>
${eyes.join('')}
${spec.extra}
${spec.mouth(variant === 'open')}
<text x="${W / 2}" y="648" text-anchor="middle" font-family="sans-serif" font-size="22" fill="${INK}">${expression} / ${variant}</text>
</svg>
`;
}

function main() {
  fs.mkdirSync(OUT, { recursive: true });
  const expressions = {};
  for (const expression of EXPRESSIONS) {
    expressions[expression] = {};
    for (const variant of ['closed', 'open', 'blink']) {
      const file = `${expression}_${variant}.svg`;
      fs.writeFileSync(path.join(OUT, file), frame(expression, variant));
      expressions[expression][variant] = file;
    }
  }
  const manifest = { name: 'riya-placeholder', version: 1, canvas: { width: W, height: H }, crops: { upper: [0, 0, W, 350] }, expressions };
  fs.writeFileSync(path.join(OUT, 'manifest.json'), `${JSON.stringify(manifest, null, 2)}\n`);
  return EXPRESSIONS.length * 3;
}

if (require.main === module) process.stdout.write(`wrote ${main()} frames to ${OUT}\n`);
module.exports = { frame, main };
