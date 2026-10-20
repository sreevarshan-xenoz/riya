const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const E = require("../src/effects-logic");

test("particles are seeded and sized", () => {
  assert.deepEqual(E.particles("confetti", 7, 240, 344), E.particles("confetti", 7, 240, 344));
  assert.notDeepEqual(E.particles("confetti", 7, 240, 344), E.particles("confetti", 8, 240, 344));
  assert.equal(E.particles("confetti", 1, 240, 344).length, 24);
  assert.equal(E.particles("sparkle", 1, 240, 344).length, 8);
  assert.equal(E.particles("sweat", 1, 240, 344).length, 1);
  assert.equal(E.particles("heart", 1, 240, 344).length, 1);
  assert.equal(E.particles("nope", 1, 240, 344).length, 0);
  assert.equal(E.particles("constructor", 1, 240, 344).length, 0, "only the four kinds");
  for (const kind of Object.keys(E.EFFECTS)) {
    assert.equal(E.particles(kind, 3, 240, 344).length, E.EFFECTS[kind].count, kind);
    assert.ok(E.particles(kind, 3, 240, 344).every((p) => E.COLORS.includes(p.color)), kind);
  }
});

test("the effect table and palette are the brief's", () => {
  assert.equal(E.EFFECT_HZ, 30);
  assert.deepEqual({ ...E.EFFECTS.confetti }, { ms: 1500, count: 24 });
  assert.deepEqual({ ...E.EFFECTS.sparkle }, { ms: 900, count: 8 });
  assert.deepEqual({ ...E.EFFECTS.sweat }, { ms: 1200, count: 1 });
  assert.deepEqual({ ...E.EFFECTS.heart }, { ms: 1200, count: 1 });
  assert.deepEqual([...E.COLORS], ["#ff8fb1", "#ffd66b", "#8fd3ff", "#b9f18f", "#c9a7ff"]);
});

test("confetti bursts up and out of her head, then falls", () => {
  const list = E.particles("confetti", 5, 240, 344);
  const start = E.particlesAt(list, 0);
  assert.ok(start.every((p) => p.x === 120 && p.y === 104), "from the middle, at the top of her sprite");
  const early = E.particlesAt(list, 150);
  assert.ok(early.some((p) => p.y < 104), "some go up");
  assert.ok(early.some((p) => p.x < 120) && early.some((p) => p.x > 120), "out to both sides");
  const late = E.particlesAt(list, 1400);
  assert.ok(late.every((p, i) => p.y > early[i].y), "gravity wins in the end");
});

test("the sparkle sits around her head, the sweat beside her face, the heart above", () => {
  const head = 104 + 0.12 * 240;
  const sparkle = E.particlesAt(E.particles("sparkle", 2, 240, 344), 0);
  assert.ok(sparkle.every((p) => Math.hypot(p.x - 120, (p.y - head) / 0.8) > 25 && Math.hypot(p.x - 120, (p.y - head) / 0.8) < 55));
  const [sweat] = E.particlesAt(E.particles("sweat", 2, 240, 344), 0);
  assert.ok(sweat.x > 120 && Math.abs(sweat.y - head) < 1, "to her side, at face height");
  const [heart] = E.particlesAt(E.particles("heart", 2, 240, 344), 0);
  const [later] = E.particlesAt(E.particles("heart", 2, 240, 344), 800);
  assert.ok(heart.y < head, "above her head");
  assert.ok(later.y < heart.y, "and it rises");
});

test("effects end on time and draw nothing after", () => {
  for (const [kind, spec] of Object.entries(E.EFFECTS)) {
    const list = E.particles(kind, 9, 240, 344);
    assert.ok(E.particlesAt(list, spec.ms - 1).some((p) => p.alpha > 0), `${kind} still shows 1 ms before its end`);
    assert.equal(E.effectDone(kind, spec.ms - 1), false, kind);
    assert.ok(E.particlesAt(list, spec.ms).every((p) => p.alpha === 0), `${kind} is clear at its end`);
    assert.ok(E.particlesAt(list, spec.ms + 5000).every((p) => p.alpha === 0), kind);
    assert.equal(E.effectDone(kind, spec.ms), true, kind);
    assert.equal(E.effectDone(kind, spec.ms + 1), true, kind);
    assert.ok(E.particlesAt(list, 0).every((p) => p.alpha === 1), `${kind} starts solid`);
    assert.ok(E.particlesAt(list, -50).every((p) => p.alpha === 0), `${kind} draws nothing before it starts`);
  }
  assert.equal(E.effectDone("nope", 0), true, "an unknown kind is done at once");
  assert.equal(E.effectDone(undefined, 0), true);
  assert.deepEqual(E.particlesAt(null, 0), []);
});

test("alpha holds for half the life and then fades", () => {
  const list = E.particles("confetti", 1, 240, 344);
  assert.ok(E.particlesAt(list, 700).every((p) => p.alpha === 1));
  const mid = E.particlesAt(list, 1125);
  assert.ok(mid.every((p) => p.alpha > 0.49 && p.alpha < 0.51));
});

test("particles stay inside the window", () => {
  for (const [w, h] of [[240, 344], [300, 520], [240, 104], [0, 0]]) {
    for (const kind of Object.keys(E.EFFECTS)) {
      for (let seed = 0; seed < 20; seed++) {
        const list = E.particles(kind, seed, w, h);
        for (let t = 0; t <= E.EFFECTS[kind].ms; t += 25) {
          for (const p of E.particlesAt(list, t)) {
            assert.ok(p.x >= 0 && p.x <= w && p.y >= 0 && p.y <= h, `${kind} ${w}x${h} seed ${seed} t ${t}: ${p.x}, ${p.y}`);
            assert.ok(Number.isFinite(p.x) && Number.isFinite(p.y) && p.r > 0);
          }
        }
      }
    }
  }
});

test("junk sizes and seeds never make a NaN", () => {
  for (const args of [[NaN, 240, 344], [1, NaN, NaN], [undefined, undefined, undefined], [1e9, -5, -5]]) {
    const list = E.particles("confetti", ...args);
    assert.equal(list.length, 24);
    for (const p of E.particlesAt(list, 300)) assert.ok(Number.isFinite(p.x) && Number.isFinite(p.y));
  }
});

test("the module loads as a plain script after pet-logic, with no top-level name clash", () => {
  const dir = path.join(__dirname, "..");
  const html = fs.readFileSync(path.join(dir, "index.html"), "utf8");
  const scripts = [...html.matchAll(/<script src="(src\/[^"]+)"><\/script>/g)].map((m) => m[1]);
  assert.ok(scripts.indexOf("src/pet-logic.js") >= 0 && scripts.indexOf("src/effects-logic.js") > scripts.indexOf("src/pet-logic.js"));
  // every classic script shares one global lexical scope in the renderer: a repeated top-level const would be a SyntaxError
  const g = {};
  g.globalThis = g;
  const context = vm.createContext(g);
  for (const file of scripts) vm.runInContext(fs.readFileSync(path.join(dir, file), "utf8"), context, { filename: file });
  assert.equal(typeof g.RiyaEffects.particles, "function");
  assert.equal(g.RiyaEffects.particles("confetti", 1, 240, 344).length, 24);
});

test("the file needs no DOM, no clock and no unseeded randomness", () => {
  const source = fs.readFileSync(path.join(__dirname, "..", "src", "effects-logic.js"), "utf8");
  const code = source.replace(/\/\*[\s\S]*?\*\//g, "").replace(/^\s*\/\/.*$/gm, "").replace(/\s\/\/ .*$/gm, "");
  assert.ok(!/Math\.random|Date\.now|performance\.now|setTimeout|setInterval/.test(code));
  assert.ok(!/\bdocument\b|\bwindow\b|\belectron\b/i.test(code));
});
