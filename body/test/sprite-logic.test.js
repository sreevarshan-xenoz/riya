'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const Ajv = require('ajv');
const {
  validateManifest, resolveFrame, MouthGate, BlinkScheduler, breathing, leanFor, frameFor,
} = require('../src/sprite-logic');

const PROTOCOL = JSON.parse(fs.readFileSync(path.join(__dirname, '..', '..', 'docs', 'protocol.schema.json'), 'utf8'));
const EXPRESSIONS = PROTOCOL.expressions;
const SCHEMA = JSON.parse(fs.readFileSync(path.join(__dirname, '..', 'sprite-manifest.schema.json'), 'utf8'));
const PACK = path.join(__dirname, '..', 'placeholder-pack');
const PLACEHOLDER = JSON.parse(fs.readFileSync(path.join(PACK, 'manifest.json'), 'utf8'));

const base = () => ({
  name: 'm', version: 1, canvas: { width: 768, height: 1344 }, crops: { upper: [0, 0, 768, 700] },
  expressions: {
    neutral: { closed: 'neutral_closed.png' },
    happy: { closed: 'happy_closed.png' },
  },
});

const lcg = (seed) => {
  let s = seed;
  return () => { s = (s * 1664525 + 1013904223) % 4294967296; return s / 4294967296; };
};

test('placeholder pack validates with no warnings', () => {
  assert.deepEqual(validateManifest(PLACEHOLDER, EXPRESSIONS), []);
});

test('placeholder pack matches the manifest schema and ships every frame', () => {
  const ajv = new Ajv({ strict: true });
  const validate = ajv.compile(SCHEMA);
  assert.ok(validate(PLACEHOLDER), JSON.stringify(validate.errors));
  assert.deepEqual(PLACEHOLDER.canvas, { width: 384, height: 672 });
  assert.deepEqual(Object.keys(PLACEHOLDER.expressions), EXPRESSIONS);
  for (const variants of Object.values(PLACEHOLDER.expressions)) {
    assert.deepEqual(Object.keys(variants).sort(), ['blink', 'closed', 'open']);
    for (const file of Object.values(variants)) assert.ok(fs.existsSync(path.join(PACK, file)), file);
  }
});

test('manifest schema accepts a minimal manifest and rejects broken ones', () => {
  const validate = new Ajv({ strict: true }).compile(SCHEMA);
  assert.ok(validate(base()), JSON.stringify(validate.errors));
  const noNeutral = base();
  delete noNeutral.expressions.neutral;
  assert.equal(validate(noNeutral), false);
  const badCrop = base();
  badCrop.crops.upper = [0, 0, 768];
  assert.equal(validate(badCrop), false);
  const unknown = base();
  unknown.expressions.grumpy = { closed: 'g.png' };
  assert.equal(validate(unknown), false);
});

test('missing variants fall back', () => {
  const m = base();
  assert.equal(resolveFrame(m, 'happy', 'open'), 'happy_closed.png');
  assert.equal(resolveFrame(m, 'sad', 'blink'), 'neutral_closed.png');
  assert.equal(resolveFrame(m, 'happy', 'closed'), 'happy_closed.png');
  assert.ok(validateManifest(m, EXPRESSIONS).includes('missing happy.open'));
  assert.ok(validateManifest(m, EXPRESSIONS).includes('missing pouty.closed'));
});

test('neutral variants beat nothing when the expression lacks closed', () => {
  const m = base();
  m.expressions.neutral.open = 'neutral_open.png';
  m.expressions.sad = { blink: 'sad_blink.png' };
  assert.equal(resolveFrame(m, 'sad', 'open'), 'neutral_open.png');
  assert.equal(resolveFrame(m, 'sad', 'blink'), 'sad_blink.png');
  assert.equal(resolveFrame(m, 'not_an_expression', 'open'), 'neutral_open.png');
});

test('manifest without neutral.closed throws', () => {
  const m = base();
  delete m.expressions.neutral.closed;
  assert.throws(() => validateManifest(m, EXPRESSIONS), /neutral\.closed/);
  for (const field of ['name', 'version', 'canvas', 'crops']) {
    const broken = base();
    delete broken[field];
    assert.throws(() => validateManifest(broken, EXPRESSIONS), new RegExp(field));
  }
  const noUpper = base();
  noUpper.crops = {};
  assert.throws(() => validateManifest(noUpper, EXPRESSIONS), /crops\.upper/);
  assert.throws(() => validateManifest(null, EXPRESSIONS));
});

test('mouth gate hysteresis', () => {
  const gate = new MouthGate({ openAt: 0.3, closeAt: 0.15 });
  assert.deepEqual([0.2, 0.31, 0.2, 0.14, 0.2].map((l) => gate.update(l)), [false, true, true, false, false]);
  const dflt = new MouthGate();
  assert.deepEqual([0.2, 0.31, 0.2, 0.14, 0.2].map((l) => dflt.update(l)), [false, true, true, false, false]);
});

test('blink timing', () => {
  const sched = new BlinkScheduler({ rng: lcg(1) });
  const starts = [];
  let runStart = null;
  const runs = [];
  let prev = false;
  for (let t = 0; starts.length < 51; t += 1) {
    const closed = sched.isClosed(t);
    if (closed && !prev) { starts.push(t); runStart = t; }
    if (!closed && prev) runs.push(t - runStart);
    prev = closed;
  }
  assert.equal(starts.length, 51);
  for (let i = 1; i < starts.length; i += 1) {
    const gap = starts[i] - starts[i - 1];
    assert.ok(gap >= 3000 && gap <= 6000, `gap ${gap}`);
  }
  assert.ok(runs.length >= 50);
  for (const r of runs) assert.equal(r, 120);
});

test('blink scheduler is deterministic and survives a long pause', () => {
  const a = new BlinkScheduler({ rng: lcg(7) });
  const b = new BlinkScheduler({ rng: lcg(7) });
  for (let t = 0; t < 60000; t += 16) assert.equal(a.isClosed(t), b.isClosed(t));
  assert.equal(typeof a.isClosed(10 * 60 * 1000), 'boolean');
});

test('breathing bounds', () => {
  let sawLow = false;
  let sawHigh = false;
  for (let t = 0; t <= 8000; t += 50) {
    const { scale, dy } = breathing(t);
    assert.ok(scale >= 1 && scale <= 1.012, `scale ${scale}`);
    assert.ok(dy >= -3 && dy <= 0, `dy ${dy}`);
    if (scale < 1.001) sawLow = true;
    if (scale > 1.011) sawHigh = true;
  }
  assert.ok(sawLow && sawHigh);
  assert.deepEqual(breathing(0), breathing(4000));
});

test('leanFor leans only when focused and curious', () => {
  assert.deepEqual(leanFor({ presence: 'focused', curiosity: 0.7 }), { rotateDeg: -6, dx: -12 });
  assert.deepEqual(leanFor({ presence: 'focused', curiosity: 0.6 }), { rotateDeg: 0, dx: 0 });
  assert.deepEqual(leanFor({ presence: 'idle', curiosity: 0.9 }), { rotateDeg: 0, dx: 0 });
  assert.deepEqual(leanFor({}), { rotateDeg: 0, dx: 0 });
});

test('frameFor prefers blink then open', () => {
  const m = base();
  m.expressions.happy.open = 'happy_open.png';
  m.expressions.happy.blink = 'happy_blink.png';
  const f = (expression, mouthOpen, blinking) => frameFor({ manifest: m, expression, mouthOpen, blinking });
  assert.equal(f('happy', true, true), 'happy_blink.png');
  assert.equal(f('happy', true, false), 'happy_open.png');
  assert.equal(f('happy', false, false), 'happy_closed.png');
  assert.equal(f('happy', false, true), 'happy_blink.png');
  assert.equal(f('sad', true, true), 'neutral_closed.png');
});
