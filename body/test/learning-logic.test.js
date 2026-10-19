"use strict";
const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const Ajv = require("ajv");
const L = require("../src/learning-logic");

const fixture = (name) => JSON.parse(fs.readFileSync(path.join(__dirname, "fixtures", name), "utf8"));
const msg = fixture("core_learning.json");

test("dialRows lists the five dials in order with their bands and rounded percentages", () => {
  const rows = L.dialRows(msg);
  assert.deepEqual(rows.map((r) => r.name), ["sweetness", "teasing", "clinginess", "verbosity", "emoji"]);
  assert.deepEqual(rows.map((r) => r.label), ["Sweetness", "Teasing", "Clinginess", "How much she says", "Emoji"]);
  const teasing = rows[1];
  assert.deepEqual([teasing.value, teasing.lo, teasing.hi], [0.35, 0.1, 0.8]);
  assert.deepEqual([teasing.valuePct, teasing.loPct, teasing.hiPct], [35, 10, 80]);
  assert.deepEqual([rows[4].loPct, rows[4].hiPct], [0, 60]);
});

test("dialRows skips a missing or non-finite value, and falls back to the core's band when a band is missing", () => {
  const rows = L.dialRows({ dials: { sweetness: { value: "x", lo: 0, hi: 1 }, teasing: { value: NaN }, clinginess: null, verbosity: { value: 0.4 }, emoji: { value: 0.2, lo: 0, hi: 0.6 }, bogus: { value: 0.5 } } });
  assert.deepEqual(rows.map((r) => r.name), ["verbosity", "emoji"]);
  assert.deepEqual([rows[0].lo, rows[0].hi], [0.15, 0.7]);
  for (const bad of [null, undefined, {}, { dials: [] }, { dials: "x" }]) assert.deepEqual(L.dialRows(bad), []);
});

test("historyRows words a change as 'Teasing 0.50 → 0.35' and keeps when, source and reason", () => {
  assert.deepEqual(L.historyRows(msg), [
    { text: "Teasing 0.50 → 0.35", at: 1792327800, source: "explicit", reason: 'they said "stop teasing me"' },
  ]);
});

test("historyRows drops rows it can't word and caps at 20", () => {
  const good = { at: 1, dial: "emoji", old: 0.3, new: 0.15, reason: "r", source: "drift" };
  const rows = L.historyRows({ history: [null, "x", { ...good, dial: "mood" }, { ...good, old: "a" }, { ...good, new: null }, ...Array(25).fill(good)] });
  assert.equal(rows.length, 20);
  assert.equal(rows[0].text, "Emoji 0.30 → 0.15");
  assert.deepEqual(L.historyRows({ history: "x" }), []);
});

test("noteRows keeps integer ids with string texts and drops the rest", () => {
  const rows = L.noteRows({ notes: [{ id: 1, text: "a", polarity: "like" }, { id: 1.5, text: "b" }, { id: "2", text: "c" }, { id: 3, text: 4 }, null, { id: 4, text: "d" }] });
  assert.deepEqual(rows, [{ id: 1, text: "a", polarity: "like" }, { id: 4, text: "d", polarity: "" }]);
  assert.deepEqual(L.noteRows(msg), [{ id: 1, text: "They like it when I ask about their day.", polarity: "like" }]);
  assert.deepEqual(L.noteRows({ notes: "x" }), []);
});

test("twinRows: one row per kind in order, rate is evidence-weighted, thin buckets are hidden", () => {
  const rows = L.twinRows(msg);
  assert.deepEqual(rows.map((r) => r.kind), ["bid", "game_offer", "report", "greeting"]);
  assert.deepEqual(rows.map((r) => r.label), ["Break check-ins", "Game offers", "Agent news", "Greetings"]);
  assert.equal(rows[0].rate, 0.75);
  assert.equal(rows[0].count, 4);
  assert.equal(L.rateText(rows[0]), "75% of 4");
  assert.deepEqual(rows[0].buckets.map((b) => b.bucket), ["evening/present/short", "morning/present/short"]);
  const thin = L.twinRows({ twin: [{ kind: "report", bucket: "a", p: 0.2, n: 0.4 }, { kind: "report", bucket: "b", p: 0.9, n: 3 }, { kind: "report", bucket: "c", p: 0.5, n: 5 }] })[2];
  assert.deepEqual(thin.buckets.map((b) => b.bucket), ["c", "b"], "n descending, under 0.5 hidden");
  assert.equal(thin.count, 8, "the thin bucket still counts toward the rate and the total");
  assert.equal(thin.rate, 0.63);
  assert.equal(L.rateText(thin), "63% of 8");
});

test("twinRows with no evidence says 'no data yet'; bad cells are ignored", () => {
  for (const bad of [{}, { twin: [] }, { twin: "x" }, { twin: [{ kind: "bid", bucket: "a", p: NaN, n: 2 }, { kind: "bid", bucket: "a", p: 0.5, n: 0 }, { kind: "nope", bucket: "a", p: 0.5, n: 2 }, null] }]) {
    const rows = L.twinRows(bad);
    assert.equal(rows.length, 4);
    assert.ok(rows.every((r) => r.rate === null && r.count === 0 && r.buckets.length === 0));
    assert.equal(L.rateText(rows[0]), "no data yet");
  }
});

test("the three message builders validate against the schema, and an unknown twin kind doesn't", () => {
  const schema = JSON.parse(fs.readFileSync(path.join(__dirname, "..", "..", "docs", "protocol.schema.json"), "utf8"));
  const validate = new Ajv({ strict: false }).compile(schema.body_to_core);
  assert.deepEqual(L.noteDeleteMessage(7), { type: "note_delete_style", id: 7 });
  assert.deepEqual(L.twinResetMessage("greeting"), { type: "twin_reset", kind: "greeting" });
  assert.deepEqual(L.resetMessage(), { type: "learning_reset" });
  for (const m of [L.noteDeleteMessage(7), L.resetMessage(), ...L.TWIN_KINDS.map((k) => L.twinResetMessage(k.kind))]) {
    assert.ok(validate(m), JSON.stringify(validate.errors));
  }
  assert.equal(validate(L.twinResetMessage("nope")), false);
});

test("the learning fixture validates as a core message", () => {
  const schema = JSON.parse(fs.readFileSync(path.join(__dirname, "..", "..", "docs", "protocol.schema.json"), "utf8"));
  const validate = new Ajv({ strict: false }).compile(schema.core_to_body);
  assert.ok(validate(msg), JSON.stringify(validate.errors));
});
