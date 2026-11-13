"use strict";
const test = require("node:test");
const assert = require("node:assert/strict");
const R = require("../src/routine-logic");

const msg = (honesty, enabled = true, cells = []) => ({ type: "routine", enabled, cells, honesty });
const hon = (t, n, hr = 0, hn = 0, claims = true) => ({
  claims_true: t, claims_scored: n, hunches_right: hr, hunches_scored: hn, unverifiable: 0, expired: 0,
  claims_enabled: claims,
});

test("honestyText gives the Health row's texts", () => {
  assert.equal(R.honestyText(msg(hon(0, 0), false)), "Off: she isn't noticing your routines.");
  assert.equal(R.honestyText(null), "Off: she isn't noticing your routines.");
  assert.equal(R.honestyText(msg(hon(0, 0, 31, 52, false))), "Hunches only in this version · hunches right 31 of 52 · 30 days");
  assert.equal(R.honestyText(msg(hon(0, 0))), "No claims checked yet: she needs a few more days");
  assert.equal(R.honestyText(msg(hon(3, 3))), "3 claims checked so far (all true): she needs a few more days");
  assert.equal(R.honestyText(msg(hon(2, 3))), "3 claims checked so far (2 true): she needs a few more days");
  assert.equal(R.honestyText(msg(hon(17, 20))), "Only 17 of 20 claims held up (85%): tell Claude");
  assert.equal(R.honestyText(msg(hon(16, 19))), "Honest 16 of 19 claims (84%) · hunches right 0 of 0 · 30 days");
  assert.equal(R.honestyText(msg(hon(18, 20))), "Honest 18 of 20 claims (90%) · hunches right 0 of 0 · 30 days");
  assert.equal(R.honestyText(msg(hon(47, 48, 31, 52))), "Honest 47 of 48 claims (98%) · hunches right 31 of 52 · 30 days");
});

test("cellRows keeps the order and the text, and reads how sure she is", () => {
  const cells = [
    { line: "Agent runs (riya): about 10 min, 14 runs, sure", confidence: 0.78, n: 14 },
    { line: "Gaming on weekday evenings: 58% of days, 8 days, fairly sure", confidence: 0.6, n: 8 },
    { line: "Coming back after leaving in the morning on weekdays: usually about 3 hours, 4 times, still learning", confidence: 0.3, n: 4 },
    { line: 5 }, null,
  ];
  assert.deepEqual(R.cellRows(msg(hon(0, 0), true, cells)), [
    { line: cells[0].line, sure: "sure" },
    { line: cells[1].line, sure: "fairly sure" },
    { line: cells[2].line, sure: "still learning" },
  ]);
  assert.deepEqual(R.cellRows({}), []);
});

test("forgetMessage is the routine_forget message", () => {
  assert.deepEqual(R.forgetMessage(), { type: "routine_forget" });
});

test("settings.html has the routines row under She learns you, and settings.js wires it", () => {
  const fs = require("node:fs");
  const path = require("node:path");
  const html = fs.readFileSync(path.join(__dirname, "..", "settings.html"), "utf8");
  assert.match(html, /class="row check" data-key="routine_enabled"/);
  assert.ok(html.includes("She notices your routines"));
  assert.ok(html.includes("She keeps small private guesses about when you come back, how long your agent runs take and when you game, says only what she can back up, and checks herself. Nothing leaves this computer."));
  assert.ok(html.indexOf('id="h-learning"') < html.indexOf('data-key="routine_enabled"'), "under She learns you");
  assert.ok(html.indexOf('data-key="routine_enabled"') < html.indexOf('aria-labelledby="h-facts"'), "inside it");
  for (const id of ["routine-cells", "routine-honesty", "routine-forget"]) assert.ok(html.includes(`id="${id}"`), id);
  assert.match(html, /<button id="routine-forget" type="button" class="danger">Forget my routines<\/button>/);
  assert.ok(html.includes('<script src="src/routine-logic.js"></script>'));
  const js = fs.readFileSync(path.join(__dirname, "..", "settings.js"), "utf8");
  assert.match(js, /CHECK_KEYS = \[[^\]]*"routine_enabled"/);
  assert.ok(js.includes('title: "Forget her routines?"'));
  assert.ok(js.includes("Everything she noticed about your routines is deleted. Her honesty tally stays. This can't be undone."));
  assert.ok(js.includes('msg.type === "routine"'));
  assert.ok(!/innerHTML/.test(js.slice(js.indexOf("function renderRoutine"), js.indexOf("function dialItem"))), "textContent only");
});
